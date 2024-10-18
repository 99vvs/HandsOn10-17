from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import networkx as nx

# Load models and scaler
rf_model = joblib.load('rf_model.pkl')
nn_model = load_model('nn_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define route for predictions with POST method
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Ensure the data is coming in JSON format

    features = np.array([data[key] for key in data]).reshape(1, -1)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Get predictions
    rf_pred = rf_model.predict(features_scaled)
    nn_pred = (nn_model.predict(features_scaled) > 0.5).astype(int).flatten()

    # Ensemble: Majority voting
    final_pred = np.round((rf_pred + nn_pred) / 2)[0]

    # Query knowledge graph for recommendations
    recommendation = get_recommendation(final_pred)

    return jsonify({
        'prediction': 'Diabetic' if final_pred == 1 else 'Not Diabetic',
        'recommendation': recommendation
    })

# Helper function to query the knowledge graph
def get_recommendation(prediction):
    G = nx.Graph()
    G.add_edge('Diabetic', 'Regular Exercise', recommendation='30 mins daily')
    G.add_edge('Diabetic', 'Diet Control', recommendation='Low sugar, balanced diet')
    G.add_edge('Not Diabetic', 'Maintain Weight', recommendation='Regular weight checks')

    if prediction == 1:
        return {
            'exercise': G['Diabetic']['Regular Exercise']['recommendation'],
            'diet': G['Diabetic']['Diet Control']['recommendation']
        }
    else:
        return {'recommendation': G['Not Diabetic']['Maintain Weight']['recommendation']}

if __name__ == '__main__':
    app.run(debug=True)
