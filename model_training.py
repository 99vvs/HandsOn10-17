import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import joblib

# Load the diabetes dataset
df = pd.read_csv('C:/Users/venka/Documents/SEM 3/DS PRO/HandsOn10-17/modified_diabetes_dataset1.csv')


# Split dataset into features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Build a neural network model
nn_model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train the neural network
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the neural network
_, nn_accuracy = nn_model.evaluate(X_test_scaled, y_test)
print(f"Neural Network Test Accuracy: {nn_accuracy * 100:.2f}%")

# Save the trained neural network
nn_model.save('nn_model.h5')

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the Random Forest model
joblib.dump(rf_model, 'rf_model.pkl')

# Ensemble predictions: Majority voting
rf_preds = rf_model.predict(X_test_scaled)
nn_preds = (nn_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
ensemble_preds = np.round((rf_preds + nn_preds) / 2)

# Evaluate the ensemble model
print(f"Ensemble Model Accuracy: {accuracy_score(y_test, ensemble_preds):.2f}")
print("Classification Report:\n", classification_report(y_test, ensemble_preds))
try:
    joblib.dump(rf_model, 'rf_model.pkl')
    nn_model.save('nn_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    print("Models and scaler saved successfully!")
except Exception as e:
    print(f"Error saving models: {e}")

