import warnings
warnings.filterwarnings('ignore')

# Importing all required Python libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import os
from sklearn import svm
from sklearn.feature_selection import mutual_info_regression
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, RepeatVector, Bidirectional, LSTM, GRU, AveragePooling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory

# Create the model directory if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")

# Defining self-attention layer
class Attention(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), 
                               initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), 
                               initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return tf.reduce_sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences
        })
        return config

# Class to normalize dataset values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = MinMaxScaler(feature_range=(0, 1))

# Loading and displaying household power consumption dataset
try:
    dataset = pd.read_csv(r"household_power_consumption.csv", sep=";", nrows=10000)
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Data preprocessing
dataset.replace('?', np.nan, inplace=True)
for col in dataset.columns:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
dataset.fillna(0, inplace=True)

# Ensure required columns exist before conversion
required_columns = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for col in required_columns:
    if col not in dataset.columns:
        print(f"Error: Required column {col} not found in dataset")
        exit()

# Convert to float while handling errors
dataset[required_columns] = dataset[required_columns].apply(pd.to_numeric, errors='coerce')
dataset.fillna(0, inplace=True)

# Visualizing graph of electricity consumption
plt.figure(figsize=(6, 4))
plt.plot(dataset['Sub_metering_1'], label='Sub-Meter 1 Consumption')
plt.plot(dataset['Sub_metering_2'], label='Sub-Meter 2 Consumption')
plt.plot(dataset['Sub_metering_3'], label='Sub-Meter 3 Consumption')
plt.title("Electricity Consumption Recorded by Different Sub-Meters")
plt.xlabel("Number of Records")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.show()

# Feature engineering
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['year'] = dataset['Date'].dt.year
dataset['month'] = dataset['Date'].dt.month
dataset['day'] = dataset['Date'].dt.day
dataset['Time'] = pd.to_datetime(dataset['Time'])
dataset['hour'] = dataset['Time'].dt.hour
dataset['minute'] = dataset['Time'].dt.minute
dataset['second'] = dataset['Time'].dt.second
dataset['label'] = dataset['Sub_metering_1'] + dataset['Sub_metering_2'] + dataset['Sub_metering_3']

# Dropping unnecessary columns
dataset.drop(['Date', 'Time', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1, inplace=True)
dataset.fillna(0, inplace=True)

# Feature selection using MIC
Y = dataset['label'].ravel()
dataset.drop(['label'], axis=1, inplace=True)
columns = dataset.columns
X = dataset.values
print("Total features before MIC selection:", X.shape[1])

mic_scores = mutual_info_regression(X, Y)
mic_scores = list(zip(columns, mic_scores))
mic_scores.sort(key=lambda x: x[1], reverse=True)
top_features = [feature for feature, _ in mic_scores[:8]]
X = dataset[top_features]
print("Total features after MIC selection:", X.shape[1])

# Normalization
Y = Y.reshape(-1, 1)
X = scaler.fit_transform(X)
Y = scaler1.fit_transform(Y)
print("Normalized Features shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("\nTrain & Test Dataset Split")
print("Training records:", X_train.shape[0])
print("Test records:", X_test.shape[0])

# Metrics storage
results = {
    'Model': [],
    'R2': [],
    'RMSE': [],
    'MAE': []
}

def calculate_metrics(model_name, predictions, true_values):
    mae = mean_absolute_error(true_values, predictions)
    rmse = sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)
    
    results['Model'].append(model_name)
    results['R2'].append(r2)
    results['RMSE'].append(rmse)
    results['MAE'].append(mae)
    
    print(f"\n{model_name} Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Inverse transform for original scale
    predictions = scaler1.inverse_transform(predictions.reshape(-1, 1)).flatten()
    true_values = scaler1.inverse_transform(true_values.reshape(-1, 1)).flatten()
    
    print("\nSample Predictions:")
    for i in range(min(10, len(true_values))):
        print(f"True: {true_values[i]:.2f} | Predicted: {predictions[i]:.2f}")
    
    # Plot first 100 samples
    plt.figure(figsize=(10, 4))
    plt.plot(true_values[:100], label='True Consumption', color='blue')
    plt.plot(predictions[:100], label='Predicted Consumption', color='orange', alpha=0.7)
    plt.title(f'{model_name} Predictions')
    plt.xlabel('Samples')
    plt.ylabel('Consumption')
    plt.legend()
    plt.show()

# SVM Model
svm_model = svm.SVR()
svm_model.fit(X_train, y_train.ravel())
svm_pred = svm_model.predict(X_test)
calculate_metrics("SVM", svm_pred, y_test.ravel())

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
calculate_metrics("Linear Regression", lr_pred, y_test.ravel())

# Reshape for CNN models
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# CNN-BiLSTM Model
def create_cnn_bilstm_model(input_shape):
    model = Sequential([
        Conv2D(32, (1, 1), activation='relu', input_shape=input_shape),
        MaxPooling2D((1, 1)),
        Conv2D(32, (1, 1), activation='relu'),
        MaxPooling2D((1, 1)),
        Flatten(),
        RepeatVector(3),
        Attention(return_sequences=True),
        Bidirectional(LSTM(64, activation='relu')),
        RepeatVector(3),
        Bidirectional(LSTM(64, activation='relu')),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

cnn_bilstm_model = create_cnn_bilstm_model((X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3]))
cnn_bilstm_path = "model/cnn_bilstm_weights.keras"

if not os.path.exists(cnn_bilstm_path):
    checkpoint = ModelCheckpoint(cnn_bilstm_path, save_best_only=True, verbose=1)
    history = cnn_bilstm_model.fit(
        X_train_cnn, y_train, 
        batch_size=16, 
        epochs=50, 
        validation_data=(X_test_cnn, y_test),
        callbacks=[checkpoint],
        verbose=1
    )
    with open('model/cnn_bilstm_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
else:
    cnn_bilstm_model.load_weights(cnn_bilstm_path)

cnn_bilstm_pred = cnn_bilstm_model.predict(X_test_cnn).flatten()
calculate_metrics("CNN-BiLSTM", cnn_bilstm_pred, y_test.ravel())

# CNN-BiGRU Model
def create_cnn_bigru_model(input_shape):
    model = Sequential([
        Conv2D(32, (1, 1), activation='relu', input_shape=input_shape),
        MaxPooling2D((1, 1)),
        Conv2D(32, (1, 1), activation='relu'),
        MaxPooling2D((1, 1)),
        Flatten(),
        RepeatVector(3),
        Attention(return_sequences=True),
        Bidirectional(GRU(64, activation='relu')),
        RepeatVector(3),
        Bidirectional(GRU(64, activation='relu')),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

cnn_bigru_model = create_cnn_bigru_model((X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3]))
cnn_bigru_path = "model/cnn_bigru_weights.keras"

if not os.path.exists(cnn_bigru_path):
    checkpoint = ModelCheckpoint(cnn_bigru_path, save_best_only=True, verbose=1)
    history = cnn_bigru_model.fit(
        X_train_cnn, y_train, 
        batch_size=16, 
        epochs=50, 
        validation_data=(X_test_cnn, y_test),
        callbacks=[checkpoint],
        verbose=1
    )
    with open('model/cnn_bigru_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
else:
    cnn_bigru_model.load_weights(cnn_bigru_path)

cnn_bigru_pred = cnn_bigru_model.predict(X_test_cnn).flatten()
calculate_metrics("CNN-BiGRU", cnn_bigru_pred, y_test.ravel())

# Results comparison
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Plotting comparison
metrics = ['R2', 'RMSE', 'MAE']
for metric in metrics:
    plt.figure(figsize=(8, 4))
    plt.bar(results_df['Model'], results_df[metric])
    plt.title(f'Model Comparison by {metric}')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.show()

# Flask App
import os
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Sample in-memory storage for users
users = {}

# Route for Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Save the new user details (Here, using the 'users' dict for simplicity)
        users[username] = password
        return redirect('/login')
    return render_template('registration.html')

# Route for Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check login credentials here (using the 'users' dict for simplicity)
        if username in users and users[username] == password:
            session['username'] = username
            return redirect('/welcome')  # Redirect to the welcome page
        else:
            return "Invalid credentials", 401

    return render_template('login.html')

# Route for Welcome Page
@app.route('/welcome')
def welcome():
    # Check if user is logged in
    if 'username' in session:
        return render_template('welcome.html', username=session['username'])
    else:
        return redirect('/login')  # Redirect to login if not logged in

# Route for Prediction Page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'username' not in session:
        return redirect('/login')  # Redirect to login if not logged in

    if request.method == 'POST':
        # Perform the prediction logic here
        # Example of a hardcoded prediction result
        prediction_result = "Predicted Electricity Consumption: 250 kWh"  # Example output
        return render_template('prediction.html', prediction=prediction_result)

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
