import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages (2 = warnings/errors only)

from flask import Flask, request, render_template, Response
import time
import logging
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Comprehensive mock data for all 26 Andhra Pradesh districts
AP_DATA = {
    "Alluri Sitharama Raju": {"groundwater_levels": {"2015": 5.0, "2019": 6.0, "2021": 4.5, "2025": 4.8}, "soil_type": "Laterite", "suitable_crops": ["Paddy", "Cashew", "Coffee"], "trend": -0.1},
    "Anakapalli": {"groundwater_levels": {"2015": 6.5, "2019": 7.0, "2021": 5.5, "2025": 5.8}, "soil_type": "Red Loamy", "suitable_crops": ["Paddy", "Sugarcane", "Banana"], "trend": -0.2},
    "Anantapur": {"groundwater_levels": {"2015": 10.0, "2019": 12.5, "2021": 8.5, "2025": 8.5}, "soil_type": "Red Sandy", "suitable_crops": ["Groundnut", "Cotton", "Maize"], "trend": -0.5},
    "Annamayya": {"groundwater_levels": {"2015": 7.0, "2019": 8.0, "2021": 6.5, "2025": 6.8}, "soil_type": "Red Loamy", "suitable_crops": ["Mango", "Rice", "Groundnut"], "trend": -0.3},
    "Bapatla": {"groundwater_levels": {"2015": 4.0, "2019": 5.0, "2021": 3.8, "2025": 4.2}, "soil_type": "Black Cotton", "suitable_crops": ["Paddy", "Chilli", "Cotton"], "trend": -0.1},
    "Chittoor": {"groundwater_levels": {"2015": 6.5, "2019": 8.0, "2021": 6.0, "2025": 6.0}, "soil_type": "Red Loamy", "suitable_crops": ["Sugarcane", "Rice", "Mango"], "trend": -0.3},
    "East Godavari": {"groundwater_levels": {"2015": 3.5, "2019": 4.0, "2021": 3.2, "2025": 3.5}, "soil_type": "Alluvial", "suitable_crops": ["Paddy", "Coconut", "Banana"], "trend": -0.1},
    "Eluru": {"groundwater_levels": {"2015": 4.5, "2019": 5.5, "2021": 4.0, "2025": 4.3}, "soil_type": "Alluvial", "suitable_crops": ["Paddy", "Sugarcane", "Oil Palm"], "trend": -0.2},
    "Guntur": {"groundwater_levels": {"2015": 7.5, "2019": 9.0, "2021": 3.0, "2025": 4.2}, "soil_type": "Black Cotton", "suitable_crops": ["Paddy", "Chilli", "Tobacco"], "trend": -0.2},
    "Kakinada": {"groundwater_levels": {"2015": 3.8, "2019": 4.5, "2021": 3.5, "2025": 3.7}, "soil_type": "Alluvial", "suitable_crops": ["Paddy", "Coconut", "Cashew"], "trend": -0.1},
    "Konaseema": {"groundwater_levels": {"2015": 3.0, "2019": 3.8, "2021": 3.2, "2025": 3.4}, "soil_type": "Alluvial", "suitable_crops": ["Paddy", "Coconut", "Banana"], "trend": -0.1},
    "Krishna": {"groundwater_levels": {"2015": 5.0, "2019": 7.0, "2021": 3.5, "2025": 4.0}, "soil_type": "Alluvial", "suitable_crops": ["Paddy", "Sugarcane", "Banana"], "trend": -0.1},
    "Kurnool": {"groundwater_levels": {"2015": 9.0, "2019": 11.0, "2021": 8.0, "2025": 8.5}, "soil_type": "Black Cotton", "suitable_crops": ["Cotton", "Sunflower", "Groundnut"], "trend": -0.4},
    "Nandyal": {"groundwater_levels": {"2015": 8.5, "2019": 10.0, "2021": 7.5, "2025": 8.0}, "soil_type": "Red Sandy", "suitable_crops": ["Groundnut", "Cotton", "Sorghum"], "trend": -0.3},
    "Nellore": {"groundwater_levels": {"2015": 5.5, "2019": 6.5, "2021": 5.0, "2025": 5.2}, "soil_type": "Red Loamy", "suitable_crops": ["Paddy", "Shrimp", "Lemon"], "trend": -0.2},
    "Palnadu": {"groundwater_levels": {"2015": 6.0, "2019": 7.5, "2021": 5.5, "2025": 6.0}, "soil_type": "Black Cotton", "suitable_crops": ["Paddy", "Cotton", "Chilli"], "trend": -0.2},
    "Parvathipuram Manyam": {"groundwater_levels": {"2015": 4.8, "2019": 5.5, "2021": 4.2, "2025": 4.5}, "soil_type": "Laterite", "suitable_crops": ["Cashew", "Coffee", "Paddy"], "trend": -0.1},
    "Prakasam": {"groundwater_levels": {"2015": 9.0, "2019": 10.6, "2021": 8.8, "2025": 10.0}, "soil_type": "Red Loam", "suitable_crops": ["Tobacco", "Cotton", "Sunflower"], "trend": -0.4},
    "Sri Sathya Sai": {"groundwater_levels": {"2015": 8.0, "2019": 9.5, "2021": 7.0, "2025": 7.5}, "soil_type": "Red Sandy", "suitable_crops": ["Groundnut", "Maize", "Cotton"], "trend": -0.3},
    "Srikakulam": {"groundwater_levels": {"2015": 4.0, "2019": 5.0, "2021": 3.8, "2025": 4.0}, "soil_type": "Alluvial", "suitable_crops": ["Paddy", "Coconut", "Cashew"], "trend": -0.1},
    "Tirupati": {"groundwater_levels": {"2015": 6.0, "2019": 7.0, "2021": 5.5, "2025": 5.8}, "soil_type": "Red Loamy", "suitable_crops": ["Rice", "Sugarcane", "Mango"], "trend": -0.2},
    "Visakhapatnam": {"groundwater_levels": {"2015": 4.5, "2019": 5.2, "2021": 4.0, "2025": 4.2}, "soil_type": "Laterite", "suitable_crops": ["Paddy", "Cashew", "Vegetables"], "trend": -0.1},
    "Vizianagaram": {"groundwater_levels": {"2015": 5.0, "2019": 6.0, "2021": 4.5, "2025": 4.8}, "soil_type": "Red Loamy", "suitable_crops": ["Paddy", "Sugarcane", "Cashew"], "trend": -0.2},
    "West Godavari": {"groundwater_levels": {"2015": 3.5, "2019": 4.5, "2021": 3.2, "2025": 3.8}, "soil_type": "Alluvial", "suitable_crops": ["Paddy", "Coconut", "Oil Palm"], "trend": -0.1},
    "YSR Kadapa": {"groundwater_levels": {"2015": 7.5, "2019": 9.0, "2021": 6.5, "2025": 7.0}, "soil_type": "Red Sandy", "suitable_crops": ["Groundnut", "Cotton", "Banana"], "trend": -0.3}
}

# Hybrid Ensemble Model (HEM) for prediction
class HybridEnsembleModel:
    def __init__(self):
        self.gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        self.scaler = MinMaxScaler()
        self.rnn = self._build_rnn_model()

    def _build_rnn_model(self):
        model = Sequential([
            Input(shape=(4, 1)),  # Explicitly define input shape with Input layer
            LSTM(50, activation='relu', return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, district):
        years = sorted([int(y) for y in AP_DATA[district]["groundwater_levels"].keys()])
        levels = [AP_DATA[district]["groundwater_levels"][str(y)] for y in years]
        X = np.array([years]).T  # Feature: years
        y = np.array(levels)     # Target: groundwater levels
        return X, y

    def train(self, district):
        X, y = self.prepare_data(district)
        # Train GBM
        self.gbm.fit(X, y)
        # Prepare RNN data (last 4 time steps)
        scaled_y = self.scaler.fit_transform(y.reshape(-1, 1))
        X_rnn = np.array([scaled_y[-4:]]).reshape(1, 4, 1)  # Shape: (1, timesteps, features)
        y_rnn = scaled_y[-1]  # Last value as target
        self.rnn.fit(X_rnn, y_rnn.reshape(1, -1), epochs=10, verbose=0)

    def predict(self, district, years_ahead):
        X, y = self.prepare_data(district)
        # GBM prediction
        future_year = 2025 + years_ahead
        gbm_pred = self.gbm.predict([[future_year]])[0]
        # RNN prediction (using last 4 time steps)
        scaled_y = self.scaler.transform(y.reshape(-1, 1))
        last_4 = scaled_y[-4:].reshape(1, 4, 1)
        rnn_pred_scaled = self.rnn.predict(last_4, verbose=0)[0][0]
        rnn_pred = self.scaler.inverse_transform([[rnn_pred_scaled]])[0][0]
        # Ensemble: Weighted average (60% GBM, 40% RNN)
        ensemble_pred = 0.6 * gbm_pred + 0.4 * rnn_pred
        return max(ensemble_pred, 0)

# Initialize HEM models for each district
hem_models = {}
for district in AP_DATA:
    hem = HybridEnsembleModel()
    hem.train(district)
    hem_models[district] = hem

# Prediction function using HEM
def predict_groundwater(district, years_ahead):
    if district not in AP_DATA:
        return "Data not available for this district."
    return hem_models[district].predict(district, years_ahead)

# Enhanced response function
def get_local_response(question):
    question = question.lower().strip()

    # Extract district name from question
    district = None
    for d in AP_DATA:
        d_lower = d.lower()
        if d_lower in question or d_lower.replace(" ", "") in question:
            district = d
            break

    # Handle Andhra Pradesh-specific queries first
    if district:
        if "groundwater" in question or "water level" in question:
            year = None
            for y in ["2015", "2019", "2021", "2025"]:
                if y in question:
                    year = y
                    break
            if year:
                level = AP_DATA[district]["groundwater_levels"].get(year, "Data not available")
                return f"The groundwater level in {district} in {year} was {level} meters below ground."
            elif "predict" in question or "future" in question:
                years_ahead = 2  # Default
                if "3 years" in question:
                    years_ahead = 3
                future_level = predict_groundwater(district, years_ahead)
                return f"The predicted groundwater level in {district} in {years_ahead} years is {future_level:.2f} meters."
            else:  # Default to current year for partial questions
                level = AP_DATA[district]["groundwater_levels"]["2025"]
                return f"The current groundwater level in {district} is {level} meters below ground."
        elif "soil" in question:
            soil = AP_DATA[district]["soil_type"]
            return f"The soil type in {district} is {soil}."
        elif "crops" in question and ("suitable" in question or "grown" in question or "can be" in question):
            crops = ", ".join(AP_DATA[district]["suitable_crops"])
            return f"Crops that can be grown in {district}: {crops}."
        elif "best crops" in question or "predict crops" in question:
            crops = AP_DATA[district]["suitable_crops"]
            return f"The best crop for {district} in coming years is {crops[0]}, based on soil and water trends."

    # Handle greetings and general queries only if no district is detected
    if "hi" in question or "hello" in question:
        return "Greetings, citizen! I’m ARSHU'S Chatbot, your cyberpunk assistant. How can I assist you today?"
    elif "how are you" in question:
        return "I’m doing great, thanks for asking! How can I help you today?"
    elif "who are you" in question:
        return "I’m ARSHU'S Chatbot, a cyberpunk-themed assistant built by xAI, here to assist with Andhra Pradesh data and more."

    # Default response for unrecognized queries
    return "I’m not sure I understand. Could you please ask about Andhra Pradesh groundwater, soil, or crops? I’m here to help!"

# Stream response in chunks
def stream_response(question):
    full_response = get_local_response(question)
    for chunk in full_response.split():
        yield f"{chunk} "
        time.sleep(0.1)

# API endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    logger.debug(f"Received message: {user_message}")
    return Response(stream_response(user_message), mimetype='text/plain')

# Serve frontend
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)