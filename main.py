"""
FastAPI Application for House Price Prediction

This FastAPI application loads a pre-trained machine learning model
and provides an endpoint to make predications on house prices.
To run the application, use the following command:
1. Install the required packages: `pip install fastapi uvicorn scikit-learn pandas numpy joblib
2. Run the server: `uvicorn main:app --reload --host 0.0.0.0 --port
3. Access the API documentation at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using a trained Linear Regression model",
    version="1.0.0",
)

# Load the trained model and scaler
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    print("Model and scaler loaded sucessfully!")
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    print("Please ensure you have run the training notebook and saved the model files.")

# Define the input data model
class HouseFeatures(BaseModel):
    """
    Input features for house price prediction.

    Features from California Housing dataset:
    - "MedInc":  Median income in block group
    "HouseAge":  Median house age in block group
    "AveRooms":  Average number of rooms per household
    "AveBedrms": Average number of bedrooms per household
    "Population":  Population of block group
    "AveOccup":  Average household size
    "Latitude":   Latitude of block group
    "Longitude":  Longitude of block group
    """

    MedInc: float 
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float   
    Longitude: float 

# Define the output model
class PredictionResponse(BaseModel):
    # Response model for predictions
    predicted_price: float
    input_feature: dict

@app.get("/", response_class = HTMLResponse)
async def root():
    """Root endpoint with a user-friebdly HTML page"""
    return"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Presiction API</title>
        <style>
            body{
                font-family: Arial, sans-serif;
                backgroung: #f9f9f9;
                margin: 0;
                padding: 0;
                }
"""

@app.post("/predict", response_model = PredictionResponse)
async def predict_house_prices(features: HouseFeatures):
    try:
        # Convert input to numpy array
        input_data = np.array([[
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude   
        ]])

        # scale the input data using the saved scaler 
        input_scaled = scaler.transform(input_data)

        # Make prediction
        