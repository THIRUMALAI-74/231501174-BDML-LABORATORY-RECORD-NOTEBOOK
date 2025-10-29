"""
FastAPI Application
Main API endpoint definitions.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from datetime import datetime
import joblib
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    HouseFeaturesInput,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
    ConfidenceInterval
)

# Import prediction preprocessing
try:
    from src.predict_preprocess import preprocess_for_prediction, get_simple_prediction_input
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False
    print("⚠ Warning: predict_preprocess module not found")

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Global model variable
model = None
MODEL_PATH = "models/trained_model.pkl"


@app.on_event("startup")
async def load_model():
    """
    Load the trained model on startup.
    """
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"⚠ Warning: Model file not found at {MODEL_PATH}")
            print("  API will start but predictions will fail until model is available")
    except Exception as e:
        print(f"✗ Error loading model: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Serve the main web interface.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and model availability.
    """
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeaturesInput):
    """
    Predict house price based on input features.
    
    Args:
        features: House features for prediction
        
    Returns:
        Predicted price with confidence interval
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        # Convert input to DataFrame
        input_dict = features.dict()
        
        # Create a DataFrame with expected column names (matching training data)
        # Map API field names to model feature names
        model_input = pd.DataFrame([{
            'OverallQual': input_dict['overall_qual'],
            'GrLivArea': input_dict['gr_liv_area'],
            'GarageArea': input_dict['garage_area'],
            'TotalBsmtSF': input_dict['total_bsmt_sf'],
            '1stFlrSF': input_dict['first_flr_sf'],
            'YearBuilt': input_dict['year_built'],
            'YearRemodAdd': input_dict['year_remod_add'],
            'FullBath': input_dict['full_bath'],
            'BedroomAbvGr': input_dict['bedroom_abv_gr'],
            'KitchenAbvGr': input_dict['kitchen_abv_gr'],
            'TotRmsAbvGrd': input_dict['totrms_abv_grd'],
            'Fireplaces': input_dict['fireplaces'],
            'GarageCars': input_dict['garage_cars'],
            'LotArea': input_dict.get('lot_area', 8000),
            'OverallCond': input_dict.get('overall_cond', 5)
        }])
        
        # Get the expected feature names from the model (if available)
        try:
            expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        except:
            expected_features = None
        
        # If we know the expected features, create a matching DataFrame
        if expected_features is not None:
            # Create empty DataFrame with all expected features
            full_input = pd.DataFrame(0, index=[0], columns=expected_features)
            
            # Fill in the values we have
            for col in model_input.columns:
                if col in full_input.columns:
                    full_input[col] = model_input[col].values[0]
            
            input_df = full_input
        else:
            # Fallback: use the input as-is
            input_df = model_input
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Calculate confidence interval (±10%)
        margin = prediction * 0.10
        lower_bound = max(0, prediction - margin)
        upper_bound = prediction + margin
        
        # Prepare response
        response = PredictionResponse(
            predicted_price=float(prediction),
            confidence_interval=ConfidenceInterval(
                lower=float(lower_bound),
                upper=float(upper_bound)
            ),
            model_used="Ensemble Model",
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        # Log the error for debugging
        import traceback
        print("="*80)
        print("PREDICTION ERROR:")
        print(traceback.format_exc())
        print("="*80)
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(features_list: list[HouseFeaturesInput]):
    """
    Predict prices for multiple houses.
    
    Args:
        features_list: List of house features
        
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        # Convert inputs to DataFrame
        input_dicts = [f.dict() for f in features_list]
        input_df = pd.DataFrame(input_dicts)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Prepare responses
        responses = []
        for pred in predictions:
            margin = pred * 0.10
            responses.append({
                "predicted_price": float(pred),
                "confidence_interval": {
                    "lower": float(max(0, pred - margin)),
                    "upper": float(pred + margin)
                }
            })
        
        return {"predictions": responses, "count": len(responses)}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    General exception handler.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)