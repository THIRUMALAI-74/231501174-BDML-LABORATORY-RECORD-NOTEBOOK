"""
API Schemas
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict
from datetime import datetime


class HouseFeaturesInput(BaseModel):
    """
    Input schema for house features.
    """
    # Most important features for prediction
    overall_qual: int = Field(..., ge=1, le=10, description="Overall material and finish quality")
    gr_liv_area: int = Field(..., gt=0, description="Above grade living area in square feet")
    garage_area: int = Field(0, ge=0, description="Garage area in square feet")
    total_bsmt_sf: int = Field(0, ge=0, description="Total basement area in square feet")
    first_flr_sf: int = Field(..., gt=0, description="First floor square feet")
    year_built: int = Field(..., ge=1800, le=2025, description="Original construction year")
    year_remod_add: int = Field(..., ge=1800, le=2025, description="Remodel year")
    full_bath: int = Field(0, ge=0, le=5, description="Full bathrooms above grade")
    bedroom_abv_gr: int = Field(0, ge=0, le=10, description="Bedrooms above grade")
    kitchen_abv_gr: int = Field(1, ge=1, le=3, description="Kitchens above grade")
    totrms_abv_grd: int = Field(0, ge=0, description="Total rooms above grade")
    fireplaces: int = Field(0, ge=0, le=5, description="Number of fireplaces")
    garage_cars: int = Field(0, ge=0, le=5, description="Garage capacity in cars")
    
    # Optional features with defaults
    lot_area: Optional[int] = Field(8000, ge=0, description="Lot size in square feet")
    overall_cond: Optional[int] = Field(5, ge=1, le=10, description="Overall condition rating")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "overall_qual": 7,
                "gr_liv_area": 1500,
                "garage_area": 500,
                "total_bsmt_sf": 1000,
                "first_flr_sf": 1000,
                "year_built": 2000,
                "year_remod_add": 2010,
                "full_bath": 2,
                "bedroom_abv_gr": 3,
                "kitchen_abv_gr": 1,
                "totrms_abv_grd": 7,
                "fireplaces": 1,
                "garage_cars": 2,
                "lot_area": 10000,
                "overall_cond": 5
            }
        }
    }
    
    @validator('year_remod_add')
    def validate_remod_year(cls, v, values):
        """Ensure remodel year is not before build year."""
        if 'year_built' in values and v < values['year_built']:
            raise ValueError('Remodel year cannot be before build year')
        return v


class ConfidenceInterval(BaseModel):
    """
    Confidence interval for prediction.
    """
    lower: float = Field(..., description="Lower bound of confidence interval")
    upper: float = Field(..., description="Upper bound of confidence interval")


class PredictionResponse(BaseModel):
    """
    Response schema for prediction.
    """
    predicted_price: float = Field(..., description="Predicted house price in USD")
    confidence_interval: ConfidenceInterval = Field(..., description="95% confidence interval")
    model_used: str = Field("XGBoost", description="Model used for prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_price": 250000.50,
                "confidence_interval": {
                    "lower": 235000.00,
                    "upper": 265000.00
                },
                "model_used": "XGBoost",
                "timestamp": "2024-10-21T10:30:00"
            }
        },
        "protected_namespaces": ()
    }


class HealthResponse(BaseModel):
    """
    Response schema for health check.
    """
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field("1.0.0", description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """
    Response schema for errors.
    """
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)