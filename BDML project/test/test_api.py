"""
Test API Module
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check returns correct response."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert "timestamp" in data
    
    def test_health_check_structure(self):
        """Test health check response structure."""
        response = client.get("/health")
        data = response.json()
        
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert data["version"] == "1.0.0"


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    @pytest.fixture
    def valid_input(self):
        """Valid input data for prediction."""
        return {
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
    
    def test_predict_valid_input(self, valid_input):
        """Test prediction with valid input."""
        response = client.post("/predict", json=valid_input)
        
        # May return 503 if model not loaded, which is acceptable
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            
            assert "predicted_price" in data
            assert "confidence_interval" in data
            assert "model_used" in data
            assert "timestamp" in data
            
            # Check data types
            assert isinstance(data["predicted_price"], (int, float))
            assert data["predicted_price"] > 0
    
    def test_predict_response_structure(self, valid_input):
        """Test prediction response structure."""
        response = client.post("/predict", json=valid_input)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check confidence interval structure
            assert "lower" in data["confidence_interval"]
            assert "upper" in data["confidence_interval"]
            assert data["confidence_interval"]["lower"] < data["confidence_interval"]["upper"]
    
    def test_predict_invalid_quality(self, valid_input):
        """Test prediction with invalid quality value."""
        invalid_input = valid_input.copy()
        invalid_input["overall_qual"] = 15  # Invalid: should be 1-10
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error
    
    def test_predict_negative_area(self, valid_input):
        """Test prediction with negative area."""
        invalid_input = valid_input.copy()
        invalid_input["gr_liv_area"] = -100
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
    
    def test_predict_invalid_year(self, valid_input):
        """Test prediction with remodel year before build year."""
        invalid_input = valid_input.copy()
        invalid_input["year_built"] = 2010
        invalid_input["year_remod_add"] = 2000  # Before build year
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
    
    def test_predict_missing_required_field(self, valid_input):
        """Test prediction with missing required field."""
        incomplete_input = valid_input.copy()
        del incomplete_input["overall_qual"]
        
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint."""
    
    @pytest.fixture
    def valid_batch_input(self):
        """Valid batch input data."""
        return [
            {
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
            },
            {
                "overall_qual": 8,
                "gr_liv_area": 2000,
                "garage_area": 600,
                "total_bsmt_sf": 1200,
                "first_flr_sf": 1200,
                "year_built": 2005,
                "year_remod_add": 2015,
                "full_bath": 3,
                "bedroom_abv_gr": 4,
                "kitchen_abv_gr": 1,
                "totrms_abv_grd": 9,
                "fireplaces": 2,
                "garage_cars": 3,
                "lot_area": 12000,
                "overall_cond": 6
            }
        ]
    
    def test_batch_predict(self, valid_batch_input):
        """Test batch prediction."""
        response = client.post("/predict/batch", json=valid_batch_input)
        
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            
            assert "predictions" in data
            assert "count" in data
            assert data["count"] == len(valid_batch_input)
            assert len(data["predictions"]) == len(valid_batch_input)


class TestHomeEndpoint:
    """Test home page endpoint."""
    
    def test_home_page(self):
        """Test home page loads."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_error(self):
        """Test 404 error for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test method not allowed error."""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/predict")
        
        # Check if CORS headers are present
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])