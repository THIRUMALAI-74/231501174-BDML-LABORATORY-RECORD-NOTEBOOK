// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        overall_qual: parseInt(document.getElementById('overall_qual').value),
        gr_liv_area: parseInt(document.getElementById('gr_liv_area').value),
        first_flr_sf: parseInt(document.getElementById('first_flr_sf').value),
        garage_area: parseInt(document.getElementById('garage_area').value),
        garage_cars: parseInt(document.getElementById('garage_cars').value),
        total_bsmt_sf: parseInt(document.getElementById('total_bsmt_sf').value),
        year_built: parseInt(document.getElementById('year_built').value),
        year_remod_add: parseInt(document.getElementById('year_remod_add').value),
        full_bath: parseInt(document.getElementById('full_bath').value),
        bedroom_abv_gr: parseInt(document.getElementById('bedroom_abv_gr').value),
        kitchen_abv_gr: parseInt(document.getElementById('kitchen_abv_gr').value),
        totrms_abv_grd: parseInt(document.getElementById('totrms_abv_grd').value),
        fireplaces: parseInt(document.getElementById('fireplaces').value),
        lot_area: parseInt(document.getElementById('lot_area').value),
        overall_cond: parseInt(document.getElementById('overall_cond').value)
    };
    
    // Validate data
    if (formData.year_remod_add < formData.year_built) {
        showError('Remodel year cannot be before build year');
        return;
    }
    
    // Show loading
    showLoading();
    
    try {
        // Make API request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        // Hide loading
        hideLoading();
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
});

// Display prediction results
function displayResults(result) {
    // Format price
    const formattedPrice = formatCurrency(result.predicted_price);
    const formattedLower = formatCurrency(result.confidence_interval.lower);
    const formattedUpper = formatCurrency(result.confidence_interval.upper);
    
    // Update DOM
    document.getElementById('predictedPrice').textContent = formattedPrice;
    document.getElementById('lowerBound').textContent = formattedLower;
    document.getElementById('upperBound').textContent = formattedUpper;
    document.getElementById('modelUsed').textContent = result.model_used;
    document.getElementById('timestamp').textContent = formatTimestamp(result.timestamp);
    
    // Hide form, show results
    document.querySelector('.form-card').style.display = 'none';
    document.getElementById('resultCard').style.display = 'block';
    
    // Scroll to results
    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
}

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Show loading spinner
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

// Hide loading spinner
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    document.getElementById('errorText').textContent = message;
    errorDiv.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        closeError();
    }, 5000);
}

// Close error message
function closeError() {
    document.getElementById('errorMessage').style.display = 'none';
}

// Show form (after viewing results)
function showForm() {
    document.querySelector('.form-card').style.display = 'block';
    document.getElementById('resultCard').style.display = 'none';
    document.querySelector('.form-card').scrollIntoView({ behavior: 'smooth' });
}

// Reset form
function resetForm() {
    document.getElementById('predictionForm').reset();
    
    // Set default values
    document.getElementById('overall_qual').value = 7;
    document.getElementById('gr_liv_area').value = 1500;
    document.getElementById('first_flr_sf').value = 1000;
    document.getElementById('garage_area').value = 500;
    document.getElementById('garage_cars').value = 2;
    document.getElementById('total_bsmt_sf').value = 1000;
    document.getElementById('year_built').value = 2000;
    document.getElementById('year_remod_add').value = 2010;
    document.getElementById('full_bath').value = 2;
    document.getElementById('bedroom_abv_gr').value = 3;
    document.getElementById('kitchen_abv_gr').value = 1;
    document.getElementById('totrms_abv_grd').value = 7;
    document.getElementById('fireplaces').value = 1;
    document.getElementById('lot_area').value = 10000;
    document.getElementById('overall_cond').value = 5;
}

// Check API health on page load
window.addEventListener('load', async function() {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        if (!health.model_loaded) {
            showError('Warning: Model not loaded. Predictions may not work.');
        }
        
        console.log('API Status:', health);
    } catch (error) {
        console.error('Health check failed:', error);
    }
});

// Input validation helpers
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
        // Remove invalid class if value is valid
        if (this.checkValidity()) {
            this.classList.remove('invalid');
        } else {
            this.classList.add('invalid');
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('predictionForm');
        if (form.style.display !== 'none') {
            form.dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to close error
    if (e.key === 'Escape') {
        closeError();
    }
});