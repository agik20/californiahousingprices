document.addEventListener('DOMContentLoaded', function() {
    const predictForm = document.getElementById('predictForm');
    const resultContainer = document.getElementById('result');
    
    predictForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        resultContainer.innerHTML = `
            <div class="result-value">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Calculating...</p>
            </div>
        `;
        
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());
        
        // Convert string to number
        Object.keys(data).forEach(key => {
            data[key] = parseFloat(data[key]);
        });
        
        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(data)
            });
            
            if (!res.ok) {
                throw new Error(`Server returned ${res.status}`);
            }
            
            const result = await res.json();
            
            // Format the price with commas
            const formattedPrice = new Intl.NumberFormat('en-US').format(result.predicted_price);
            
            // Display result with animation
            resultContainer.innerHTML = `
                <div class="result-value fade-in">
                    <h3>ESTIMATED PROPERTY VALUE</h3>
                    <p>$${formattedPrice}</p>
                </div>
            `;
            
        } catch (error) {
            console.error('Prediction error:', error);
            resultContainer.innerHTML = `
                <div class="result-value">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error calculating prediction</p>
                    <p class="text-sm mt-2">Please try again later</p>
                </div>
            `;
        }
    });
    
    // Add input validation
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.validity.valid) {
                this.classList.remove('border-red-500');
            } else {
                this.classList.add('border-red-500');
            }
        });
    });
});