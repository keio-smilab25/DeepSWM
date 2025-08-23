// Prediction data loading and display

class PredictionManager {
    constructor() {
        this.predictionData = null;
        this.basePath = this.getBasePath();
        this.translationManager = null;
    }
    
    setTranslationManager(manager) {
        this.translationManager = manager;
    }
    
    getBasePath() {
        if (window.location.hostname.includes('github.io')) {
            const pathSegments = window.location.pathname.split('/').filter(Boolean);
            if (pathSegments.length > 0) {
                return '/' + pathSegments[0];
            }
        }
        return '';
    }
    
    async loadPredictionData() {
        try {
            const response = await fetch(`${this.basePath}/data/pred_24.json`);
            if (response.ok) {
                this.predictionData = await response.json();
                console.log('Prediction data loaded:', Object.keys(this.predictionData).length, 'entries');
                return true;
            } else {
                console.warn('Failed to load prediction data:', response.status);
                this.predictionData = {};
                return false;
            }
        } catch (error) {
            console.warn('Error loading prediction data:', error);
            this.predictionData = {};
            return false;
        }
    }
    
    hasDataForDate(dateObj) {
        if (!this.predictionData) return false;
        
        const year = dateObj.getFullYear();
        const month = (dateObj.getMonth() + 1).toString().padStart(2, '0');
        const day = dateObj.getDate().toString().padStart(2, '0');
        
        // Check if any hour has data for this date
        for (let hour = 0; hour < 24; hour++) {
            const dataKey = `${year}${month}${day}${hour.toString().padStart(2, '0')}`;
            if (this.predictionData[dataKey]) {
                return true;
            }
        }
        
        return false;
    }
    
    getLatestAvailableDate() {
        if (!this.predictionData || Object.keys(this.predictionData).length === 0) {
            return new Date();
        }
        
        const keys = Object.keys(this.predictionData);
        keys.sort((a, b) => b.localeCompare(a));
        
        if (keys.length > 0) {
            const latestKey = keys[0];
            const year = parseInt(latestKey.slice(0, 4));
            const month = parseInt(latestKey.slice(4, 6)) - 1;
            const day = parseInt(latestKey.slice(6, 8));
            return new Date(year, month, day);
        }
        
        return new Date();
    }
    
    displayPrediction(date, hour) {
        if (!this.predictionData) {
            this.showLoadingPrediction();
            return;
        }
        
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hourStr = String(hour).padStart(2, '0');
        const dataKey = `${year}${month}${day}${hourStr}`;
        
        if (this.predictionData[dataKey]) {
            const probs = this.predictionData[dataKey];
            const prediction = this.getPredictionFromProbs(probs);
            this.updatePredictionDisplay(prediction);
        } else {
            this.showNoPrediction();
        }
    }
    
    getPredictionFromProbs(probs) {
        const classes = ['O', 'C', 'M', 'X'];
        const maxIndex = probs.indexOf(Math.max(...probs));
        const confidence = Math.max(...probs);
        
        return {
            class: classes[maxIndex],
            confidence: confidence,
            probabilities: {
                O: probs[0],
                C: probs[1],
                M: probs[2],
                X: probs[3]
            }
        };
    }
    
    updatePredictionDisplay(prediction) {
        const resultDiv = document.getElementById('prediction-result');
        if (!resultDiv) return;
        
        const classNames = ['loading', 'x-class', 'm-class', 'c-class', 'o-class'];
        
        // Remove all class names
        classNames.forEach(cls => resultDiv.classList.remove(cls));
        
        // Add appropriate class
        resultDiv.classList.add(`${prediction.class.toLowerCase()}-class`);
        
        const description = this.translationManager 
            ? this.translationManager.t(`flare_desc_${prediction.class.toLowerCase()}`)
            : this.getFlareDescription(prediction.class);
        
        const confidenceText = this.translationManager 
            ? this.translationManager.t('confidence')
            : 'Confidence';
        
        resultDiv.innerHTML = `
            <div class="flare-class" style="margin-bottom: 0.00rem; font-size: 1.75rem;">${prediction.class}-Class</div>
            <div class="flare-description" style="margin-bottom: 0.00rem; font-size: 0.85rem;">${description}</div>
            <div class="confidence" style="font-size: 1rem;">${confidenceText}: ${(prediction.confidence * 100).toFixed(1)}%</div>
        `;
        
        // Apply inline styling to the result div itself
        resultDiv.style.padding = '0.2rem';
        
        // Update prediction details separately with color coding
        const detailsDiv = document.getElementById('prediction-details');
        if (detailsDiv) {
            // Apply inline styling to the details div itself
            detailsDiv.style.gap = '0.15rem';
            
            const classColors = {
                'X': '#ff6b6b',
                'M': '#ffa726', 
                'C': '#81c784',
                'O': '#4caf50'
            };
            
            detailsDiv.innerHTML = Object.entries(prediction.probabilities).map(([cls, prob]) => {
                const isActive = cls === prediction.class;
                const bgColor = isActive ? classColors[cls] : '#f8f9fa';
                const textColor = isActive ? '#fff' : '#333';
                const borderColor = classColors[cls];
                
                return `
                    <div class="detail-item ${isActive ? 'active' : ''}" 
                         style="background: ${bgColor}; color: ${textColor}; border: 2px solid ${borderColor}; padding: 0.2rem; font-size: 0.875rem;">
                        ${cls}: ${(prob * 100).toFixed(1)}%
                    </div>
                `;
            }).join('');
        }
        
        // Update probabilities section
        this.updateProbabilitiesSection(prediction);
        
        // Update binary classification
        this.updateBinaryClassification(prediction);
    }
    
    updateProbabilitiesSection(prediction) {
        const probSection = document.getElementById('probabilities-section');
        if (!probSection) return;
        
        // Apply inline styling to the section itself
        probSection.style.gap = '0.15rem';
        
        const classColors = {
            'X': '#ff6b6b',
            'M': '#ffa726', 
            'C': '#81c784',
            'O': '#4caf50'
        };
        
        const classes = ['X', 'M', 'C', 'O'];
        
        probSection.innerHTML = classes.map(cls => {
            const prob = prediction.probabilities[cls] || 0;
            const isPredicted = cls === prediction.class;
            const colorClass = cls.toLowerCase() + '-class';
            
            return `
                <div class="prob-item ${colorClass} ${isPredicted ? 'predicted' : ''}" 
                     style="padding: 0.15rem; ${isPredicted ? `--class-color: ${classColors[cls]};` : ''}">
                    <div class="prob-value" style="margin-bottom: 0.03rem; font-size: 1rem;">${(prob * 100).toFixed(1)}%</div>
                    <div class="prob-label" style="font-size: 0.75rem;">${cls}-Class</div>
                </div>
            `;
        }).join('');
    }
    
    updateBinaryClassification(prediction) {
        const binarySection = document.getElementById('binary-section');
        if (!binarySection) return;
        
        // Calculate binary classification (M+X vs O+C)
        const significantFlareProb = (prediction.probabilities.M || 0) + (prediction.probabilities.X || 0);
        const binaryAccuracy = 87.3; // This could be dynamically calculated or fetched
        
        const binaryLabel = this.translationManager 
            ? this.translationManager.t('classification')
            : 'O+C vs M+X';
            
        const sincePeriod = this.translationManager 
            ? this.translationManager.t('since_april_2025')
            : 'Since April 2025';
            
        // Apply inline styling to the binary section itself
        binarySection.style.gap = '0.3rem';
        
        binarySection.innerHTML = `
            <div class="binary-item">
                <div class="binary-value">${binaryAccuracy.toFixed(1)}%</div>
                <div class="binary-label">${binaryLabel}</div>
                <div class="binary-period">${sincePeriod}</div>
            </div>
        `;
    }
    
    getFlareDescription(flareClass) {
        const descriptions = {
            'X': 'Major solar flare - significant impact',
            'M': 'Moderate solar flare - possible effects', 
            'C': 'Minor solar flare - minimal space weather impact',
            'O': 'No significant solar flare activity'
        };
        return descriptions[flareClass] || 'Unknown flare class';
    }
    

    
    showLoadingPrediction() {
        const resultDiv = document.getElementById('prediction-result');
        if (!resultDiv) return;
        
        resultDiv.className = 'prediction-result loading';
        const loadingText = this.translationManager 
            ? this.translationManager.t('loading_prediction')
            : 'Loading prediction...';
            
        resultDiv.innerHTML = `
            <div class="flare-class"><span class="loading-spinner"></span></div>
            <div class="flare-description">${loadingText}</div>
            <div class="confidence">Confidence: --%</div>
        `;
    }
    
    showNoPrediction() {
        const resultDiv = document.getElementById('prediction-result');
        if (!resultDiv) return;
        
        resultDiv.className = 'prediction-result loading';
        const noDataText = this.translationManager 
            ? this.translationManager.t('no_prediction')
            : 'No prediction data available for this date';
            
        resultDiv.innerHTML = `
            <div class="flare-class">--</div>
            <div class="flare-description">${noDataText}</div>
            <div class="confidence">Confidence: --%</div>
        `;
    }
}

// Export for use in other modules
window.PredictionManager = PredictionManager;
