// Prediction data loading and display

class PredictionManager {
    constructor() {
        this.predictionData = null;
        this.xrsData = null;
        this.basePath = this.getBasePath();
        this.translationManager = null;
        this.currentDate = null;
    }
    
    setTranslationManager(manager) {
        this.translationManager = manager;
        
        // Listen for language change events to refresh displays
        window.addEventListener('languageChanged', () => {
            setTimeout(() => {
                // Update with current date if available, otherwise use today
                const currentDate = this.currentDate || new Date();
                this.updatePerformanceDisplays(currentDate);
            }, 150); // Delay to ensure translation manager is updated
        });
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
            const [predResponse, xrsResponse] = await Promise.all([
                fetch(`${this.basePath}/data/pred_24.json`),
                fetch(`${this.basePath}/data/xrs.json`)
            ]);
            
            if (predResponse.ok) {
                this.predictionData = await predResponse.json();
                console.log('Prediction data loaded:', Object.keys(this.predictionData).length, 'entries');
            } else {
                console.warn('Failed to load prediction data:', predResponse.status);
                this.predictionData = {};
            }
            
            if (xrsResponse.ok) {
                this.xrsData = await xrsResponse.json();
                console.log('XRS data loaded:', Object.keys(this.xrsData).length, 'entries');
            } else {
                console.warn('Failed to load XRS data:', xrsResponse.status);
                this.xrsData = {};
            }
            
            return this.predictionData && this.xrsData;
        } catch (error) {
            console.warn('Error loading data:', error);
            this.predictionData = {};
            this.xrsData = {};
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
            return this.createUTCDateToday();
        }
        
        const keys = Object.keys(this.predictionData);
        keys.sort((a, b) => b.localeCompare(a));
        
        if (keys.length > 0) {
            const latestKey = keys[0];
            const year = parseInt(latestKey.slice(0, 4));
            const month = parseInt(latestKey.slice(4, 6)) - 1;
            const day = parseInt(latestKey.slice(6, 8));
            return new Date(Date.UTC(year, month, day));
        }
        
        return this.createUTCDateToday();
    }
    
    createUTCDateToday() {
        const now = new Date();
        return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    }
    
    displayPrediction(date, hour) {
        // Store current date for language change updates
        this.currentDate = date;
        
        if (!this.predictionData) {
            this.showLoadingPrediction();
            return;
        }
        
        // Update performance displays instead of individual predictions
        this.updatePerformanceDisplays(date);
        
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hourStr = String(hour).padStart(2, '0');
        const dataKey = `${year}${month}${day}${hourStr}`;
        
        if (this.predictionData[dataKey]) {
            const probs = this.predictionData[dataKey];
            const prediction = this.getPredictionFromProbs(probs);
            
            // Hide old prediction displays
            const resultDiv = document.getElementById('prediction-result');
            if (resultDiv) {
                resultDiv.style.display = 'none';
            }
            
            const detailsDiv = document.getElementById('prediction-details');
            if (detailsDiv) {
                detailsDiv.style.display = 'none';
            }
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
        
        const classInfo = {
            'X': { 
                status: this.translationManager ? this.translationManager.t('flare_status_major') : 'Major Flares', 
                level: this.translationManager ? this.translationManager.t('flare_level_x') : 'Lv.4 (X-Class)' 
            },
            'M': { 
                status: this.translationManager ? this.translationManager.t('flare_status_active') : 'Active', 
                level: this.translationManager ? this.translationManager.t('flare_level_m') : 'Lv.3 (M-Class)' 
            },
            'C': { 
                status: this.translationManager ? this.translationManager.t('flare_status_eruptive') : 'Eruptive', 
                level: this.translationManager ? this.translationManager.t('flare_level_c') : 'Lv.2 (C-Class)' 
            },
            'O': { 
                status: this.translationManager ? this.translationManager.t('flare_status_quiet') : 'Quiet', 
                level: this.translationManager ? this.translationManager.t('flare_level_o') : 'Lv.1 (O-Class)' 
            }
        };
        
        const classes = ['X', 'M', 'C', 'O'];
        
        probSection.innerHTML = classes.map(cls => {
            const prob = prediction.probabilities[cls] || 0;
            const isPredicted = cls === prediction.class;
            const colorClass = cls.toLowerCase() + '-class';
            const info = classInfo[cls];
            
            return `
                <div class="prob-item ${colorClass} ${isPredicted ? 'predicted' : ''}" 
                     style="padding: 0.15rem; ${isPredicted ? `--class-color: ${classColors[cls]};` : ''}">
                    <div class="prob-value" style="margin-bottom: 0.03rem; font-size: 1rem; color: #333;">${(prob * 100).toFixed(1)}%</div>
                    <div class="prob-label" style="font-size: 0.65rem; line-height: 1.2; color: #333; font-weight: 600;">
                        ${info.status}<br/>${info.level}
                    </div>
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
            ? this.translationManager.t('since_may_2025')
            : 'Since May 2025';
            
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
            'O': 'Quiet'
        };
        return descriptions[flareClass] || 'Unknown flare class';
    }
    

    
    showLoadingPrediction() {
        // Show loading state for performance displays
        this.updateMonthPerformanceDisplay(null);
        this.updateAllPerformanceDisplay(null);
        
        // Hide old displays
        const resultDiv = document.getElementById('prediction-result');
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        
        const detailsDiv = document.getElementById('prediction-details');
        if (detailsDiv) {
            detailsDiv.style.display = 'none';
        }
    }
    
    showNoPrediction() {
        // Show no data state for performance displays
        this.updateMonthPerformanceDisplay(null);
        this.updateAllPerformanceDisplay(null);
        
        // Hide old displays
        const resultDiv = document.getElementById('prediction-result');
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        
        const detailsDiv = document.getElementById('prediction-details');
        if (detailsDiv) {
            detailsDiv.style.display = 'none';
        }
    }
    
    updatePastPredictionDisplay(prediction) {
        // Determine flare level and status based on prediction
        const { level, status, statusClass, flareClass } = this.getFlareLevel(prediction);
        
        // Update level blocks
        this.updatePastLevelBlocks(level);
        
        // Update status text
        const statusElement = document.getElementById('past-flare-status');
        if (statusElement) {
            statusElement.className = `flare-status ${statusClass}`;
            statusElement.querySelector('.status-text').textContent = status;
            statusElement.querySelector('.level-text').textContent = `Lv.${level} (${flareClass})`;
        }
    }
    
    getFlareLevel(prediction) {
        // Extract probabilities
        const xProb = prediction.probabilities.X || 0;
        const mProb = prediction.probabilities.M || 0;
        const cProb = prediction.probabilities.C || 0;
        const oProb = prediction.probabilities.O || 0;
        
        // Determine the highest probability class
        const maxProb = Math.max(xProb, mProb, cProb, oProb);
        
        const majorStatus = this.translationManager ? this.translationManager.t('flare_status_major') : 'Major Flares';
        const activeStatus = this.translationManager ? this.translationManager.t('flare_status_active') : 'Active';
        const eruptiveStatus = this.translationManager ? this.translationManager.t('flare_status_eruptive') : 'Eruptive';
        const quietStatus = this.translationManager ? this.translationManager.t('flare_status_quiet') : 'Quiet';
        
        if (maxProb === xProb && xProb > 0.1) {
            return { level: 4, status: majorStatus, statusClass: 'status-major', flareClass: 'X-class' };
        } else if (maxProb === mProb && mProb > 0.05) {
            return { level: 3, status: activeStatus, statusClass: 'status-active', flareClass: 'M-class' };
        } else if (maxProb === cProb && cProb > 0.1) {
            return { level: 2, status: eruptiveStatus, statusClass: 'status-eruptive', flareClass: 'C-class' };
        } else {
            return { level: 1, status: quietStatus, statusClass: 'status-quiet', flareClass: 'O-class' };
        }
    }
    
    updatePastLevelBlocks(level) {
        const blocksContainer = document.getElementById('past-flare-level-blocks');
        if (!blocksContainer) return;
        
        // Clear existing blocks
        blocksContainer.innerHTML = '';
        blocksContainer.className = `flare-level-blocks level-${level}`;
        
        // Always show 4 blocks (from bottom to top: 1, 2, 3, 4)
        for (let i = 4; i >= 1; i--) {
            const block = document.createElement('div');
            block.className = 'level-block';
            
            // Fill blocks up to the current level
            if (i <= level) {
                block.classList.add('filled');
            }
            
            blocksContainer.appendChild(block);
        }
    }
    
    // Convert XRS flux to flare class based on the table
    getFlareClassFromFlux(flux) {
        // Handle missing data (null, undefined, 0, or negative values)
        if (flux === null || flux === undefined || flux <= 0) {
            return null; // Missing or invalid data
        }
        
        if (flux > 1e-4) return 'X';
        if (flux > 1e-5) return 'M';
        if (flux > 1e-6) return 'C';
        return 'O';
    }
    
    // Get predicted class from prediction probabilities
    getPredictedClassFromProbs(probs) {
        const classes = ['O', 'C', 'M', 'X'];
        const maxIndex = probs.indexOf(Math.max(...probs));
        return classes[maxIndex];
    }
    
    // Convert classes to binary (M+X vs C+O)
    getBinaryClass(flareClass) {
        return (flareClass === 'M' || flareClass === 'X') ? 'MX' : 'CO';
    }
    
    // Calculate actual performance for a given period
    calculatePerformanceForPeriod(currentDate, days) {
        if (!this.predictionData || !this.xrsData) return null;
        
        const endDate = new Date(currentDate);
        const calculatedStartDate = new Date(currentDate.getTime() - days * 24 * 60 * 60 * 1000);
        
        // Ensure we don't go before May 1, 2025
        const mayFirst2025 = new Date(2025, 4, 1); // May is month 4 (0-indexed)
        const startDate = calculatedStartDate > mayFirst2025 ? calculatedStartDate : mayFirst2025;
        
        let correct = 0;
        let total = 0;
        
        // Iterate through all prediction data entries
        for (const [key, probs] of Object.entries(this.predictionData)) {
            // Parse date from key (YYYYMMDDHH format)
            const year = parseInt(key.substr(0, 4));
            const month = parseInt(key.substr(4, 2));
            const day = parseInt(key.substr(6, 2));
            const hour = parseInt(key.substr(8, 2));
            
            const entryDate = new Date(year, month - 1, day, hour);
            
            // Check if entry is within the specified period
            if (entryDate >= startDate && entryDate <= endDate) {
                // Get corresponding XRS data - exclude missing data
                const xrsFlux = this.xrsData[key];
                
                // Only process if XRS data exists, is not null/undefined, and is greater than 0
                if (xrsFlux !== null && xrsFlux !== undefined && xrsFlux > 0) {
                    const actualClass = this.getFlareClassFromFlux(xrsFlux);
                    const predictedClass = this.getPredictedClassFromProbs(probs);
                    
                    // Only count if both actual and predicted classes are valid
                    if (actualClass && predictedClass) {
                        total++;
                        
                        // Convert to binary classification (M+X vs C+O)
                        const actualBinary = this.getBinaryClass(actualClass);
                        const predictedBinary = this.getBinaryClass(predictedClass);
                        
                        if (actualBinary === predictedBinary) {
                            correct++;
                        }
                    }
                }
            }
        }
        
        if (total === 0) return null;
        
        const accuracy = correct / total;
        return { accuracy, total, correct };
    }
    
    // Update performance displays for different periods
    updatePerformanceDisplays(currentDate) {
        if (!currentDate) return;
        
        // Calculate performance for different periods
        const monthPerformance = this.calculatePerformanceForPeriod(currentDate, 30);
        const allPerformance = this.calculatePerformanceForPeriod(currentDate, 365); // All available data
        
        // Update displays using the existing UI structure
        this.updateMonthPerformanceDisplay(monthPerformance);
        this.updateAllPerformanceDisplay(allPerformance);
    }
    
    // Update week performance display - unified design
    updateWeekPerformanceDisplay(performance) {
        const performanceContainer = document.querySelector('#week-performance-section .performance-display');
        if (!performanceContainer) return;
        
        // Check if dark theme is active
        const isDarkTheme = document.body.classList.contains('dark-theme');
        const primaryColor = isDarkTheme ? '#ffffff' : '#212529';
        const secondaryColor = isDarkTheme ? '#cbd5e1' : '#666';
        
        if (!performance) {
            performanceContainer.innerHTML = `
                <div style="font-size: 3rem; font-weight: 800; color: ${primaryColor}; margin-bottom: 0.5rem;">--%</div>
                <div style="font-size: 1rem; color: ${secondaryColor}; font-weight: 600;">Past Week</div>
                <div style="font-size: 0.8rem; color: ${secondaryColor}; margin-top: 0.25rem;">No data</div>
            `;
            return;
        }
        
        const accuracyPercent = (performance.accuracy * 100).toFixed(1);
        const mAccuracyText = this.translationManager ? this.translationManager.t('m_accuracy') : 'M≥ Accuracy';
        
        performanceContainer.innerHTML = `
            <div style="font-size: 3rem; font-weight: 800; color: ${primaryColor}; margin-bottom: 0.5rem;">${accuracyPercent}%</div>
            <div style="font-size: 1rem; color: ${secondaryColor}; font-weight: 600;">${mAccuracyText}</div>
            <div style="font-size: 0.8rem; color: ${secondaryColor}; margin-top: 0.25rem;">${performance.total} ${this.translationManager ? this.translationManager.t('predictions') : 'predictions'}</div>
        `;
    }
    
    // Update month performance display - unified design
    updateMonthPerformanceDisplay(performance) {
        const performanceContainer = document.querySelector('#month-performance-section .performance-display');
        if (!performanceContainer) return;
        
        // Check if dark theme is active
        const isDarkTheme = document.body.classList.contains('dark-theme');
        const primaryColor = isDarkTheme ? '#ffffff' : '#212529';
        const secondaryColor = isDarkTheme ? '#cbd5e1' : '#666';
        
        if (!performance) {
            performanceContainer.innerHTML = `
                <div style="font-size: 3rem; font-weight: 800; color: ${primaryColor}; margin-bottom: 0.5rem;">--%</div>
                <div style="font-size: 1rem; color: ${secondaryColor}; font-weight: 600;">Past Month</div>
                <div style="font-size: 0.8rem; color: ${secondaryColor}; margin-top: 0.25rem;">No data</div>
            `;
            return;
        }
        
        const accuracyPercent = (performance.accuracy * 100).toFixed(1);
        const mAccuracyText = this.translationManager ? this.translationManager.t('m_accuracy') : 'M≥ Accuracy';
        
        performanceContainer.innerHTML = `
            <div style="font-size: 3rem; font-weight: 800; color: ${primaryColor}; margin-bottom: 0.5rem;">${accuracyPercent}%</div>
            <div style="font-size: 1rem; color: ${secondaryColor}; font-weight: 600;">${mAccuracyText}</div>
            <div style="font-size: 0.8rem; color: ${secondaryColor}; margin-top: 0.25rem;">${performance.total} ${this.translationManager ? this.translationManager.t('predictions') : 'predictions'}</div>
        `;
    }
    
    // Update all period performance display - unified design
    updateAllPerformanceDisplay(performance) {
        const performanceContainer = document.querySelector('#all-performance-section .performance-display');
        if (!performanceContainer) return;
        
        // Check if dark theme is active
        const isDarkTheme = document.body.classList.contains('dark-theme');
        const primaryColor = isDarkTheme ? '#ffffff' : '#212529';
        const secondaryColor = isDarkTheme ? '#cbd5e1' : '#666';
        
        const mAccuracyText = this.translationManager ? this.translationManager.t('m_accuracy') : 'M≥ Accuracy';
        const sinceMayText = this.translationManager ? this.translationManager.t('since_may_2025') : 'Since May 2025';
        
        if (!performance) {
            performanceContainer.innerHTML = `
                <div style="font-size: 3rem; font-weight: 800; color: ${primaryColor}; margin-bottom: 0.5rem;">--%</div>
                <div style="font-size: 1rem; color: ${secondaryColor}; font-weight: 600;">${mAccuracyText}</div>
                <div style="font-size: 0.8rem; color: ${secondaryColor}; margin-top: 0.25rem;">${sinceMayText}</div>
            `;
            return;
        }
        
        const accuracyPercent = (performance.accuracy * 100).toFixed(1);
        
        performanceContainer.innerHTML = `
            <div style="font-size: 3rem; font-weight: 800; color: ${primaryColor}; margin-bottom: 0.5rem;">${accuracyPercent}%</div>
            <div style="font-size: 1rem; color: ${secondaryColor}; font-weight: 600;">${mAccuracyText}</div>
            <div style="font-size: 0.8rem; color: ${secondaryColor}; margin-top: 0.25rem;">${sinceMayText}</div>
        `;
    }
}

// Export for use in other modules
window.PredictionManager = PredictionManager;
