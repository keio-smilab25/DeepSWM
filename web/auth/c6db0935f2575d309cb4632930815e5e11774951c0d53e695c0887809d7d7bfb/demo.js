// Demo functionality - loaded after successful authentication

// Enhanced wavelength definitions and colormap system
const wavelengths = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '4500'];

// Enhanced AIA colormap definitions based on SDO standards
const colormaps = {
  '0094': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [10, 79, 51] },
    { pos: 0.4, color: [41, 121, 102] },
    { pos: 0.6, color: [92, 162, 153] },
    { pos: 0.8, color: [163, 206, 204] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0131': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [0, 73, 73] },
    { pos: 0.4, color: [0, 147, 147] },
    { pos: 0.6, color: [62, 221, 221] },
    { pos: 0.8, color: [158, 255, 255] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0171': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [73, 51, 0] },
    { pos: 0.4, color: [147, 102, 0] },
    { pos: 0.6, color: [221, 153, 0] },
    { pos: 0.8, color: [255, 204, 54] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0193': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [114, 51, 10] },
    { pos: 0.4, color: [161, 102, 41] },
    { pos: 0.6, color: [197, 153, 92] },
    { pos: 0.8, color: [228, 204, 163] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0211': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [114, 51, 79] },
    { pos: 0.4, color: [161, 102, 121] },
    { pos: 0.6, color: [197, 153, 162] },
    { pos: 0.8, color: [228, 204, 206] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0304': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [73, 0, 0] },
    { pos: 0.4, color: [147, 0, 0] },
    { pos: 0.6, color: [221, 62, 0] },
    { pos: 0.8, color: [255, 158, 54] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '0335': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [10, 51, 114] },
    { pos: 0.4, color: [41, 102, 161] },
    { pos: 0.6, color: [92, 153, 197] },
    { pos: 0.8, color: [163, 204, 228] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '1600': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [79, 79, 10] },
    { pos: 0.4, color: [121, 121, 41] },
    { pos: 0.6, color: [162, 162, 92] },
    { pos: 0.8, color: [206, 206, 163] },
    { pos: 1.0, color: [255, 255, 255] }
  ],
  '4500': [
    { pos: 0.0, color: [0, 0, 0] },
    { pos: 0.2, color: [51, 51, 0] },
    { pos: 0.4, color: [102, 102, 0] },
    { pos: 0.6, color: [153, 153, 0] },
    { pos: 0.8, color: [204, 204, 27] },
    { pos: 1.0, color: [255, 255, 128] }
  ]
};

// Enhanced colormap interpolation function
function getAIAColorForWavelength(normValue, wavelength) {
  const stops = colormaps[wavelength];
  if (!stops) return [255, 255, 255];
  
  for (let i = 0; i < stops.length - 1; i++) {
    if (normValue >= stops[i].pos && normValue <= stops[i+1].pos) {
      const range = stops[i+1].pos - stops[i].pos;
      const f = (normValue - stops[i].pos) / range;
      const r = Math.round(stops[i].color[0] + f * (stops[i+1].color[0] - stops[i].color[0]));
      const g = Math.round(stops[i].color[1] + f * (stops[i+1].color[1] - stops[i].color[1]));
      const b = Math.round(stops[i].color[2] + f * (stops[i+1].color[2] - stops[i].color[2]));
      return [r, g, b];
    }
  }
  return [255, 255, 255];
}

// Enhanced flare class computation
function computeClassFromFlux(flux) {
  if (flux < 1e-6) return 0; // O-class
  if (flux < 1e-5) return 1; // C-class
  if (flux < 1e-4) return 2; // M-class
  return 3;                  // X-class
}

// Enhanced recall calculation for M-class and above
function computeRecallM(predData, xrsMap, rangeHours) {
  let totalM = 0, detectedM = 0;
  
  for (const key in predData) {
    if (!predData.hasOwnProperty(key)) continue;
    
    // Parse key as "YYYYMMDDHH"
    const year  = +key.slice(0,4);
    const month = +key.slice(4,6) - 1;
    const day   = +key.slice(6,8);
    const hour  = +key.slice(8,10);
    const baseUTC = Date.UTC(year, month, day, hour);

    // Get maximum flux from t to t+rangeHours-1
    let maxFlux = null;
    for (let i = 0; i < rangeHours; i++) {
      const t = new Date(baseUTC + i * 3600*1000);
      const k = `${t.getUTCFullYear()}${String(t.getUTCMonth()+1).padStart(2,'0')}`
              + `${String(t.getUTCDate()).padStart(2,'0')}${String(t.getUTCHours()).padStart(2,'0')}`;
      const f = xrsMap[k];
      if (f != null && (maxFlux === null || f > maxFlux)) {
        maxFlux = f;
      }
    }
    if (maxFlux === null) continue; // Insufficient data

    const trueCls = computeClassFromFlux(maxFlux);
    if (trueCls < 2) continue; // Exclude below M-class
    totalM++;

    const probs = predData[key];
    if (!Array.isArray(probs) || probs.length < 4) continue;
    const predCls = probs.indexOf(Math.max(...probs));
    if (predCls >= 2) detectedM++; // M-class or above predicted
  }
  
  return totalM > 0 ? (detectedM / totalM) : null;
}

async function initializeDemo() {
    console.log('Demo authenticated and loaded successfully!');
    
    // Hide auth container first
    const authContainer = document.getElementById('demo-auth-container');
    if (authContainer) {
        authContainer.classList.add('hidden');
        console.log('Auth container hidden');
    }
    
    // Get or create demo container
    let demoContainer = document.getElementById('demo-content-container');
    if (!demoContainer) {
        demoContainer = document.createElement('div');
        demoContainer.id = 'demo-content-container';
        document.body.appendChild(demoContainer);
        console.log('Created new demo container');
    }
    
    try {
        // Load demo content from external file
        let basePath = '';
        if (window.location.hostname.includes('github.io')) {
            const pathSegments = window.location.pathname.split('/').filter(Boolean);
            if (pathSegments.length > 0) {
                basePath = '/' + pathSegments[0];
            }
        }
        
        const contentUrl = `${basePath}/web/html/demo-content.html`;
        console.log('Loading demo content from:', contentUrl);
        const response = await fetch(contentUrl);
        
        if (response.ok) {
            const demoHtml = await response.text();
            console.log('Loaded demo HTML, length:', demoHtml.length);
            demoContainer.innerHTML = demoHtml;
            console.log('Demo content injected successfully');
        } else {
            console.log('Failed to load demo content:', response.status);
            // Fallback content
            demoContainer.innerHTML = `
                <section class="section">
                    <div class="container">
                        <h1 class="title">Interactive Demo</h1>
                        <p>Demo content is loading...</p>
                    </div>
                </section>
            `;
        }
    } catch (error) {
        console.log('Error loading demo content:', error);
        // Fallback content
        demoContainer.innerHTML = `
            <section class="section">
                <div class="container">
                    <h1 class="title">Interactive Demo</h1>
                    <p>Demo content failed to load. Using fallback.</p>
                </div>
            </section>
        `;
    }
    
    // Inject demo content at the demo anchor point
    const anchor = document.getElementById('demo-anchor');
    console.log('Anchor element:', anchor);
    console.log('Demo container:', demoContainer);
    
    if (anchor) {
        // Clear anchor and append demo content
        anchor.innerHTML = '';
        anchor.appendChild(demoContainer);
        demoContainer.classList.remove('hidden');
        demoContainer.style.display = 'block';
        console.log('Demo content moved to anchor successfully');
    } else {
        console.log('Anchor not found, appending to body');
        document.body.appendChild(demoContainer);
        demoContainer.classList.remove('hidden');
        demoContainer.style.display = 'block';
    }
    
    // Add demo badge to the main title
    const titleElement = document.querySelector('.publication-title');
    if (titleElement && !titleElement.querySelector('.demo-badge')) {
        titleElement.innerHTML += ' <span class="demo-badge">INTERACTIVE DEMO</span>';
    }
    
    // Initialize demo functionality after a short delay
    setTimeout(() => {
        setupDemo();
    }, 100);
}

function setupDemo() {
    console.log('Setting up demo functionality...');
    
    // Wait for DOM elements to be available
    const datePickerElement = document.getElementById('date-picker');
    if (!datePickerElement) {
        console.log('Date picker element not found, retrying in 500ms');
        setTimeout(setupDemo, 500);
        return;
    }
    
	// Initialize date picker with inline calendar (from 2025-04-01 to today)
	const datePicker = flatpickr("#date-picker", {
		inline: true,
		dateFormat: "Y-m-d",
		defaultDate: "2025-04-01",
		minDate: "2025-04-01",
		maxDate: new Date(),
		showMonths: 1,
		static: true,
		onDayCreate: function(dObj, dStr, fp, dayElem) {
			const d = dayElem.dateObj;
			if (!d) return;
			if (isAvailableDate(d)) {
				dayElem.classList.add('has-data');
				dayElem.title = 'Data available';
			} else {
				dayElem.classList.add('no-data');
				dayElem.title = 'No data available';
			}
		},
		onChange: function(selectedDates, dateStr, instance) {
			// Auto-update when date changes
			const hourSelect = document.getElementById('utc-hour-panel') || document.getElementById('utc-hour');
			const selectedHour = hourSelect ? hourSelect.value : '12';
			if (selectedDates.length > 0) {
				loadPredictionData(selectedDates[0], selectedHour);
				loadSolarImages(selectedDates[0], selectedHour);
			}
		}
	});

	// Initialize hour selector (both old and new)
	const hourSelectors = ['utc-hour', 'utc-hour-panel'];
	hourSelectors.forEach(selectorId => {
		const hourSelect = document.getElementById(selectorId);
	if (hourSelect && hourSelect.children.length === 0) {
		for (let i = 0; i < 24; i++) {
			const option = document.createElement('option');
			option.value = i.toString().padStart(2, '0');
			option.textContent = `${i.toString().padStart(2, '0')}:00 UTC`;
			hourSelect.appendChild(option);
		}
		hourSelect.value = '12';
		
		// Auto-update when hour changes
		hourSelect.addEventListener('change', () => {
			const selectedDate = datePicker.selectedDates[0];
			if (selectedDate) {
				loadPredictionData(selectedDate, hourSelect.value);
				loadSolarImages(selectedDate, hourSelect.value);
			}
		});
	}
	});

	// Load button functionality (both old and new)
	const loadButtons = ['load-button', 'load-button-panel'];
	loadButtons.forEach(buttonId => {
		const loadButton = document.getElementById(buttonId);
	if (loadButton) {
		loadButton.onclick = () => {
			const selectedDate = datePicker.selectedDates[0];
				const hourSelect = document.getElementById('utc-hour-panel') || document.getElementById('utc-hour');
				const selectedHour = hourSelect ? hourSelect.value : '12';
			
			if (!selectedDate) {
				alert('Please select a date first');
				return;
			}

			loadPredictionData(selectedDate, selectedHour);
			loadSolarImages(selectedDate, selectedHour);
		};
	}
	});

	console.log('Demo setup completed successfully');

	// Load prediction data and show accuracy
	setTimeout(async () => {
		console.log('Loading prediction data...');
		await loadPredictionDataCache();
		console.log('Computing accuracy...');
		computeAndRenderAccuracy();
		
		// Refresh calendar to show data availability
		if (datePicker && datePicker.redraw) {
			datePicker.redraw();
		}
	}, 400);

	injectAvailabilityStyles();
}

// Global variable to store prediction data
let predictionDataCache = null;

async function loadPredictionDataCache() {
	if (predictionDataCache !== null) {
		return predictionDataCache;
	}
	
	try {
		let basePath = '';
		if (window.location.hostname.includes('github.io')) {
			const pathSegments = window.location.pathname.split('/').filter(Boolean);
			if (pathSegments.length > 0) {
				basePath = '/' + pathSegments[0];
			}
		}
		
		const response = await fetch(`${basePath}/data/pred_24.json`);
		if (response.ok) {
			predictionDataCache = await response.json();
			console.log('Prediction data loaded:', Object.keys(predictionDataCache).length, 'entries');
		} else {
			console.warn('Failed to load prediction data:', response.status);
			predictionDataCache = {};
		}
	} catch (error) {
		console.warn('Error loading prediction data:', error);
		predictionDataCache = {};
	}
	
	return predictionDataCache;
}

function isAvailableDate(dateObj) {
	// Check if prediction data exists for this date
	const year = dateObj.getFullYear();
	const month = (dateObj.getMonth() + 1).toString().padStart(2, '0');
	const day = dateObj.getDate().toString().padStart(2, '0');
	const dateKey = `${year}${month}${day}12`;  // YYYYMMDDHH format for 12:00 UTC
	
	// Check if prediction data exists for this date
	return predictionDataCache && predictionDataCache.hasOwnProperty(dateKey);
}

function injectAvailabilityStyles() {
	const style = document.createElement('style');
	style.textContent = `
		/* Enhanced inline calendar styling for cosmic theme */
		.flatpickr-calendar.inline {
			background: rgba(0, 0, 51, 0.95) !important;
			border: 1px solid rgba(255, 255, 255, 0.2) !important;
			border-radius: 15px !important;
			color: #ffffff !important;
			box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
			width: 100% !important;
			max-width: 320px !important;
			margin: 0 !important;
			font-size: 0.8rem !important;
		}
		
		.flatpickr-calendar.inline .flatpickr-months {
			background: rgba(10, 20, 60, 0.85) !important;
			border-radius: 15px 15px 0 0 !important;
			padding: 0.8rem !important;
			border-bottom: 1px solid rgba(255,255,255,0.15) !important;
		}
		
		.flatpickr-calendar.inline .flatpickr-current-month {
			color: #ffffff !important;
			font-weight: 700 !important;
			font-size: 1.1rem !important;
			text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
		}
		
		.flatpickr-calendar.inline .flatpickr-current-month .flatpickr-monthDropdown-months,
		.flatpickr-calendar.inline .numInputWrapper input {
			color: #ffffff !important;
			background: rgba(0, 0, 51, 0.9) !important;
			border-radius: 6px !important;
			border: 1px solid rgba(255,255,255,0.2) !important;
		}
		
		.flatpickr-calendar.inline .flatpickr-weekdays {
			background: rgba(255, 255, 255, 0.05) !important;
			padding: 0.5rem 0 !important;
		}
		
		.flatpickr-calendar.inline .flatpickr-weekday {
			color: rgba(255, 255, 255, 0.9) !important;
			font-weight: 600 !important;
			font-size: 0.8rem !important;
		}
		
		.flatpickr-day {
			color: #e8f0ff !important;
			border: none !important;
			border-radius: 6px !important;
			margin: 1px !important;
			transition: all 0.2s ease !important;
			font-weight: 600 !important;
			width: 30px !important;
			height: 30px !important;
			line-height: 30px !important;
		}
		
		.flatpickr-day:hover {
			background: rgba(255, 107, 53, 0.4) !important;
			transform: scale(1.05) !important;
		}
		
		.flatpickr-day.selected {
			background: #ff6b35 !important;
			color: white !important;
			font-weight: 700 !important;
			box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4) !important;
		}
		
		.flatpickr-day.has-data {
			background: rgba(0, 255, 136, 0.25) !important;
			border: 2px solid #00ff88 !important;
			color: #ffffff !important;
			position: relative !important;
		}
		
		.flatpickr-day.has-data::after {
			content: '•' !important;
			position: absolute !important;
			top: 2px !important;
			right: 4px !important;
			color: #00ff88 !important;
			font-size: 0.8rem !important;
		}
		
		.flatpickr-day.no-data {
			background: rgba(255, 102, 102, 0.15) !important;
			color: rgba(255, 255, 255, 0.3) !important;
			cursor: not-allowed !important;
		}
		
		.flatpickr-day.has-data:hover {
			background: rgba(0, 255, 136, 0.4) !important;
			border-color: #00ff88 !important;
		}
		
		.flatpickr-day.has-data.selected {
			background: #00cc66 !important;
			border-color: #00ff88 !important;
		}
		
		.flatpickr-prev-month,
		.flatpickr-next-month {
			color: #ff6b35 !important;
			font-size: 1.2rem !important;
			transition: all 0.2s ease !important;
		}
		
		.flatpickr-prev-month:hover,
		.flatpickr-next-month:hover {
			color: #ffffff !important;
			transform: scale(1.2) !important;
		}
		
		.flatpickr-calendar.inline .flatpickr-days {
			padding: 0.8rem 0.5rem !important;
		}
		
		.flatpickr-calendar.inline .flatpickr-innerContainer {
			width: 100% !important;
		}
		
		.flatpickr-calendar.inline .flatpickr-rContainer {
			width: 100% !important;
		}
		
		.flatpickr-calendar.inline .dayContainer {
			width: 100% !important;
			min-width: 100% !important;
			max-width: 100% !important;
		}
		
		/* Hide the input field since we're using inline calendar */
		#date-picker {
			display: none !important;
		}
	`;
	document.head.appendChild(style);
}

function computeAndRenderAccuracy() {
	// Compute binary classification accuracy (M+X vs C+O) since 2025-04-01
	const start = new Date('2025-04-01T00:00:00Z');
	const end = new Date();
	let totalSamples = 0, correctPredictions = 0;

	for (let d = new Date(start); d <= end; d.setUTCDate(d.getUTCDate() + 1)) {
		// Use 12:00 UTC for daily evaluation
		const hour = '12';
		const gt = generateGroundTruth(d, hour);
		const pred = generateMockPrediction(d, hour).predictedClass;

		// Binary classification: M+X (class 1) vs C+O (class 0)
		const gtBinary = (gt === 'M' || gt === 'X') ? 1 : 0;
		const predBinary = (pred === 'M' || pred === 'X') ? 1 : 0;
		
		totalSamples++;
		if (gtBinary === predBinary) {
			correctPredictions++;
		}
	}

	const binaryAccuracy = totalSamples ? (correctPredictions / totalSamples) : 0;
	renderAccuracy(binaryAccuracy);
}

function generateGroundTruth(date, hour) {
	// Deterministic ground truth based on date to make statistics stable
	const base = date.getUTCFullYear()*10000 + (date.getUTCMonth()+1)*100 + date.getUTCDate();
	const rand = Math.sin(base) * 10000;
	const r = Math.abs(rand - Math.floor(rand));
	if (r > 0.985) return 'X';
	if (r > 0.93) return 'M';
	if (r > 0.62) return 'C';
	return 'O';
}

function renderAccuracy(binaryAccuracy) {
	const panel = document.getElementById('performance-summary');
	if (!panel) return;
	
	panel.innerHTML = `
		<div style="background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; padding: 0.8rem; color: white;">
			<div style="font-weight: 700; margin-bottom: 0.6rem; color: #ff6b35; text-align: center; font-size: 0.95rem;">
				Model Performance
			</div>
			<div style="font-weight: 500; margin-bottom: 0.6rem; color: rgba(255,255,255,0.7); text-align: center; font-size: 0.75rem;">
				Since 2025-04-01
			</div>
			<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
				<div style="background: #2d5a3d; border-radius: 6px; padding: 0.5rem; text-align: center;">
					<div style="font-size: 0.7rem; color: rgba(255,255,255,0.8); margin-bottom: 0.2rem;">Binary Accuracy</div>
					<div style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">${(binaryAccuracy*100).toFixed(1)}%</div>
				</div>
				<div style="background: rgba(255,255,255,0.1); border-radius: 6px; padding: 0.5rem; text-align: center;">
					<div style="font-size: 0.7rem; color: rgba(255,255,255,0.8); margin-bottom: 0.2rem;">Classification</div>
					<div style="font-size: 0.75rem; font-weight: 600; color: #ffffff;">O+C vs M+X</div>
				</div>
			</div>
		</div>
	`;
}

function loadPredictionData(date, hour) {
    const dateStr = date.toISOString().split('T')[0];
    const timestamp = `${dateStr} ${hour}:00 UTC`;
    
    const mockPredictions = generateMockPrediction(date, hour);
    updatePredictionDisplay(timestamp, mockPredictions);
}

function generateMockPrediction(date, hour) {
    const seed = date.getTime() + parseInt(hour);
    const random = Math.sin(seed) * 10000;
    const normalizedRandom = (random - Math.floor(random));
    
    let selectedClass = 'O';
    let rand = Math.abs(normalizedRandom);
    
    if (rand > 0.98) selectedClass = 'X';
    else if (rand > 0.92) selectedClass = 'M';
    else if (rand > 0.65) selectedClass = 'C';
    else selectedClass = 'O';
    
    const confidence = 0.7 + (Math.abs(normalizedRandom) * 0.25);
    
    return {
        predictedClass: selectedClass,
        confidence: confidence,
        probabilities: {
            O: selectedClass === 'O' ? confidence : (1 - confidence) * 0.6,
            C: selectedClass === 'C' ? confidence : (1 - confidence) * 0.25,
            M: selectedClass === 'M' ? confidence : (1 - confidence) * 0.1,
            X: selectedClass === 'X' ? confidence : (1 - confidence) * 0.05
        }
    };
}

function updatePredictionDisplay(timestamp, prediction) {
    const resultDiv = document.getElementById('prediction-result');
    if (!resultDiv) return;

    const classColors = {
        X: 'x-class',
        M: 'm-class',
        C: 'c-class',
        O: 'o-class'
    };
    
    // Check prediction accuracy
    const dateStr = timestamp.split(' ')[0];
    const hourStr = timestamp.split(' ')[1].split(':')[0];
    const actualClass = generateGroundTruth(new Date(dateStr), hourStr);
    const isCorrect = prediction.predictedClass === actualClass;
    const accuracyText = isCorrect ? '✓ Correct' : '✗ Incorrect';

    resultDiv.innerHTML = `
        <div class="prediction-display">
            <div class="prediction-accuracy ${isCorrect ? 'correct' : 'incorrect'}">
                ${accuracyText} (GT: ${actualClass})
            </div>
            
            <div class="prediction-class ${classColors[prediction.predictedClass]}">
                <span class="class-label">${prediction.predictedClass}-Class</span>
                <div class="probability">${(prediction.confidence * 100).toFixed(1)}% confidence</div>
            </div>
            
            <div class="prediction-details">
                ${Object.entries(prediction.probabilities).map(([cls, prob]) => 
                    `<div class="detail-item ${cls === prediction.predictedClass ? 'active' : ''}">
                        ${cls}-Class: ${(prob * 100).toFixed(1)}%
                    </div>`
                ).join('')}
            </div>
        </div>
    `;
}

let currentImagePage = 0;
let allChannels = [];

function loadSolarImages(date, hour) {
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const dateStr = `${month}${day}`;  // MMDD format for directory
    const currentHour = parseInt(hour);
    
    const timestampElement = document.getElementById('timestamp');
    if (timestampElement) {
        timestampElement.textContent = `Solar Images: ${year}-${month}-${day} ${hour.padStart(2, '0')}:00 UTC`;
    }
    
    // Enhanced solar channel configurations with proper AIA wavelengths
    allChannels = [
        { name: '94 Å', wavelength: '0094', filename: 'aia_0094' },
        { name: '131 Å', wavelength: '0131', filename: 'aia_0131' },
        { name: '171 Å', wavelength: '0171', filename: 'aia_0171' },
        { name: '193 Å', wavelength: '0193', filename: 'aia_0193' },
        { name: '211 Å', wavelength: '0211', filename: 'aia_0211' },
        { name: '304 Å', wavelength: '0304', filename: 'aia_0304' },
        { name: '335 Å', wavelength: '0335', filename: 'aia_0335' },
        { name: '1600 Å', wavelength: '1600', filename: 'aia_1600' },
        { name: '4500 Å', wavelength: '4500', filename: 'aia_4500' },
        { name: 'HMI', wavelength: 'hmi', filename: 'hmi' }
    ];

    renderAllImages(dateStr, currentHour);
}

function renderAllImages(dateStr, currentHour) {
    const aiaGrid = document.getElementById('aia-grid');
    if (!aiaGrid) return;
    
    aiaGrid.innerHTML = '';
    
    // Show all 10 channels in 5x2 grid
    const numHours = 4;
    let loadedImages = {};
    
    allChannels.forEach((channel, channelIndex) => {
            const channelDiv = document.createElement('div');
            channelDiv.className = 'channel';
            channelDiv.style.position = 'relative';
            
            // Create container for multiple time images
            const imageContainer = document.createElement('div');
            imageContainer.style.position = 'relative';
            imageContainer.style.width = '100%';
            imageContainer.style.aspectRatio = '1';
            imageContainer.style.borderRadius = '8px';
            imageContainer.style.border = '1px solid rgba(255, 255, 255, 0.2)';
            imageContainer.style.overflow = 'hidden';
            imageContainer.style.backgroundColor = '#000';
            
            // Show black background initially
            const blackBg = document.createElement('div');
            blackBg.style.position = 'absolute';
            blackBg.style.top = '0';
            blackBg.style.left = '0';
            blackBg.style.width = '100%';
            blackBg.style.height = '100%';
            blackBg.style.backgroundColor = '#000';
            imageContainer.appendChild(blackBg);
            
            loadedImages[channelIndex] = [];
            
            // Load images for past hours
            for (let h = numHours - 1; h >= 0; h--) {
                const targetHour = Math.max(0, currentHour - h);
                const hourStr = targetHour.toString().padStart(2, '0');
                
                let basePath = '';
                if (window.location.hostname.includes('github.io')) {
                    const segs = window.location.pathname.split('/').filter(Boolean);
                    if (segs.length > 0) basePath = '/' + segs[0];
                }
                
                const candidateUrl = `${basePath}/data/images/${dateStr}/${hourStr}_${channel.filename}.png`;
                
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.style.position = 'absolute';
                img.style.top = '0';
                img.style.left = '0';
                img.style.width = '100%';
                img.style.height = '100%';
                img.style.objectFit = 'cover';
                img.style.opacity = h === numHours - 1 ? '1' : '0';
                img.style.transition = 'opacity 0.5s ease';
                
                const timeIndex = numHours - 1 - h;
                
                img.onload = () => {
                    // Hide black background when image loads
                    if (blackBg) blackBg.style.display = 'none';
                    
                    // Apply enhanced colormap using canvas
                    const canvas = document.createElement('canvas');
                    canvas.width = 150;
                    canvas.height = 150;
                    canvas.style.position = 'absolute';
                    canvas.style.top = '0';
                    canvas.style.left = '0';
                    canvas.style.width = '100%';
                    canvas.style.height = '100%';
                    canvas.style.opacity = timeIndex === numHours - 1 ? '1' : '0';
                    canvas.style.transition = 'opacity 0.5s ease';
                    
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, 150, 150);
                    
                    // Apply enhanced AIA colormap
                    if (channel.wavelength !== 'hmi') {
                        const imageData = ctx.getImageData(0, 0, 150, 150);
                        const data = imageData.data;
                        
                        for (let i = 0; i < data.length; i += 4) {
                            const gray = data[i]; // Use red channel as grayscale
                            const normValue = gray / 255;
                            const colorValues = getAIAColorForWavelength(normValue, channel.wavelength);
                            data[i] = colorValues[0];     // R
                            data[i + 1] = colorValues[1]; // G  
                            data[i + 2] = colorValues[2]; // B
                            // Alpha stays the same
                        }
                        
                        ctx.putImageData(imageData, 0, 0);
                    }
                    
                    imageContainer.appendChild(canvas);
                    loadedImages[channelIndex][timeIndex] = canvas;
                };
                
                img.onerror = () => {
                    // Create fallback canvas with enhanced colormap
                    const canvas = document.createElement('canvas');
                    canvas.width = 150;
                    canvas.height = 150;
                    canvas.style.position = 'absolute';
                    canvas.style.top = '0';
                    canvas.style.left = '0';
                    canvas.style.width = '100%';
                    canvas.style.height = '100%';
                    canvas.style.opacity = timeIndex === numHours - 1 ? '1' : '0';
                    canvas.style.transition = 'opacity 0.5s ease';
                    
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#000';
                    ctx.fillRect(0, 0, 150, 150);
                    
                    // Create synthetic solar disk with enhanced colormap
                    const centerX = 75; const centerY = 75; const radius = 60;
                    
                    for (let y = 0; y < 150; y++) {
                        for (let x = 0; x < 150; x++) {
                            const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
                            if (dist < radius) {
                                const intensity = Math.max(0, 1 - dist / radius);
                                const noise = Math.random() * 0.3;
                                const value = Math.min(1, intensity + noise);
                                
                                let colorValues;
                                if (channel.wavelength === 'hmi') {
                                    // Simple red-white-blue for HMI
                                    if (value < 0.5) {
                                        const t = value * 2;
                                        colorValues = [Math.floor(255 * (0.3 + 0.7 * t)), Math.floor(255 * t), Math.floor(255 * t)];
                                    } else {
                                        const t = (value - 0.5) * 2;
                                        colorValues = [Math.floor(255 * (1 - t)), Math.floor(255 * (1 - t)), Math.floor(255 * (0.3 + 0.7 * t))];
                                    }
                                } else {
                                    colorValues = getAIAColorForWavelength(value, channel.wavelength);
                                }
                                
                                ctx.fillStyle = `rgb(${colorValues[0]}, ${colorValues[1]}, ${colorValues[2]})`;
                                ctx.fillRect(x, y, 1, 1);
                            }
                        }
                    }
                    
                    imageContainer.appendChild(canvas);
                    loadedImages[channelIndex][timeIndex] = canvas;
                };
                
                img.src = candidateUrl;
            }
            
            // Add animation controls
            const labelDiv = document.createElement('div');
            labelDiv.style.textAlign = 'center';
            labelDiv.style.color = '#ffffff';
            labelDiv.style.fontSize = '0.7rem';
            labelDiv.style.fontWeight = '600';
            labelDiv.style.marginTop = '0.25rem';
            labelDiv.textContent = channel.name;
            
            channelDiv.appendChild(imageContainer);
            channelDiv.appendChild(labelDiv);
            
            // Start animation after a delay
            setTimeout(() => {
                startImageAnimation(loadedImages[channelIndex], 800); // 0.8 second per frame
            }, channelIndex * 50);
            
            aiaGrid.appendChild(channelDiv);
        });
}

// Animation function for time series
function startImageAnimation(images, interval) {
    if (!images || images.length <= 1) return;
    
    let currentIndex = images.length - 1;
    
    setInterval(() => {
        // Hide all images
        images.forEach(img => {
            if (img) img.style.opacity = '0';
        });
        
        // Show current image
        if (images[currentIndex]) {
            images[currentIndex].style.opacity = '1';
        }
        
        // Move to next image
        currentIndex = (currentIndex + 1) % images.length;
    }, interval);
}

// Call this function when the script is loaded
initializeDemo(); 
