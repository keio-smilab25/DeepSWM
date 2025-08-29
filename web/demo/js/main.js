// Main application logic

class SolarFlareDemo {
    constructor() {
        this.translationManager = new window.TranslationManager();
        this.solarImagesManager = new window.SolarImagesManager();
        this.predictionManager = new window.PredictionManager();
        this.goesChartManager = new window.GOESChartManager();
        this.predictionManager.setTranslationManager(this.translationManager);
        
        this.currentDate = null;
        this.currentHour = 12;
        this.datePicker = null;
        
        this.init();
    }
    
    async init() {
        console.log('Initializing Solar Flare Demo...');
        
        // Load prediction data
        await this.predictionManager.loadPredictionData();
        
        // Initialize calendar and time selector
        this.initCalendar();
        this.initTimeSelector();
        
        // Initialize theme and language
        this.initTheme();
        this.languageToggleManager = new window.LanguageToggleManager(this.translationManager);
        
        // Initialize expandable sections
        this.initExpandableSections();
        
        // Initialize current forecast
        this.initCurrentForecast();
        
        // Load latest data automatically
        this.loadLatestData();
        
        console.log('Demo initialized successfully');
    }
    
    initCalendar() {
        const defaultDateTime = this.getDefaultDateTime();
        this.currentDate = defaultDateTime.date;
        this.currentHour = defaultDateTime.hour;
        
        this.createCustomCalendar();
    }
    
    createCustomCalendar() {
        const calendarContainer = document.getElementById('custom-calendar');
        if (!calendarContainer) return;
        
        this.currentMonth = this.currentDate.getMonth();
        this.currentYear = this.currentDate.getFullYear();
        
        this.updateTimestamp();
        this.renderCalendar();
    }
    
    renderCalendar() {
        const calendarContainer = document.getElementById('custom-calendar');
        if (!calendarContainer) return;
        
        const monthNames = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ];
        
        const weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        
        calendarContainer.innerHTML = `
            <div class="custom-calendar-header">
                <button class="custom-calendar-nav" id="prev-month">&lt;</button>
                <div class="custom-calendar-month-year">
                    <select class="month-year-select" id="month-select">
                        ${monthNames.map((month, index) => 
                            `<option value="${index}" ${index === this.currentMonth ? 'selected' : ''}>${month}</option>`
                        ).join('')}
                    </select>
                    <select class="month-year-select" id="year-select">
                        ${this.generateYearOptions()}
                    </select>
                </div>
                <button class="custom-calendar-nav" id="next-month">&gt;</button>
            </div>
            <div class="custom-calendar-body">
                <div class="custom-calendar-weekdays">
                    ${weekdays.map(day => `<div class="custom-calendar-weekday">${day}</div>`).join('')}
                </div>
                <div class="custom-calendar-days" id="calendar-days">
                    <!-- Days will be populated here -->
                </div>
            </div>
        `;
        
        this.renderDays();
        this.attachCalendarEvents();
    }
    
    generateYearOptions() {
        const currentYear = new Date().getFullYear();
        const startYear = 2020;
        const endYear = currentYear + 5;
        let options = '';
        
        for (let year = startYear; year <= endYear; year++) {
            options += `<option value="${year}" ${year === this.currentYear ? 'selected' : ''}>${year}</option>`;
        }
        
        return options;
    }
    
    renderDays() {
        const daysContainer = document.getElementById('calendar-days');
        if (!daysContainer) return;
        
        const firstDay = new Date(this.currentYear, this.currentMonth, 1);
        const lastDay = new Date(this.currentYear, this.currentMonth + 1, 0);
        const firstDayOfWeek = firstDay.getDay();
        const daysInMonth = lastDay.getDate();
        
        let daysHTML = '';
        
        // Previous month days
        const prevMonth = new Date(this.currentYear, this.currentMonth - 1, 0);
        for (let i = firstDayOfWeek - 1; i >= 0; i--) {
            const day = prevMonth.getDate() - i;
            daysHTML += `<div class="custom-calendar-day other-month" data-date="${this.currentYear}-${String(this.currentMonth).padStart(2, '0')}-${String(day).padStart(2, '0')}">${day}</div>`;
        }
        
        // Current month days
        for (let day = 1; day <= daysInMonth; day++) {
            const date = new Date(this.currentYear, this.currentMonth, day);
            const dateStr = `${this.currentYear}-${String(this.currentMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
            const isSelected = this.currentDate && 
                this.currentDate.getDate() === day && 
                this.currentDate.getMonth() === this.currentMonth && 
                this.currentDate.getFullYear() === this.currentYear;
            const hasData = this.predictionManager.hasDataForDate(date);
            
            let classes = 'custom-calendar-day';
            if (isSelected) classes += ' selected';
            if (hasData) classes += ' has-data';
            
            daysHTML += `<div class="${classes}" data-date="${dateStr}">${day}</div>`;
        }
        
        // Next month days
        const totalCells = Math.ceil((firstDayOfWeek + daysInMonth) / 7) * 7;
        const remainingCells = totalCells - (firstDayOfWeek + daysInMonth);
        for (let day = 1; day <= remainingCells; day++) {
            const nextMonth = this.currentMonth + 2;
            const nextYear = nextMonth > 12 ? this.currentYear + 1 : this.currentYear;
            const month = nextMonth > 12 ? 1 : nextMonth;
            daysHTML += `<div class="custom-calendar-day other-month" data-date="${nextYear}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}">${day}</div>`;
        }
        
        daysContainer.innerHTML = daysHTML;
    }
    
    attachCalendarEvents() {
        const prevButton = document.getElementById('prev-month');
        const nextButton = document.getElementById('next-month');
        const monthSelect = document.getElementById('month-select');
        const yearSelect = document.getElementById('year-select');
        const daysContainer = document.getElementById('calendar-days');
        
        if (prevButton) {
            prevButton.addEventListener('click', () => {
                this.currentMonth--;
                if (this.currentMonth < 0) {
                    this.currentMonth = 11;
                    this.currentYear--;
                }
                this.renderCalendar();
            });
        }
        
        if (nextButton) {
            nextButton.addEventListener('click', () => {
                this.currentMonth++;
                if (this.currentMonth > 11) {
                    this.currentMonth = 0;
                    this.currentYear++;
                }
                this.renderCalendar();
            });
        }
        
        if (monthSelect) {
            monthSelect.addEventListener('change', (e) => {
                this.currentMonth = parseInt(e.target.value);
                this.renderCalendar();
            });
        }
        
        if (yearSelect) {
            yearSelect.addEventListener('change', (e) => {
                this.currentYear = parseInt(e.target.value);
                this.renderCalendar();
            });
        }
        
        if (daysContainer) {
            daysContainer.addEventListener('click', (e) => {
                if (e.target.classList.contains('custom-calendar-day') && !e.target.classList.contains('other-month')) {
                    const dateStr = e.target.getAttribute('data-date');
                    if (dateStr) {
                        this.currentDate = new Date(dateStr + 'T00:00:00Z');
                        this.renderCalendar();
                        this.updateDisplay();
                    }
                }
            });
        }
    }
    
    getDefaultDateTime() {
        const now = new Date();
        const currentHour = now.getUTCHours();
        
        // Try current time - 3 hours first
        let targetHour = currentHour - 3;
        let targetDate = new Date(now);
        
        if (targetHour < 0) {
            targetDate.setUTCDate(targetDate.getUTCDate() - 1);
            targetHour = 24 + targetHour;
        }
        
        // Check if data exists for this date/hour
        const year = targetDate.getUTCFullYear();
        const month = String(targetDate.getUTCMonth() + 1).padStart(2, '0');
        const day = String(targetDate.getUTCDate()).padStart(2, '0');
        const hour = String(targetHour).padStart(2, '0');
        const dataKey = `${year}${month}${day}${hour}`;
        
        if (this.predictionManager.predictionData && this.predictionManager.predictionData[dataKey]) {
            return { date: targetDate, hour: targetHour };
        }
        
        // If no data for current-3h, find the latest available data
        const latestDate = this.predictionManager.getLatestAvailableDate();
        return { date: latestDate, hour: 12 }; // Default to noon
    }
    
    initTimeSelector() {
        const timeSelect = document.getElementById('time-select');
        if (!timeSelect) return;
        
        // Clear existing options
        timeSelect.innerHTML = '';
        
        // Add 24 hour options
        for (let h = 0; h < 24; h++) {
            const option = document.createElement('option');
            option.value = h;
            option.textContent = `${String(h).padStart(2, '0')}:00`;
            
            if (h === this.currentHour) {
                option.selected = true;
            }
            
            timeSelect.appendChild(option);
        }
        
        // Add change listener
        timeSelect.addEventListener('change', (e) => {
            this.currentHour = parseInt(e.target.value);
            this.updateTimestamp();
            this.updateDisplay();
        });
    }
    
    loadLatestData() {
        if (this.currentDate) {
            this.updateDisplay();
        }
    }
    
    updateDisplay() {
        this.updateDateDisplay();
        this.updateTimestamp();
        this.solarImagesManager.loadImages(this.currentDate, this.currentHour);
        this.predictionManager.displayPrediction(this.currentDate, this.currentHour);
        
        // Update performance displays based on current date
        this.predictionManager.updatePerformanceDisplays(this.currentDate);
        
        // Update Current Forecast with selected date/time
        this.updateCurrentForecast(this.currentDate, this.currentHour);
        
        // Update GOES chart with current date and hour
        const baseTime = new Date(this.currentDate);
        baseTime.setUTCHours(this.currentHour, 0, 0, 0);
        this.goesChartManager.updateChart(baseTime);
    }
    
    updateDateDisplay() {
        const lang = this.translationManager.getCurrentLang();
        const dateStr = this.currentDate.toLocaleDateString(lang === 'ja' ? 'ja-JP' : 'en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        
        const selectedDateEl = document.getElementById('selected-date');
        if (selectedDateEl) {
            selectedDateEl.textContent = dateStr;
        }
        
        const selectedTimeEl = document.getElementById('selected-time');
        if (selectedTimeEl) {
            selectedTimeEl.textContent = `${String(this.currentHour).padStart(2, '0')}:00 UTC`;
        }
    }
    
    updateTimestamp() {
        const timestampEl = document.getElementById('timestamp');
        if (timestampEl && this.currentDate) {
            const dateStr = `${this.currentDate.getFullYear()}-${String(this.currentDate.getMonth() + 1).padStart(2, '0')}-${String(this.currentDate.getDate()).padStart(2, '0')}`;
            const timeStr = `${String(this.currentHour).padStart(2, '0')}:00 UTC`;
            timestampEl.textContent = `${dateStr} ${timeStr}`;
        }
        
        // Update Multi-wavelength Solar Images title with time range
        const solarTitleEl = document.querySelector('.section-title[data-i18n="solar_images"]');
        if (solarTitleEl && this.solarImagesManager && this.solarImagesManager.loadedTimeRange) {
            const { startTime, endTime } = this.solarImagesManager.loadedTimeRange;
            solarTitleEl.textContent = `Multi-wavelength Solar Images ${startTime} - ${endTime} UTC`;
        } else if (solarTitleEl && this.currentDate) {
            const month = String(this.currentDate.getMonth() + 1).padStart(2, '0');
            const day = String(this.currentDate.getDate()).padStart(2, '0');
            const hour = String(this.currentHour).padStart(2, '0');
            solarTitleEl.textContent = `Multi-wavelength Solar Images ${month}/${day} ${hour}:00 UTC`;
        }
    }
    
    initTheme() {
        // Check URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        const themeParam = urlParams.get('theme');
        const langParam = urlParams.get('lang');
        
        // Initialize language from URL parameter
        if (langParam === 'ja') {
            this.translationManager.setLanguage('ja');
        }
        
        // Check if dark theme should be enabled
        const isDarkTheme = themeParam === 'space';
        
        if (isDarkTheme) {
            this.enableSpaceMode();
        }
        
        // Add theme toggle event listener
        const themeToggleBtn = document.getElementById('theme-toggle-btn');
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    }
    
    toggleTheme() {
        const body = document.body;
        const isCurrentlyDark = body.classList.contains('dark-theme');
        
        if (isCurrentlyDark) {
            this.enableLightMode();
        } else {
            this.enableSpaceMode();
        }
        
        // Update URL parameter
        this.updateURLTheme(!isCurrentlyDark);
    }
    
    enableSpaceMode() {
        document.body.classList.add('dark-theme');
        document.getElementById('night-sky').style.display = 'block';
        document.getElementById('starry-background').style.display = 'block';
        
        // Update GOES chart theme
        this.goesChartManager.updateTheme();
        
        const themeBtn = document.getElementById('theme-toggle-btn');
        if (themeBtn) {
            themeBtn.innerHTML = `
                <span class="theme-icon">☀️</span>
                <span class="theme-text">Light Mode</span>
            `;
        }
    }
    
    enableLightMode() {
        document.body.classList.remove('dark-theme');
        document.getElementById('night-sky').style.display = 'none';
        document.getElementById('starry-background').style.display = 'none';
        
        // Update GOES chart theme
        this.goesChartManager.updateTheme();
        
        const themeBtn = document.getElementById('theme-toggle-btn');
        if (themeBtn) {
            themeBtn.innerHTML = `
                <span class="theme-icon">🌙</span>
                <span class="theme-text">Space Mode</span>
            `;
        }
    }
    
    updateURLTheme(isDark) {
        const url = new URL(window.location);
        if (isDark) {
            url.searchParams.set('theme', 'space');
        } else {
            url.searchParams.delete('theme');
        }
        window.history.replaceState({}, '', url);
    }
    
    initExpandableSections() {
        const infoHeaders = document.querySelectorAll('.info-header');
        
        infoHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const section = header.parentElement;
                const isExpanded = section.classList.contains('expanded');
                
                // Close all other sections
                document.querySelectorAll('.info-section').forEach(otherSection => {
                    if (otherSection !== section) {
                        otherSection.classList.remove('expanded');
                    }
                });
                
                // Toggle current section
                if (isExpanded) {
                    section.classList.remove('expanded');
                } else {
                    section.classList.add('expanded');
                    
                    // Smooth scroll to the section after a short delay to allow animation
                    setTimeout(() => {
                        section.scrollIntoView({ 
                            behavior: 'smooth', 
                            block: 'start',
                            inline: 'nearest'
                        });
                    }, 100);
                }
            });
        });
    }
    
    async initCurrentForecast() {
        await this.loadCurrentForecastAndImages();
        
        // Initialize the new panel structure
        this.initializePanels();
    }
    
    initializePanels() {
        // Initialize current status panel (no percentages) with loading state
        this.updateLevelBlocksNew(1, null, 'current-level-blocks');
        
        // Initialize forecast panel (with percentages when data is available) with loading state
        this.updateLevelBlocksNew(1, null, 'forecast-level-blocks');
        
        // Set default loading states
        const currentStatus = document.getElementById('current-status');
        const forecastStatus = document.getElementById('forecast-status');
        
        if (currentStatus) {
            currentStatus.querySelector('.status-text').textContent = 'Loading...';
            currentStatus.querySelector('.level-text').textContent = '--';
        }
        
        if (forecastStatus) {
            forecastStatus.querySelector('.status-text').textContent = 'Loading...';
            forecastStatus.querySelector('.level-text').textContent = '--';
        }
    }
    
    updateDataTime(timestamp) {
        const timeElement = document.getElementById('current-time-value');
        if (timeElement && timestamp) {
            const year = timestamp.getFullYear();
            const month = String(timestamp.getMonth() + 1).padStart(2, '0');
            const day = String(timestamp.getDate()).padStart(2, '0');
            const hour = String(timestamp.getHours()).padStart(2, '0');
            const minute = String(timestamp.getMinutes()).padStart(2, '0');
            
            timeElement.textContent = `${year}/${month}/${day} ${hour}:${minute}`;
        }
        
        // Update the forecast title to show 24-hour range
        this.updateForecastTitle(timestamp);
    }
    
    updateForecastTitle(timestamp) {
        const titleElement = document.querySelector('.current-prediction-panel .subsection-title');
        if (titleElement && timestamp) {
            // Calculate start and end times (24 hours from current time)
            const startTime = new Date(timestamp);
            const endTime = new Date(timestamp.getTime() + 24 * 60 * 60 * 1000); // Add 24 hours
            
            // Format start time
            const startYear = startTime.getFullYear();
            const startMonth = String(startTime.getMonth() + 1).padStart(2, '0');
            const startDay = String(startTime.getDate()).padStart(2, '0');
            const startHour = String(startTime.getHours()).padStart(2, '0');
            const startMinute = String(startTime.getMinutes()).padStart(2, '0');
            
            // Format end time  
            const endYear = endTime.getFullYear();
            const endMonth = String(endTime.getMonth() + 1).padStart(2, '0');
            const endDay = String(endTime.getDate()).padStart(2, '0');
            const endHour = String(endTime.getHours()).padStart(2, '0');
            const endMinute = String(endTime.getMinutes()).padStart(2, '0');
            
            // Create the title with date range
            let startDateStr, endDateStr;
            if (startYear === endYear) {
                // Same year
                startDateStr = `${startYear}/${startMonth}/${startDay} ${startHour}:${startMinute}`;
                endDateStr = `${endMonth}/${endDay} ${endHour}:${endMinute}`;
            } else {
                // Different years (rare case)
                startDateStr = `${startYear}/${startMonth}/${startDay} ${startHour}:${startMinute}`;
                endDateStr = `${endYear}/${endMonth}/${endDay} ${endHour}:${endMinute}`;
            }
            
            titleElement.innerHTML = `Solar Flare Forecast Over the Next 24 Hours<br>(${startDateStr} - ${endDateStr} UTC)`;
        }
    }
    
    async loadCurrentForecastAndImages() {
        try {
            console.log('Loading current forecast and images...');
            
            // Use the already loaded prediction data from PredictionManager
            const data = this.predictionManager.predictionData;
            if (!data || Object.keys(data).length === 0) {
                throw new Error('No prediction data available');
            }
            
            console.log('Using prediction data:', Object.keys(data).length, 'entries');
            console.log('Sample keys:', Object.keys(data).slice(0, 5));
            
            // Find the latest available data that has corresponding images
            const sortedKeys = Object.keys(data).sort().reverse(); // Latest first
            console.log('Latest keys:', sortedKeys.slice(0, 5));
            let selectedKey = null;
            let selectedTimestamp = null;
            
            for (const key of sortedKeys) {
                // Parse timestamp from key (YYYYMMDDHH format)
                const year = parseInt(key.substr(0, 4));
                const month = parseInt(key.substr(4, 2));
                const day = parseInt(key.substr(6, 2));
                const hour = parseInt(key.substr(8, 2));
                
                const timestamp = new Date(year, month - 1, day, hour, 0, 0);
                
                // Check if images exist for this timestamp
                const monthStr = String(month).padStart(2, '0');
                const dayStr = String(day).padStart(2, '0');
                const hourStr = String(hour).padStart(2, '0');
                
                const imagePath = `${this.solarImagesManager.basePath}/data/images/${monthStr}${dayStr}/${hourStr}_aia_0304.png`;
                console.log('Checking image:', imagePath);
                
                const imageExists = await this.checkImageExists(imagePath);
                console.log('Image exists:', imageExists, 'for', imagePath);
                if (imageExists) {
                    selectedKey = key;
                    selectedTimestamp = timestamp;
                    console.log('Selected timestamp:', selectedKey, timestamp);
                    break;
                }
            }
            
            if (selectedKey && selectedTimestamp) {
                // Update the data time display
                this.updateDataTime(selectedTimestamp);
                
                // Display the prediction
                const predictionArray = data[selectedKey];
                const predictionObj = {
                    o_prob: predictionArray[0],
                    c_prob: predictionArray[1], 
                    m_prob: predictionArray[2],
                    x_prob: predictionArray[3]
                };
                this.displayCurrentForecast(predictionObj);
                
                // Load 4 images going backwards from this timestamp
                await this.loadAIA304ImagesFromTimestamp(selectedTimestamp);
                
            } else {
                throw new Error('No data with corresponding images found');
            }
            
        } catch (error) {
            console.error('Error loading current forecast:', error);
            this.displayCurrentForecastError();
        }
    }
    
    displayCurrentForecast(prediction) {
        // Determine flare level and status based on prediction
        const { level, status, statusClass, flareClass, icon, color } = this.getFlareLevel(prediction);
        
        // Update level blocks with prediction data
        this.updateLevelBlocks(level, prediction);
        
        // Update status text with icon
        const statusElement = document.getElementById('flare-status');
        if (statusElement) {
            statusElement.className = `flare-status ${statusClass}`;
            statusElement.querySelector('.status-text').innerHTML = `${icon} ${status}`;
            statusElement.querySelector('.level-text').textContent = `Lv.${level} (${flareClass})`;
            
            // Apply color to the icon
            statusElement.querySelector('.status-text').style.color = color;
        }
    }
    
    getFlareLevel(prediction) {
        // Extract probabilities
        const xProb = prediction.x_prob || 0;
        const mProb = prediction.m_prob || 0;
        const cProb = prediction.c_prob || 0;
        const oProb = prediction.o_prob || 0;
        
        // Determine the highest probability class
        const maxProb = Math.max(xProb, mProb, cProb, oProb);
        
        if (maxProb === xProb && xProb > 0.1) {
            return { level: 4, status: 'Major Flares', statusClass: 'status-major', flareClass: 'X-class', icon: '🔴', color: '#ff6b6b' };
        } else if (maxProb === mProb && mProb > 0.05) {
            return { level: 3, status: 'Active', statusClass: 'status-active', flareClass: 'M-class', icon: '🟡', color: '#ffa726' };
        } else if (maxProb === cProb && cProb > 0.1) {
            return { level: 2, status: 'Eruptive', statusClass: 'status-eruptive', flareClass: 'C-class', icon: '🟢', color: '#81c784' };
        } else {
            return { level: 1, status: 'Quiet', statusClass: 'status-quiet', flareClass: 'O-class', icon: '🟢', color: '#4caf50' };
        }
    }
    
    updateLevelBlocks(level, prediction = null) {
        const blocksContainer = document.getElementById('flare-level-blocks');
        if (!blocksContainer) return;
        
        // Clear existing blocks
        blocksContainer.innerHTML = '';
        blocksContainer.className = `flare-level-blocks level-${level}`;
        
        const levelInfo = [
            { label: 'Major Flares', level: 'Lv.4 (X-class)', className: 'x-class', key: 'x_prob', baseColor: [255, 107, 107] },
            { label: 'Active', level: 'Lv.3 (M-class)', className: 'm-class', key: 'm_prob', baseColor: [255, 167, 38] },
            { label: 'Eruptive', level: 'Lv.2 (C-class)', className: 'c-class', key: 'c_prob', baseColor: [129, 199, 132] },
            { label: 'Quiet', level: 'Lv.1 (O-class)', className: 'o-class', key: 'o_prob', baseColor: [76, 175, 80] }
        ];
        
        // Always show 4 blocks (from top to bottom: 4, 3, 2, 1)
        for (let i = 4; i >= 1; i--) {
            const block = document.createElement('div');
            block.className = 'level-block';
            
            // Fill blocks up to the current level
            if (i <= level) {
                block.classList.add('filled');
            }
            
            const info = levelInfo[4-i];
            const isCurrentLevel = i === level;
            const textColor = isCurrentLevel ? '#fff' : 'rgba(255, 255, 255, 0.3)';
            const textShadow = isCurrentLevel ? '1px 1px 2px rgba(0, 0, 0, 0.8)' : 'none';
            
            // Set border width - thicker for current level
            const borderWidth = isCurrentLevel ? '4px' : '2px';
            block.style.setProperty('border-width', borderWidth, 'important');
            
            // Apply appropriate background colors
            if (i <= level) {
                // For filled blocks, use appropriate colors
                const [r, g, b] = info.baseColor;
                block.style.background = `rgba(${r}, ${g}, ${b}, 0.8)`;
                block.style.borderColor = `rgb(${r}, ${g}, ${b})`;
            } else {
                // For unfilled blocks, use gray background
                block.style.background = 'rgba(248, 249, 250, 0.3)';
                block.style.borderColor = '#e9ecef';
            }
            
            // Get percentage for this level if prediction is available
            let percentageText = '';
            if (prediction && prediction[info.key] !== undefined) {
                const percentage = (prediction[info.key] * 100).toFixed(1);
                percentageText = `${percentage}%`;
            }
            
            block.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; height: 100%;">
                    <span style="font-size: 1rem; font-weight: 700; color: ${textColor}; text-shadow: ${textShadow}; line-height: 1;">${info.label} (${info.level})</span>
                    <span style="font-size: 1.2rem; font-weight: 800; color: ${textColor}; text-shadow: ${textShadow}; line-height: 1;">${percentageText}</span>
                </div>
            `;
            
            blocksContainer.appendChild(block);
        }
    }
    
    async loadAIA304ImagesFromTimestamp(baseTimestamp) {
        const container = document.getElementById('aia-304-container');
        if (!container) return;
        
        this.aia304Canvases = [];
        this.loadedTimes = [];
        
        console.log('Loading AIA 304 images from timestamp:', baseTimestamp);
        
        // Load 4 images going backwards from the base timestamp
        for (let i = 3; i >= 0; i--) { // Start from 3 hours back, go to current (oldest to newest)
            const timestamp = new Date(baseTimestamp.getTime() - i * 60 * 60 * 1000); // Go back i hours
            const month = String(timestamp.getMonth() + 1).padStart(2, '0');
            const day = String(timestamp.getDate()).padStart(2, '0');
            const hour = String(timestamp.getHours()).padStart(2, '0');
            
            const imagePath = `${this.solarImagesManager.basePath}/data/images/${month}${day}/${hour}_aia_0304.png`;
            
            try {
                const canvas = await this.loadAndProcessAIA304Image(imagePath);
                if (canvas) {
                    this.aia304Canvases.push(canvas); // Add to end for chronological order (oldest first)
                    this.loadedTimes.push(`${month}/${day} ${hour}:00`);
                    console.log('Loaded and processed image:', imagePath);
                }
            } catch (error) {
                console.log('Image not found:', imagePath);
            }
        }
        
        // Update solar activity period display
        this.updateSolarActivityPeriod();
        
        // Display canvases
        container.innerHTML = '';
        
        // Add copyright
        const copyright = document.createElement('div');
        copyright.className = 'aia-304-copyright';
        copyright.textContent = 'SDO©NASA';
        container.appendChild(copyright);
        
        this.aia304Canvases.forEach((canvas, index) => {
            canvas.className = 'aia-304-canvas';
            canvas.classList.toggle('active', index === 0);
            container.appendChild(canvas);
        });
        
        console.log('Total images loaded:', this.aia304Canvases.length);
        
        // Start automatic playback
        if (this.aia304Canvases.length > 1) {
            this.startAutoPlayback();
        } else if (this.aia304Canvases.length === 0) {
            container.innerHTML = '<div style="color: #6c757d; font-style: italic; text-align: center; padding: 2rem;">No AIA 304 Å images found</div>';
        }
    }
    
    updateSolarActivityPeriod() {
        const periodElement = document.getElementById('solar-activity-period');
        if (periodElement && this.loadedTimes.length > 0) {
            const startTime = this.loadedTimes[0]; // oldest
            const endTime = this.loadedTimes[this.loadedTimes.length - 1]; // newest
            periodElement.textContent = `(${startTime} - ${endTime} UTC)`;
        }
    }
    
    async loadAndProcessAIA304Image(imagePath) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            
            img.onload = () => {
                try {
                    // Use the colormap from solar-images.js
                    if (window.AIAColormaps) {
                        const coloredCanvas = window.AIAColormaps.apply(img, '0304');
                        resolve(coloredCanvas);
                    } else {
                        // Fallback: create canvas without colormap
                        const canvas = document.createElement('canvas');
                        const ctx = canvas.getContext('2d');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        resolve(canvas);
                    }
                } catch (error) {
                    console.error('Error processing image:', error);
                    resolve(null);
                }
            };
            
            img.onerror = () => {
                resolve(null);
            };
            
            img.src = imagePath;
        });
    }
    
    async checkImageExists(url) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(true);
            img.onerror = () => resolve(false);
            img.src = url;
        });
    }
    
    startAutoPlayback() {
        if (this.aia304Canvases.length <= 1) return;
        
        this.currentFrame = 0;
        this.autoPlayInterval = null;
        
        console.log('Starting auto playback with', this.aia304Canvases.length, 'canvases');
        
        // Start continuous loop
        this.autoPlayInterval = setInterval(() => {
            this.currentFrame = (this.currentFrame + 1) % this.aia304Canvases.length;
            this.updateAutoDisplay();
        }, 1000); // 1 FPS for smooth viewing
    }
    
    updateAutoDisplay() {
        // Update active canvas
        this.aia304Canvases.forEach((canvas, index) => {
            canvas.classList.toggle('active', index === this.currentFrame);
        });
        
        console.log('Displaying frame:', this.currentFrame + 1, '/', this.aia304Canvases.length);
    }
    
    stopAutoPlayback() {
        if (this.autoPlayInterval) {
            clearInterval(this.autoPlayInterval);
            this.autoPlayInterval = null;
        }
    }
    
    displayCurrentForecastError() {
        const statusElement = document.getElementById('flare-status');
        if (statusElement) {
            statusElement.className = 'flare-status status-quiet';
            statusElement.querySelector('.status-text').textContent = 'Loading...';
            statusElement.querySelector('.level-text').textContent = '--';
        }
        
        this.updateLevelBlocks(1);
    }
    
    updateCurrentForecast(date, hour) {
        // Update the data time display
        const timestamp = new Date(date);
        timestamp.setUTCHours(hour, 0, 0, 0);
        this.updateDataTime(timestamp);
        
        // Update forecast period display
        this.updateForecastPeriod(timestamp);
        
        // Update current status time display
        this.updateCurrentStatusTime(timestamp);
        
        // Get prediction data for the selected date/hour
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hourStr = String(hour).padStart(2, '0');
        const dataKey = `${year}${month}${day}${hourStr}`;
        
        // Update current status from XRS data
        this.updateCurrentStatus(dataKey);
        
        // Update current status time period (24 hours range)
        this.updateCurrentStatusTimePeriod(timestamp);
        
        // Update 24-hour forecast
        if (this.predictionManager.predictionData && this.predictionManager.predictionData[dataKey]) {
            const predictionArray = this.predictionManager.predictionData[dataKey];
            const predictionObj = {
                o_prob: predictionArray[0],
                c_prob: predictionArray[1], 
                m_prob: predictionArray[2],
                x_prob: predictionArray[3]
            };
            this.updateForecastPanel(predictionObj);
            
            // Load AIA 304 images for this timestamp
            this.loadAIA304ImagesFromTimestamp(timestamp);
        } else {
            this.displayForecastError();
        }
    }
    
    updateCurrentStatusTime(timestamp) {
        const statusTimeEl = document.getElementById('current-status-time');
        if (statusTimeEl) {
            const timeStr = this.formatTimeForDisplay(timestamp);
            statusTimeEl.textContent = `(${timeStr} UTC)`;
        }
    }
    
    updateForecastPeriod(timestamp) {
        // Update 24-hour forecast period
        const startTime = new Date(timestamp);
        const endTime = new Date(timestamp.getTime() + 24 * 60 * 60 * 1000);
        
        const startStr = this.formatTimeForDisplay(startTime);
        const endStr = this.formatTimeForDisplay(endTime);
        
        const forecastPeriodEl = document.getElementById('forecast-period');
        if (forecastPeriodEl) {
            forecastPeriodEl.textContent = `(${startStr} - ${endStr} UTC)`;
        }
    }
    
    formatTimeForDisplay(timestamp) {
        const month = String(timestamp.getMonth() + 1);
        const day = String(timestamp.getDate());
        const hour = String(timestamp.getHours()).padStart(2, '0');
        return `${month}/${day} ${hour}:00`;
    }
    
    updateCurrentStatus(dataKey) {
        // Get XRS data for current status - use 24-hour maximum
        const xrsData = this.predictionManager.xrsData;
        if (xrsData) {
            const maxFlux = this.get24HourMaxXRS(dataKey, xrsData);
            const flareClass = this.getFlareClassFromFlux(maxFlux);
            const statusInfo = this.getStatusFromFlareClass(flareClass);
            
            // Update current status display
            this.updateStatusPanel(statusInfo, 'current');
        } else {
            // Default to quiet status if no data
            const defaultStatus = {
                level: 1,
                status: 'Quiet',
                statusClass: 'status-quiet',
                flareClass: 'O-class',
                icon: '🟢',
                color: '#4caf50'
            };
            this.updateStatusPanel(defaultStatus, 'current');
        }
    }
    
    get24HourMaxXRS(currentDataKey, xrsData) {
        // Parse current timestamp
        const year = parseInt(currentDataKey.substr(0, 4));
        const month = parseInt(currentDataKey.substr(4, 2));
        const day = parseInt(currentDataKey.substr(6, 2));
        const hour = parseInt(currentDataKey.substr(8, 2));
        
        const currentTime = new Date(year, month - 1, day, hour, 0, 0);
        const startTime = new Date(currentTime.getTime() - 24 * 60 * 60 * 1000); // 24 hours ago
        
        let maxFlux = 0;
        
        // Check all data points in the past 24 hours
        for (let i = 0; i <= 24; i++) {
            const checkTime = new Date(startTime.getTime() + i * 60 * 60 * 1000); // Each hour
            const checkKey = `${checkTime.getFullYear()}${String(checkTime.getMonth() + 1).padStart(2, '0')}${String(checkTime.getDate()).padStart(2, '0')}${String(checkTime.getHours()).padStart(2, '0')}`;
            
            if (xrsData[checkKey] !== undefined) {
                maxFlux = Math.max(maxFlux, xrsData[checkKey]);
            }
        }
        
        return maxFlux;
    }
    
    updateCurrentStatusTimePeriod(timestamp) {
        const statusTimeEl = document.getElementById('current-status-time');
        if (statusTimeEl) {
            const endTime = new Date(timestamp);
            const startTime = new Date(timestamp.getTime() - 24 * 60 * 60 * 1000);
            
            const startStr = this.formatTimeForDisplay(startTime);
            const endStr = this.formatTimeForDisplay(endTime);
            
            statusTimeEl.textContent = `(${startStr} - ${endStr} UTC)`;
        }
    }
    
    getFlareClassFromFlux(flux) {
        if (flux >= 1e-4) return 'X';
        if (flux >= 1e-5) return 'M';
        if (flux >= 1e-6) return 'C';
        return 'O';
    }
    
    getStatusFromFlareClass(flareClass) {
        switch (flareClass) {
            case 'X':
                return { level: 4, status: 'Major Flares', statusClass: 'status-major', flareClass: 'X-class', icon: '🔴', color: '#ff6b6b' };
            case 'M':
                return { level: 3, status: 'Active', statusClass: 'status-active', flareClass: 'M-class', icon: '🟡', color: '#ffa726' };
            case 'C':
                return { level: 2, status: 'Eruptive', statusClass: 'status-eruptive', flareClass: 'C-class', icon: '🟢', color: '#81c784' };
            default:
                return { level: 1, status: 'Quiet', statusClass: 'status-quiet', flareClass: 'O-class', icon: '🟢', color: '#4caf50' };
        }
    }
    
    updateStatusPanel(statusInfo, panelType) {
        const prefix = panelType === 'current' ? 'current' : 'forecast';
        const blocksId = `${prefix}-level-blocks`;
        const statusId = `${prefix}-status`;
        
        // Update level blocks
        this.updateLevelBlocksNew(statusInfo.level, null, blocksId);
        
        // Update status display
        const statusElement = document.getElementById(statusId);
        if (statusElement) {
            statusElement.className = `panel-status ${statusInfo.statusClass}`;
            statusElement.querySelector('.status-text').innerHTML = `${statusInfo.icon} ${statusInfo.status}`;
            statusElement.querySelector('.level-text').textContent = `Lv.${statusInfo.level} (${statusInfo.flareClass})`;
            
            // Apply color to the icon
            statusElement.querySelector('.status-text').style.color = statusInfo.color;
        }
    }
    
    updateForecastPanel(prediction) {
        // Determine flare level and status based on prediction
        const { level, status, statusClass, flareClass, icon, color } = this.getFlareLevel(prediction);
        
        // Update level blocks with prediction data
        this.updateLevelBlocksNew(level, prediction, 'forecast-level-blocks');
        
        // Update status display
        const statusElement = document.getElementById('forecast-status');
        if (statusElement) {
            statusElement.className = `panel-status ${statusClass}`;
            statusElement.querySelector('.status-text').innerHTML = `${icon} ${status}`;
            statusElement.querySelector('.level-text').textContent = `Lv.${level} (${flareClass})`;
            
            // Apply color to the icon
            statusElement.querySelector('.status-text').style.color = color;
        }
    }
    
    updateLevelBlocksNew(level, prediction = null, containerId) {
        const blocksContainer = document.getElementById(containerId);
        if (!blocksContainer) return;
        
        // Clear existing blocks
        blocksContainer.innerHTML = '';
        
        const levelInfo = [
            { label: 'Major Flares', level: 'Lv.4', className: 'x-class', key: 'x_prob', baseColor: [255, 107, 107] },
            { label: 'Active', level: 'Lv.3', className: 'm-class', key: 'm_prob', baseColor: [255, 167, 38] },
            { label: 'Eruptive', level: 'Lv.2', className: 'c-class', key: 'c_prob', baseColor: [129, 199, 132] },
            { label: 'Quiet', level: 'Lv.1', className: 'o-class', key: 'o_prob', baseColor: [76, 175, 80] }
        ];
        
        // Create 4 blocks (from top to bottom: 4, 3, 2, 1)
        for (let i = 4; i >= 1; i--) {
            const block = document.createElement('div');
            block.className = 'level-block';
            
            // Fill blocks up to the current level
            if (i <= level) {
                block.classList.add('filled');
            }
            
            const info = levelInfo[4-i];
            
            // For filled blocks, use the color of the current level
            // This makes all filled blocks the same color (e.g., all yellow for Active level)
            let colorInfo;
            if (i <= level) {
                colorInfo = levelInfo[4-level]; // Use current level's color for all filled blocks
            } else {
                colorInfo = info; // Use individual color for unfilled blocks
            }
            const [r, g, b] = colorInfo.baseColor;
            
            // Apply appropriate background colors
            if (i <= level) {
                block.style.background = `rgba(${r}, ${g}, ${b}, 0.8)`;
                // If this is the current level, use white border, otherwise use the color border
                if (i === level) {
                    block.style.borderColor = '#fff';
                    block.style.borderWidth = '3px';
                    block.style.color = '#fff';
                } else {
                    block.style.borderColor = `rgb(${r}, ${g}, ${b})`;
                    block.style.borderWidth = '2px';
                    block.style.color = 'rgba(255, 255, 255, 0.6)';
                }
            } else {
                block.style.background = 'rgba(248, 249, 250, 0.1)';
                block.style.borderColor = '#666';
                block.style.borderWidth = '2px';
                block.style.color = '#999';
            }
            
            // Get percentage for this level if prediction is available
            let percentageText = '';
            if (prediction && prediction[info.key] !== undefined) {
                const percentage = (prediction[info.key] * 100).toFixed(1);
                percentageText = `${percentage}%`;
            }
            
            // Determine text color based on level
            const textColor = (i === level) ? '#fff' : (i <= level) ? 'rgba(255, 255, 255, 0.6)' : '#999';
            
            block.innerHTML = `
                <span style="font-size: 1.2rem; font-weight: 600; color: ${textColor};">${info.label} (${info.level})</span>
                <span style="font-size: 1.3rem; font-weight: 700; color: ${textColor};">${percentageText}</span>
            `;
            
            blocksContainer.appendChild(block);
        }
    }
    
    displayForecastError() {
        const statusElement = document.getElementById('forecast-status');
        if (statusElement) {
            statusElement.className = 'panel-status status-quiet';
            statusElement.querySelector('.status-text').textContent = 'Loading...';
            statusElement.querySelector('.level-text').textContent = '--';
        }
        
        this.updateLevelBlocksNew(1, null, 'forecast-level-blocks');
    }
    

}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SolarFlareDemo();
});
