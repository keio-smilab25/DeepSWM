// Main application logic

class SolarFlareDemo {
    constructor() {
        this.translationManager = new window.TranslationManager();
        this.timezoneManager = new window.TimezoneManager();
        this.contentLoader = new window.ContentLoader(this.translationManager);
        this.solarImagesManager = new window.SolarImagesManager();
        this.predictionManager = new window.PredictionManager();
        this.goesChartManager = new window.GOESChartManager();
        this.predictionManager.setTranslationManager(this.translationManager);
        this.goesChartManager.setTimezoneManager(this.timezoneManager);
        this.goesChartManager.init();
        
        this.currentDate = null;
        this.currentHour = 12;
        this.datePicker = null;
        
        // Add debouncing and queue management for iOS Safari fix
        this.updateDebounceTimer = null;
        this.isUpdating = false;
        this.pendingUpdate = null;
        
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
        
        // Listen for language change events
        window.addEventListener('languageChanged', () => {
            this.renderCalendar();
            // Refresh dynamic content after language change
            setTimeout(() => {
                this.refreshDynamicContent();
                this.updateDataTimeLabel();
                this.updateTimeSelectorLabel();
            }, 100); // Small delay to ensure translation manager is updated
        });
        
        // Initialize expandable sections
        this.initExpandableSections();
        
        // Initialize image modals
        this.initImageModals();
        
        // Initialize current forecast
        this.initCurrentForecast();
        
        // Load latest data automatically
        this.loadLatestData();
        
        // Update timezone-dependent labels
        setTimeout(() => {
            this.updateDataTimeLabel();
            this.updateTimeSelectorLabel();
        }, 100);
        
        console.log('Demo initialized successfully');
    }
    
    initCalendar() {
        const defaultDateTime = this.getDefaultDateTime();
        this.currentDate = defaultDateTime.date;
        this.currentHour = defaultDateTime.hour;
        
        // Ensure current year is within allowed range (2025 May 1 or later)
        const currentYear = new Date().getFullYear();
        const cutoffDate = new Date(2025, 4, 1); // May 1, 2025
        
        if (this.currentDate < cutoffDate) {
            this.currentDate = new Date(cutoffDate); // May 1, 2025
        } else if (this.currentDate.getFullYear() > currentYear) {
            this.currentDate = new Date(currentYear, 11, 31); // December 31, current year
        }
        
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
        
        const monthNames = this.translationManager.t('months') || [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ];
        
        const weekdays = this.translationManager.t('weekdays') || ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        
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
        const startYear = 2025; // 2025年から選択可能
        const endYear = currentYear; // 現在年まで選択可能
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
        const today = new Date();
        today.setHours(23, 59, 59, 999); // Set to end of today for comparison
        
        // Define the cutoff date: May 1, 2025
        const cutoffDate = new Date(2025, 4, 1); // Month is 0-indexed, so 4 = May
        
        for (let day = 1; day <= daysInMonth; day++) {
            const date = new Date(this.currentYear, this.currentMonth, day);
            const dateStr = `${this.currentYear}-${String(this.currentMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
            const isSelected = this.currentDate && 
                this.currentDate.getDate() === day && 
                this.currentDate.getMonth() === this.currentMonth && 
                this.currentDate.getFullYear() === this.currentYear;
            // Convert local date to UTC for data check
            const utcDateForCheck = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
            const hasData = this.predictionManager.hasDataForDate(utcDateForCheck);
            
            // Check if date is before cutoff (2025 May 1) or in the future
            const isBeforeCutoff = date < cutoffDate;
            const isFutureDate = date > today;
            const isDisabled = isBeforeCutoff || isFutureDate;
            
            let classes = 'custom-calendar-day';
            if (isSelected) classes += ' selected';
            if (hasData && !isDisabled) classes += ' has-data';
            if (isDisabled) classes += ' disabled';
            
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
                    // 2025年5月1日以前には移動しない
                    const cutoffDate = new Date(2025, 4, 1); // May 1, 2025
                    const checkDate = new Date(this.currentYear, this.currentMonth, 1);
                    if (checkDate < cutoffDate) {
                        this.currentYear = 2025;
                        this.currentMonth = 4; // May (0-indexed)
                    }
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
                    // 現在年以降には移動しない
                    const currentYear = new Date().getFullYear();
                    if (this.currentYear > currentYear) {
                        this.currentYear = currentYear;
                        this.currentMonth = 11;
                    }
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
                const selectedYear = parseInt(e.target.value);
                const currentYear = new Date().getFullYear();
                
                // Ensure selected year is within allowed range (2025 May 1 or later)
                const cutoffDate = new Date(2025, 4, 1); // May 1, 2025
                const checkDate = new Date(selectedYear, 0, 1); // January 1 of selected year
                
                if (checkDate >= cutoffDate && selectedYear <= currentYear) {
                    this.currentYear = selectedYear;
                    this.renderCalendar();
                } else {
                    // Reset to current selection if invalid
                    e.target.value = this.currentYear;
                }
            });
        }
        
        if (daysContainer) {
            daysContainer.addEventListener('click', (e) => {
                if (e.target.classList.contains('custom-calendar-day') && 
                    !e.target.classList.contains('other-month') && 
                    !e.target.classList.contains('disabled')) {
                    const dateStr = e.target.getAttribute('data-date');
                    if (dateStr) {
                        // Parse date in user's local timezone to avoid Invalid Date
                        const [year, month, day] = dateStr.split('-').map(num => parseInt(num));
                        this.currentDate = new Date(year, month - 1, day);
                        this.renderCalendar();
                        this.debouncedUpdateDisplay();
                    }
                }
            });
        }
    }
    
    getDefaultDateTime() {
        // Local-first: start from local now - 2h, convert to UTC for data check, step back if missing
        const pred = this.predictionManager?.predictionData || {};
        
        // Build prediction key from UTC date parts
        const buildKey = (utcDateObj, utcHour) => {
            const y = utcDateObj.getUTCFullYear();
            const m = String(utcDateObj.getUTCMonth() + 1).padStart(2, '0');
            const d = String(utcDateObj.getUTCDate()).padStart(2, '0');
            const h = String(utcHour).padStart(2, '0');
            return `${y}${m}${d}${h}`;
        };
        
        const nowLocal = new Date();
        const earliestUtcMs = Date.UTC(2025, 4, 1, 0, 0, 0); // 2025-05-01 00:00:00 UTC
        
        // Search window: look back up to 7 days from local now-2h
        const maxLookbackHours = 24 * 7;
        for (let i = 2; i <= maxLookbackHours + 2; i++) {
            // Candidate local time = nowLocal - i hours
            const candidateLocalTs = new Date(
                nowLocal.getFullYear(),
                nowLocal.getMonth(),
                nowLocal.getDate(),
                nowLocal.getHours(),
                0, 0, 0
            );
            candidateLocalTs.setHours(candidateLocalTs.getHours() - i);
            
            // Convert local candidate to UTC parts used by data keys
            const localDateOnly = new Date(
                candidateLocalTs.getFullYear(),
                candidateLocalTs.getMonth(),
                candidateLocalTs.getDate()
            );
            const localHourOnly = candidateLocalTs.getHours();
            const { utcDate, utcHour, utcTimestamp } = this.getUtcFromLocal(localDateOnly, localHourOnly);
            
            // Stop if before operational cutoff (in UTC)
            if (utcTimestamp.getTime() < earliestUtcMs) break;
            
            const key = buildKey(utcDate, utcHour);
            if (Object.prototype.hasOwnProperty.call(pred, key)) {
                // Return as local date/hour for UI
                return { date: localDateOnly, hour: localHourOnly };
            }
        }
        
        // Fallback: use latest prediction entry; convert UTC -> local for default selection
        const keys = this.predictionManager && this.predictionManager.predictionData
            ? Object.keys(this.predictionManager.predictionData)
            : [];
        if (keys.length > 0) {
            keys.sort((a, b) => b.localeCompare(a));
            const latestKey = keys[0]; // YYYYMMDDHH in UTC
            const y = parseInt(latestKey.slice(0, 4), 10);
            const m = parseInt(latestKey.slice(4, 6), 10) - 1;
            const d = parseInt(latestKey.slice(6, 8), 10);
            const h = parseInt(latestKey.slice(8, 10), 10);
            const utcTs = new Date(Date.UTC(y, m, d, h, 0, 0));
            const localDate = new Date(utcTs.getFullYear(), utcTs.getMonth(), utcTs.getDate());
            const localHour = utcTs.getHours();
            return { date: localDate, hour: localHour };
        }
        
        // Absolute fallback: clamp within allowed range and use local now - 2h
        const cutoffLocal = new Date(2025, 4, 1, 0, 0, 0);
        const fallbackTs = new Date(
            nowLocal.getFullYear(),
            nowLocal.getMonth(),
            nowLocal.getDate(),
            nowLocal.getHours(),
            0, 0, 0
        );
        fallbackTs.setHours(fallbackTs.getHours() - 2);
        const fallbackDate = fallbackTs < cutoffLocal
            ? new Date(cutoffLocal.getFullYear(), cutoffLocal.getMonth(), cutoffLocal.getDate())
            : new Date(fallbackTs.getFullYear(), fallbackTs.getMonth(), fallbackTs.getDate());
        const fallbackHour = fallbackTs < cutoffLocal ? 0 : fallbackTs.getHours();
        return { date: fallbackDate, hour: fallbackHour };
    }

	// Convert a local date and hour selection to UTC parts used by data APIs
	getUtcFromLocal(localDate, localHour) {
		const localDateTime = new Date(
			localDate.getFullYear(),
			localDate.getMonth(),
			localDate.getDate(),
			localHour, 0, 0, 0
		);
		const utcYear = localDateTime.getUTCFullYear();
		const utcMonth = localDateTime.getUTCMonth();
		const utcDay = localDateTime.getUTCDate();
		const utcHour = localDateTime.getUTCHours();
		const utcDate = new Date(Date.UTC(utcYear, utcMonth, utcDay));
		const utcTimestamp = new Date(Date.UTC(utcYear, utcMonth, utcDay, utcHour, 0, 0, 0));
		return { utcDate, utcHour, utcTimestamp };
	}
	
	getSelectedUtcParts() {
		if (!this.currentDate || typeof this.currentHour !== 'number') {
			return { utcDate: new Date(), utcHour: 0, utcTimestamp: new Date() };
		}
		return this.getUtcFromLocal(this.currentDate, this.currentHour);
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
        
        // Update the time selector label with timezone
        this.updateTimeSelectorLabel();
        
        // Add change listener
        timeSelect.addEventListener('change', (e) => {
            this.currentHour = parseInt(e.target.value);
            this.updateTimestamp();
            this.debouncedUpdateDisplay();
        });
    }

    updateTimeSelectorLabel() {
        const timeLabel = document.querySelector('label[data-i18n="time_label"]');
        if (timeLabel && this.timezoneManager) {
            const baseText = this.translationManager.t('time_label');
            const timezone = this.timezoneManager.getTimezoneAbbreviation();
            timeLabel.textContent = `${baseText} (${timezone})`;
        }
    }
    
    loadLatestData() {
        if (this.currentDate) {
            this.updateDisplay();
        }
    }
    
    // Debounced update display method for iOS Safari fix
    debouncedUpdateDisplay() {
        // Clear existing timer
        if (this.updateDebounceTimer) {
            clearTimeout(this.updateDebounceTimer);
        }
        
        // Set pending update data
        this.pendingUpdate = {
            date: new Date(this.currentDate),
            hour: this.currentHour,
            timestamp: Date.now()
        };
        
        // Set debounced timer (50ms delay for responsiveness while preventing iOS Safari issues)
        this.updateDebounceTimer = setTimeout(() => {
            this.executeQueuedUpdate();
        }, 50);
    }
    
    // Execute queued update with race condition protection
    async executeQueuedUpdate() {
        // If already updating, skip this call
        if (this.isUpdating) {
            // Update already in progress, skipping
            return;
        }
        
        // If no pending update, skip
        if (!this.pendingUpdate) {
            return;
        }
        
        const updateData = this.pendingUpdate;
        this.pendingUpdate = null;
        this.isUpdating = true;
        
        try {
            // Executing queued update
            
            // Set current date/hour to the latest pending values
            this.currentDate = updateData.date;
            this.currentHour = updateData.hour;
            
            // Execute the actual update
            await this.updateDisplay();
            
        } catch (error) {
            console.error('Error during queued update:', error);
        } finally {
            this.isUpdating = false;
            
            // Check if there's a newer pending update that came in during processing
            if (this.pendingUpdate && this.pendingUpdate.timestamp > updateData.timestamp) {
                // Newer update pending, executing
                // Small delay to prevent infinite loops
                setTimeout(() => this.executeQueuedUpdate(), 50);
            }
        }
    }
    
    async updateDisplay() {
        // Convert selected local date/hour to UTC parts for data operations
        const { utcDate, utcHour } = this.getSelectedUtcParts();
        
        // Update basic display elements first (synchronous)
        this.updateDateDisplay(utcDate);
        this.updateTimestamp(utcDate);
        
        // Start async operations in parallel but wait for critical ones
        const imageLoadPromise = this.solarImagesManager.loadImages(utcDate, utcHour);
        
        // Update prediction display (synchronous)
        this.predictionManager.displayPrediction(utcDate, utcHour);
        
        // Update performance displays (synchronous)
        this.predictionManager.updatePerformanceDisplays(utcDate);
        
        // Update Current Forecast with selected date/time (includes AIA 304 images)
        this.updateCurrentForecast(utcDate, utcHour);
        
        // Update GOES chart (synchronous)
        const baseTime = new Date(utcDate.getTime());
        baseTime.setUTCHours(utcHour, 0, 0, 0);
        this.goesChartManager.updateChart(baseTime);
        
        try {
            await imageLoadPromise;
            // Image loading completed -> refresh timestamp/header to avoid stale range
            this.updateTimestamp(utcDate);
        } catch (error) {
            console.error('Error loading images:', error);
        }
    }
    
    updateDateDisplay(utcDate = null) {
        const lang = this.translationManager.getCurrentLang();
        
        // Use provided UTC date or convert local date to UTC for data processing
        if (!utcDate) {
            utcDate = new Date(Date.UTC(this.currentDate.getFullYear(), this.currentDate.getMonth(), this.currentDate.getDate()));
        }
        const displayDate = new Date(utcDate);
        displayDate.setUTCHours(this.currentHour, 0, 0, 0);
        
        // Format date in user's timezone
        const dateStr = this.timezoneManager.formatDateInTimezone(displayDate, {
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
            const timeStr = this.timezoneManager.formatTimeWithTimezone(displayDate);
            selectedTimeEl.textContent = timeStr;
        }
    }
    
    updateTimestamp(utcDate = null) {
        const timestampEl = document.getElementById('timestamp');
        if (timestampEl && this.currentDate) {
            // Display in user's local timezone for calendar UI
            const { utcDate: _utcDate, utcHour } = this.getSelectedUtcParts();
            const displayDate = new Date(_utcDate);
            displayDate.setUTCHours(utcHour, 0, 0, 0);
            
            const formattedTime = this.timezoneManager.formatDateInTimezone(displayDate, {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
            
            const timezone = this.timezoneManager.getTimezoneAbbreviation();
            timestampEl.textContent = `${formattedTime} ${timezone}`;
        }
        
        // Update Multi-wavelength Solar Images title with time range
        const solarTitleEl = document.querySelector('.section-title[data-i18n="solar_images"]');
        if (solarTitleEl && this.solarImagesManager && this.solarImagesManager.loadedTimeRange) {
            const { startTime, endTime } = this.solarImagesManager.loadedTimeRange;
            const solarImagesText = this.translationManager.t('solar_images');
            
            // Convert time strings to dates and format in user's timezone
            try {
                // Use UTC base date (data stored in UTC)
                const dataUtcDate = utcDate || new Date(Date.UTC(this.currentDate.getFullYear(), this.currentDate.getMonth(), this.currentDate.getDate()));
                // startTime/endTime = "MM/DD HH:MM" → parse explicitly
                const [startMD, startHM] = String(startTime).split(' ');
                const [endMD, endHM] = String(endTime).split(' ');
                const [startMonthStr, startDayStr] = startMD.split('/');
                const [startHourStr] = startHM.split(':');
                const [endMonthStr, endDayStr] = endMD.split('/');
                const [endHourStr] = endHM.split(':');
                const year = dataUtcDate.getUTCFullYear();
                const startUTC = new Date(Date.UTC(
                    year,
                    parseInt(startMonthStr, 10) - 1,
                    parseInt(startDayStr, 10),
                    parseInt(startHourStr, 10),
                    0,
                    0
                ));
                const endUTC = new Date(Date.UTC(
                    year,
                    parseInt(endMonthStr, 10) - 1,
                    parseInt(endDayStr, 10),
                    parseInt(endHourStr, 10),
                    0,
                    0
                ));
                
                const timeRange = this.timezoneManager.formatDateRange(startUTC, endUTC);
                solarTitleEl.textContent = `${solarImagesText} ${timeRange}`;
            } catch (error) {
                // Fallback to original format
                const timezone = this.timezoneManager.getTimezoneAbbreviation();
                solarTitleEl.textContent = `${solarImagesText} ${startTime} - ${endTime} ${timezone}`;
            }
        } else if (solarTitleEl && this.currentDate) {
            // Create UTC date from local date for display
            const displayDate = new Date(utcDate || new Date(Date.UTC(this.currentDate.getFullYear(), this.currentDate.getMonth(), this.currentDate.getDate())));
            displayDate.setUTCHours(this.currentHour, 0, 0, 0);
            
            const timeStr = this.timezoneManager.formatDateInTimezone(displayDate, {
                month: 'numeric',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
            
            const timezone = this.timezoneManager.getTimezoneAbbreviation();
            const solarImagesText = this.translationManager.t('solar_images');
            solarTitleEl.textContent = `${solarImagesText} ${timeStr} ${timezone}`;
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
        } else {
            this.enableLightMode();
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
        
        // Refresh dynamic content to update text colors
        this.refreshDynamicContent();
        
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
        
        // Refresh dynamic content to update text colors
        this.refreshDynamicContent();
        
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
                }
            });
        });
    }
    
    async initCurrentForecast() {
        // Initialize the new panel structure first
        this.initializePanels();
        
        // Set initial panel titles to "current" version
        this.updateInitialPanelTitles();
        
        // Then load the actual data
        await this.loadCurrentForecastAndImages();
    }
    
    initializePanels() {
        // Set default quiet status for both panels
        const defaultStatus = {
            level: 1,
            status: this.translationManager.t('flare_status_quiet'),
            statusClass: 'status-quiet',
            flareClass: 'O-class',
            icon: '🟢',
            color: '#4caf50'
        };
        
        // Initialize current status panel with default quiet status
        this.updateStatusPanel(defaultStatus, 'current');
        
        // Initialize forecast panel with default quiet prediction
        const defaultPrediction = {
            o_prob: 1.0,
            c_prob: 0.0,
            m_prob: 0.0,
            x_prob: 0.0
        };
        this.updateForecastPanel(defaultPrediction);
    }
    
    updateDataTime(timestamp) {
        const timeElement = document.getElementById('current-time-value');
        if (timeElement && timestamp) {
            // Format timestamp in user's timezone (without timezone suffix since it's in the label)
            const formattedTime = this.timezoneManager.formatDateInTimezone(timestamp, {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
            
            timeElement.textContent = formattedTime;
        }
        
        // Update the data time label with current timezone
        this.updateDataTimeLabel();
        
        // Update the forecast title to show 24-hour range
        this.updateForecastTitle(timestamp);
    }

    updateDataTimeLabel() {
        const timeLabel = document.querySelector('.time-label[data-i18n="data_time_utc"]');
        if (timeLabel && this.timezoneManager) {
            const baseText = this.translationManager.t('data_time_utc');
            timeLabel.textContent = `${baseText}`;
        }
    }
    
    updateForecastTitle(timestamp) {
        const titleElement = document.querySelector('.current-prediction-panel .subsection-title');
        if (titleElement && timestamp) {
            // Calculate start and end times (24 hours from current time)
            const startTime = new Date(timestamp);
            const endTime = new Date(timestamp.getTime() + 24 * 60 * 60 * 1000); // Add 24 hours
            
            // Format the date range in user's timezone
            const dateRange = this.timezoneManager.formatDateRange(startTime, endTime);
            
            titleElement.innerHTML = `Solar Flare Forecast Over the Next 24 Hours<br>(${dateRange})`;
        }
    }
    
    async loadCurrentForecastAndImages() {
        try {
            // Loading current forecast and images
            
            // Use the same default date/time as calendar
            const defaultDateTime = this.getDefaultDateTime();
            const targetDate = defaultDateTime.date;
            const targetHour = defaultDateTime.hour;
            
            // Convert local date/hour to UTC for data access
            const { utcDate: utcTargetDate, utcHour: utcTargetHour } = this.getUtcFromLocal(targetDate, targetHour);
            
            // Create data key for the selected date/hour
            const year = utcTargetDate.getUTCFullYear();
            const month = String(utcTargetDate.getUTCMonth() + 1).padStart(2, '0');
            const day = String(utcTargetDate.getUTCDate()).padStart(2, '0');
            const hour = String(utcTargetHour).padStart(2, '0');
            const dataKey = `${year}${month}${day}${hour}`;
            
            // Check if prediction data exists for this key
            const data = this.predictionManager.predictionData;
            if (!data || !data[dataKey]) {
                throw new Error(`No prediction data available for ${dataKey}`);
            }
            
            // Create timestamp for this data
            const timestamp = new Date(Date.UTC(year, parseInt(month) - 1, parseInt(day), parseInt(hour), 0, 0));
            
            // Update the data time display
            this.updateDataTime(timestamp);
            
            // Display the prediction
            const predictionArray = data[dataKey];
            const predictionObj = {
                o_prob: predictionArray[0],
                c_prob: predictionArray[1], 
                m_prob: predictionArray[2],
                x_prob: predictionArray[3]
            };
            
            // Update both current status and forecast panels (use UTC date for data operations)
            this.updateCurrentForecast(utcTargetDate, utcTargetHour);
            
            // Load 4 images going backwards from this timestamp
            await this.loadAIA304ImagesFromTimestamp(timestamp);
            
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
        
        const majorStatus = this.translationManager.t('flare_status_major');
        const activeStatus = this.translationManager.t('flare_status_active');
        const eruptiveStatus = this.translationManager.t('flare_status_eruptive');
        const quietStatus = this.translationManager.t('flare_status_quiet');
        
        if (maxProb === xProb && xProb > 0.1) {
            return { level: 4, status: majorStatus, statusClass: 'status-major', flareClass: 'X-class', icon: '🔴', color: '#ff6b6b' };
        } else if (maxProb === mProb && mProb > 0.05) {
            return { level: 3, status: activeStatus, statusClass: 'status-active', flareClass: 'M-class', icon: '🟡', color: '#ffa726' };
        } else if (maxProb === cProb && cProb > 0.1) {
            return { level: 2, status: eruptiveStatus, statusClass: 'status-eruptive', flareClass: 'C-class', icon: '🟢', color: '#81c784' };
        } else {
            return { level: 1, status: quietStatus, statusClass: 'status-quiet', flareClass: 'O-class', icon: '🟢', color: '#4caf50' };
        }
    }
    
    updateLevelBlocks(level, prediction = null) {
        const blocksContainer = document.getElementById('flare-level-blocks');
        if (!blocksContainer) return;
        
        // Clear existing blocks
        blocksContainer.innerHTML = '';
        blocksContainer.className = `flare-level-blocks level-${level}`;
        
        const levelInfo = [
            { 
                label: this.translationManager.t('flare_status_major'), 
                level: this.translationManager.t('flare_level_x'), 
                className: 'x-class', key: 'x_prob', baseColor: [255, 107, 107] 
            },
            { 
                label: this.translationManager.t('flare_status_active'), 
                level: this.translationManager.t('flare_level_m'), 
                className: 'm-class', key: 'm_prob', baseColor: [255, 167, 38] 
            },
            { 
                label: this.translationManager.t('flare_status_eruptive'), 
                level: this.translationManager.t('flare_level_c'), 
                className: 'c-class', key: 'c_prob', baseColor: [129, 199, 132] 
            },
            { 
                label: this.translationManager.t('flare_status_quiet'), 
                level: this.translationManager.t('flare_level_o'), 
                className: 'o-class', key: 'o_prob', baseColor: [76, 175, 80] 
            }
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
                    <span style="font-weight: 700; color: ${textColor}; text-shadow: ${textShadow}; line-height: 1;">${info.label} (${info.level})</span>
                    <span style="font-weight: 800; color: ${textColor}; text-shadow: ${textShadow}; line-height: 1;">${percentageText}</span>
                </div>
            `;
            
            blocksContainer.appendChild(block);
        }
    }
    
    async loadAIA304ImagesFromTimestamp(baseTimestamp) {
        const container = document.getElementById('aia-304-container');
        if (!container) return;
        
        // Create unique request ID to handle race conditions
        const requestId = Date.now() + Math.random();
        this.currentAIA304RequestId = requestId;
        
        this.aia304Canvases = [];
        this.loadedTimes = [];
        
        // Loading AIA 304 images
        
        // Load 4 images going backwards from the base timestamp
        for (let i = 3; i >= 0; i--) { // Start from 3 hours back, go to current (oldest to newest)
            const timestamp = new Date(baseTimestamp.getTime() - i * 60 * 60 * 1000); // Go back i hours
            const month = String(timestamp.getUTCMonth() + 1).padStart(2, '0');
            const day = String(timestamp.getUTCDate()).padStart(2, '0');
            const hour = String(timestamp.getUTCHours()).padStart(2, '0');
            
            const imagePath = `${this.solarImagesManager.basePath}/data/images/${month}${day}/${hour}_aia_0304.png`;
            
            try {
                const canvas = await this.loadAndProcessAIA304Image(imagePath);
                if (canvas) {
                    this.aia304Canvases.push(canvas); // Add to end for chronological order (oldest first)
                    this.loadedTimes.push(`${month}/${day} ${hour}:00`);
                    // Image loaded and processed successfully
                }
            } catch (error) {
                // Image not found (expected for some timestamps)
            }
        }
        
        // Check if this request is still the latest one
        if (this.currentAIA304RequestId !== requestId) {
            // AIA 304 request superseded, skipping display update
            return;
        }
        
        // Update solar activity period display
        this.updateSolarActivityPeriod();
        
        // Display canvases
        container.innerHTML = '';
        
        // Add all canvases with proper overlay positioning
        this.aia304Canvases.forEach((canvas, index) => {
            canvas.className = 'aia-304-canvas';
            canvas.classList.toggle('active', index === 0);
            // Use absolute positioning within the container for overlay effect
            canvas.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 100%;
                height: 100%;
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                display: ${index === 0 ? 'block' : 'none'};
            `;
            container.appendChild(canvas);
        });
        
        // Add copyright directly to container (positioned relative to actual image)
        const copyright = document.createElement('div');
        copyright.className = 'aia-304-copyright';
        copyright.textContent = 'SDO©NASA';
        
        // Position copyright relative to the actual image area
        setTimeout(() => {
            this.positionCopyrightOnImage(copyright, container);
        }, 100); // Small delay to ensure container is rendered
        
        container.appendChild(copyright);
        
        // Add resize listener to reposition copyright when container size changes
        const resizeObserver = new ResizeObserver(() => {
            // Add small delay to ensure container has finished resizing
            setTimeout(() => {
                this.positionCopyrightOnImage(copyright, container);
            }, 50);
        });
        resizeObserver.observe(container);
        
        // Images loaded successfully
        
        // Final check before starting playback
        if (this.currentAIA304RequestId !== requestId) {
            // AIA 304 request superseded before playback, skipping
            return;
        }
        
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
            
            // Convert time strings to proper dates and format in user's timezone to match other panels
            try {
                // Parse the MM/DD HH:MM format from loadedTimes
                // Use the year from the UTC data to ensure consistency with data timestamps
                const utcDate = new Date(Date.UTC(this.currentDate.getFullYear(), this.currentDate.getMonth(), this.currentDate.getDate()));
                const currentYear = utcDate.getUTCFullYear();
                const startParts = startTime.split(' ');
                const startDate = startParts[0]; // MM/DD
                const startHour = startParts[1]; // HH:MM
                
                const endParts = endTime.split(' ');
                const endDate = endParts[0]; // MM/DD
                const endHour = endParts[1]; // HH:MM
                
                const [startMonth, startDay] = startDate.split('/');
                const [startH] = startHour.split(':');
                
                const [endMonth, endDay] = endDate.split('/');
                const [endH] = endHour.split(':');
                
                const startUTC = new Date(Date.UTC(currentYear, parseInt(startMonth) - 1, parseInt(startDay), parseInt(startH), 0, 0));
                const endUTC = new Date(Date.UTC(currentYear, parseInt(endMonth) - 1, parseInt(endDay), parseInt(endH), 0, 0));
                
                const timeRange = this.timezoneManager.formatDateRange(startUTC, endUTC);
                periodElement.textContent = `(${timeRange})`;
            } catch (error) {
                // Fallback to original format with timezone abbreviation
                const timezone = this.timezoneManager.getTimezoneAbbreviation();
                periodElement.textContent = `(${startTime} - ${endTime} ${timezone})`;
            }
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
        
        // Starting auto playback
        
        // Start continuous loop
        this.autoPlayInterval = setInterval(() => {
            this.currentFrame = (this.currentFrame + 1) % this.aia304Canvases.length;
            this.updateAutoDisplay();
        }, 1000); // 1 FPS for smooth viewing
    }
    
    updateAutoDisplay() {
        // Update active canvas using display property instead of active class
        this.aia304Canvases.forEach((canvas, index) => {
            if (index === this.currentFrame) {
                canvas.style.display = 'block';
                canvas.classList.add('active');
            } else {
                canvas.style.display = 'none';
                canvas.classList.remove('active');
            }
        });
        
        // console.log('Displaying frame:', this.currentFrame + 1, '/', this.aia304Canvases.length);
    }
    
    stopAutoPlayback() {
        if (this.autoPlayInterval) {
            clearInterval(this.autoPlayInterval);
            this.autoPlayInterval = null;
        }
    }
    
    displayCurrentForecastError() {
        // Set default quiet status for both panels
        const defaultStatus = {
            level: 1,
            status: this.translationManager.t('flare_status_quiet'),
            statusClass: 'status-quiet',
            flareClass: 'O-class',
            icon: '🟢',
            color: '#4caf50'
        };
        
        // Update current status panel
        this.updateStatusPanel(defaultStatus, 'current');
        
        // Update forecast panel with default prediction
        const defaultPrediction = {
            o_prob: 1.0,
            c_prob: 0.0,
            m_prob: 0.0,
            x_prob: 0.0
        };
        this.updateForecastPanel(defaultPrediction);
    }
    
    updateCurrentForecast(date, hour) {
        // Update the data time display
        const timestamp = new Date(date);
        timestamp.setUTCHours(hour, 0, 0, 0);
        this.updateDataTime(timestamp);
        
        // Update panel titles based on selected date
        this.updatePanelTitles(timestamp);
        
        // Update forecast period display
        this.updateForecastPeriod(timestamp);
        
        // Update current status time display
        this.updateCurrentStatusTime(timestamp);
        
        // Get prediction data for the selected date/hour
        const year = date.getUTCFullYear();
        const month = String(date.getUTCMonth() + 1).padStart(2, '0');
        const day = String(date.getUTCDate()).padStart(2, '0');
        const hourStr = String(hour).padStart(2, '0');
        const dataKey = `${year}${month}${day}${hourStr}`;
        
        // Update current status
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
            const timeStr = this.timezoneManager.formatTimeWithTimezone(timestamp);
            statusTimeEl.textContent = `(${timeStr})`;
        }
    }
    
    updateForecastPeriod(timestamp) {
        // Update 24-hour forecast period
        const startTime = new Date(timestamp);
        const endTime = new Date(timestamp.getTime() + 24 * 60 * 60 * 1000);
        
        const dateRange = this.timezoneManager.formatDateRange(startTime, endTime);
        
        const forecastPeriodEl = document.getElementById('forecast-period');
        if (forecastPeriodEl) {
            forecastPeriodEl.textContent = `(${dateRange})`;
        }
    }
    
    formatTimeForDisplay(timestamp) {
        const month = String(timestamp.getUTCMonth() + 1);
        const day = String(timestamp.getUTCDate());
        const hour = String(timestamp.getUTCHours()).padStart(2, '0');
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
        
        const currentTime = new Date(Date.UTC(year, month - 1, day, hour, 0, 0));
        const startTime = new Date(currentTime.getTime() - 24 * 60 * 60 * 1000); // 24 hours ago
        
        let maxFlux = 0;
        
        // Check all data points in the past 24 hours
        for (let i = 0; i <= 24; i++) {
            const checkTime = new Date(startTime.getTime() + i * 60 * 60 * 1000); // Each hour
            const checkKey = `${checkTime.getUTCFullYear()}${String(checkTime.getUTCMonth() + 1).padStart(2, '0')}${String(checkTime.getUTCDate()).padStart(2, '0')}${String(checkTime.getUTCHours()).padStart(2, '0')}`;
            
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
            
            const dateRange = this.timezoneManager.formatDateRange(startTime, endTime);
            
            statusTimeEl.textContent = `(${dateRange})`;
        }
    }
    
    getFlareClassFromFlux(flux) {
        if (flux >= 1e-4) return 'X';
        if (flux >= 1e-5) return 'M';
        if (flux >= 1e-6) return 'C';
        return 'O';
    }
    
    getStatusFromFlareClass(flareClass) {
        const majorStatus = this.translationManager.t('flare_status_major');
        const activeStatus = this.translationManager.t('flare_status_active');
        const eruptiveStatus = this.translationManager.t('flare_status_eruptive');
        const quietStatus = this.translationManager.t('flare_status_quiet');
        
        switch (flareClass) {
            case 'X':
                return { level: 4, status: majorStatus, statusClass: 'status-major', flareClass: 'X-class', icon: '🔴', color: '#ff6b6b' };
            case 'M':
                return { level: 3, status: activeStatus, statusClass: 'status-active', flareClass: 'M-class', icon: '🟡', color: '#ffa726' };
            case 'C':
                return { level: 2, status: eruptiveStatus, statusClass: 'status-eruptive', flareClass: 'C-class', icon: '🟢', color: '#81c784' };
            default:
                return { level: 1, status: quietStatus, statusClass: 'status-quiet', flareClass: 'O-class', icon: '🟢', color: '#4caf50' };
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
            const statusTextElement = statusElement.querySelector('.status-text');
            statusTextElement.innerHTML = `${statusInfo.icon} ${statusInfo.status}`;
            statusTextElement.removeAttribute('data-i18n'); // Remove translation attribute
            statusElement.querySelector('.level-text').textContent = `Lv.${statusInfo.level} (${statusInfo.flareClass})`;
            
            // Apply color to the icon
            statusTextElement.style.color = statusInfo.color;
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
            const statusTextElement = statusElement.querySelector('.status-text');
            statusTextElement.innerHTML = `${icon} ${status}`;
            statusTextElement.removeAttribute('data-i18n'); // Remove translation attribute
            statusElement.querySelector('.level-text').textContent = `Lv.${level} (${flareClass})`;
            
            // Apply color to the icon
            statusTextElement.style.color = color;
        }
    }
    
    updateLevelBlocksNew(level, prediction = null, containerId) {
        const blocksContainer = document.getElementById(containerId);
        if (!blocksContainer) return;
        
        // Clear existing blocks
        blocksContainer.innerHTML = '';
        
        const levelInfo = [
            { 
                label: this.translationManager.t('flare_status_major'), 
                level: 'Lv.4', 
                className: 'x-class', key: 'x_prob', baseColor: [255, 107, 107] 
            },
            { 
                label: this.translationManager.t('flare_status_active'), 
                level: 'Lv.3', 
                className: 'm-class', key: 'm_prob', baseColor: [255, 167, 38] 
            },
            { 
                label: this.translationManager.t('flare_status_eruptive'), 
                level: 'Lv.2', 
                className: 'c-class', key: 'c_prob', baseColor: [129, 199, 132] 
            },
            { 
                label: this.translationManager.t('flare_status_quiet'), 
                level: 'Lv.1', 
                className: 'o-class', key: 'o_prob', baseColor: [76, 175, 80] 
            }
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
            
            // Each level uses its own color
            const [r, g, b] = info.baseColor;
            
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
                <span style="font-weight: 600; color: ${textColor};">${info.label} (${info.level})</span>
                <span style="font-weight: 700; color: ${textColor};">${percentageText}</span>
            `;
            
            blocksContainer.appendChild(block);
        }
    }
    
    displayForecastError() {
        // Set default quiet forecast
        const defaultPrediction = {
            o_prob: 1.0,
            c_prob: 0.0,
            m_prob: 0.0,
            x_prob: 0.0
        };
        this.updateForecastPanel(defaultPrediction);
    }
    
    refreshDynamicContent() {
        // Refresh performance displays
        // Month performance depends on current date, all period performance is independent
        if (this.predictionManager && this.currentDate) {
            this.predictionManager.updatePerformanceDisplays(this.currentDate);
        }
        
        // Update panel titles with current language
        if (this.currentDate && this.currentHour) {
            // Convert local date to UTC for data operations
            const { utcDate, utcHour } = this.getSelectedUtcParts();
            this.updateCurrentForecast(utcDate, utcHour);
        }
        
        // Update page content with current translations
        this.updatePageContent();
    }
    
    updatePageContent() {
        // Update all translatable elements
        if (this.translationManager) {
            const translatableElements = document.querySelectorAll('[data-i18n]');
            translatableElements.forEach(element => {
                const key = element.getAttribute('data-i18n');
                const translation = this.translationManager.t(key);
                
                // Skip updating status elements that have been updated with actual status
                // Check if the element contains an icon emoji (indicating it has been set)
                if (key === 'loading' && element.classList.contains('status-text')) {
                    const currentContent = element.innerHTML || element.textContent || '';
                    // If content contains emoji or has been set to actual status, skip translation
                    if (currentContent.match(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/u) ||
                        (!currentContent.includes('Loading') && !currentContent.includes('読み込み中') && currentContent !== '--')) {
                        return; // Skip updating this element
                    }
                }
                
                if (translation) {
                    if (translation.includes('<br/>')) {
                        element.innerHTML = translation;
                    } else {
                        element.textContent = translation;
                    }
                }
            });
        }
    }
    
    initImageModals() {
        // Add click event listeners to clickable images
        const clickableImages = document.querySelectorAll('.clickable-image');
        clickableImages.forEach(image => {
            image.addEventListener('click', (e) => {
                const modalTarget = e.target.getAttribute('data-modal-target');
                if (modalTarget) {
                    this.openModal(modalTarget);
                }
            });
        });
        
        // Add click event listeners to modal close buttons
        const closeButtons = document.querySelectorAll('.modal-close');
        closeButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                this.closeModal(e.target.closest('.image-modal').id);
            });
        });
        
        // Add click event listeners to modal backdrops
        const modalBackdrops = document.querySelectorAll('.modal-backdrop');
        modalBackdrops.forEach(backdrop => {
            backdrop.addEventListener('click', (e) => {
                this.closeModal(e.target.closest('.image-modal').id);
            });
        });
        
        // Add ESC key listener to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const openModal = document.querySelector('.image-modal.show');
                if (openModal) {
                    this.closeModal(openModal.id);
                }
            }
        });
    }
    
    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('show');
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
        }
    }
    
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('show');
            document.body.style.overflow = ''; // Restore scrolling
        }
    }
    
    updatePanelTitles(selectedTimestamp) {
        // Check if the selected date is "current" (today's latest data)
        // Compare using local dates to match user's perception
        const today = new Date();
        const selectedLocal = new Date(selectedTimestamp.getUTCFullYear(), selectedTimestamp.getUTCMonth(), selectedTimestamp.getUTCDate());
        const todayLocal = new Date(today.getFullYear(), today.getMonth(), today.getDate());
        const isToday = selectedLocal.getTime() === todayLocal.getTime();
        
        // Update solar surface title
        const solarTitleEl = document.getElementById('solar-activity-title');
        if (solarTitleEl) {
            if (isToday) {
                solarTitleEl.textContent = this.translationManager.t('current_solar_surface');
            } else {
                // For past dates, just use "Solar Surface" without "Current"
                solarTitleEl.textContent = this.translationManager.t('current_solar_surface').replace('Current ', '').replace('現在の', '');
            }
        }
        
        // Update solar flare status title  
        const statusTitleEl = document.querySelector('[data-i18n="current_solar_flare_status"]');
        if (statusTitleEl) {
            if (isToday) {
                statusTitleEl.textContent = this.translationManager.t('current_solar_flare_status');
            } else {
                // For past dates, just use "Solar Flare Status" without "Current"
                statusTitleEl.textContent = this.translationManager.t('current_solar_flare_status').replace('Current ', '').replace('現在の', '');
            }
        }
    }
    
    updateInitialPanelTitles() {
        // Set initial titles to "current" version
        const solarTitleEl = document.getElementById('solar-activity-title');
        if (solarTitleEl) {
            solarTitleEl.textContent = this.translationManager.t('current_solar_surface');
        }
        
        const statusTitleEl = document.querySelector('[data-i18n="current_solar_flare_status"]');
        if (statusTitleEl) {
            statusTitleEl.textContent = this.translationManager.t('current_solar_flare_status');
        }
    }
    
    positionCopyrightOnImage(copyright, container) {
        // Get container dimensions
        const containerRect = container.getBoundingClientRect();
        const containerStyle = window.getComputedStyle(container);
        const containerWidth = containerRect.width;
        const containerHeight = containerRect.height;
        
        // Since the container has aspect-ratio: 1 and centers the image with flex,
        // the actual image area is the smaller dimension (width or height)
        const imageSize = Math.min(containerWidth, containerHeight);
        
        // Calculate offsets to center the square image within the container
        const horizontalOffset = (containerWidth - imageSize) / 2;
        const verticalOffset = (containerHeight - imageSize) / 2;
        
        // Adjust position and font size for different devices
        const screenWidth = window.innerWidth;
        const isIPad = (screenWidth >= 768 && screenWidth <= 1366) || navigator.userAgent.includes('iPad');
        const isIPhone = screenWidth <= 480; // iPhone size
        const bottomMargin = isIPad ? 16 : 4; // Reduced margins for closer positioning
        const rightMargin = 4; // Reduced right margin
        const fontSize = isIPhone ? '0.45rem' : '0.6rem'; // Smaller font for iPhone
        
        // Position relative to the actual image bounds, closer to the edge
        const bottomPosition = verticalOffset + bottomMargin;
        const rightPosition = horizontalOffset + rightMargin;
        
        copyright.style.cssText = `
            position: absolute;
            bottom: ${bottomPosition}px;
            right: ${rightPosition}px;
            z-index: 1000;
            background: transparent;
            color: #fff;
            font-size: ${fontSize};
            font-weight: 700;
            font-family: "Kanit", sans-serif;
            pointer-events: none;
            letter-spacing: 0.5px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        `;
        
        // Copyright positioned
    }
    
    applyCopyrightPositioning(copyright) {
        // For AIA 304 copyright, always keep it at bottom-right of image
        // Let CSS handle all device-specific positioning
        // AIA 304 copyright positioning handled by CSS
    }

}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SolarFlareDemo();
});
