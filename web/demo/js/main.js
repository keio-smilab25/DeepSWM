// Main application logic

class SolarFlareDemo {
    constructor() {
        this.translationManager = new window.TranslationManager();
        this.solarImagesManager = new window.SolarImagesManager();
        this.predictionManager = new window.PredictionManager();
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
                        this.currentDate = new Date(dateStr + 'T00:00:00');
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
}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SolarFlareDemo();
});
