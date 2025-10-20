// Timezone Management System for Dynamic Timezone Detection and Display

class TimezoneManager {
    constructor() {
        this.userTimezone = this.detectUserTimezone();
        this.timezoneOffset = this.calculateTimezoneOffset();
        this.timezoneDisplayName = this.getTimezoneDisplayName();
    }

    /**
     * Detect user's timezone using Intl.DateTimeFormat
     * @returns {string} User's timezone (e.g., 'Asia/Tokyo', 'America/New_York')
     */
    detectUserTimezone() {
        try {
            return Intl.DateTimeFormat().resolvedOptions().timeZone;
        } catch (error) {
            console.warn('Failed to detect user timezone, falling back to UTC:', error);
            return 'UTC';
        }
    }

    /**
     * Calculate timezone offset in minutes from UTC
     * @returns {number} Offset in minutes (negative for timezones behind UTC)
     */
    calculateTimezoneOffset() {
        const now = new Date();
        return -now.getTimezoneOffset(); // getTimezoneOffset returns negative for ahead of UTC
    }

    /**
     * Get human-readable timezone display name
     * @param {string} locale - Locale for display name (default: 'en')
     * @returns {string} Display name (e.g., 'Japan Standard Time', 'Eastern Standard Time')
     */
    getTimezoneDisplayName(locale = 'en') {
        try {
            const formatter = new Intl.DateTimeFormat(locale, {
                timeZoneName: 'long',
                timeZone: this.userTimezone
            });
            
            const parts = formatter.formatToParts(new Date());
            const timeZonePart = parts.find(part => part.type === 'timeZoneName');
            return timeZonePart ? timeZonePart.value : this.userTimezone;
        } catch (error) {
            console.warn('Failed to get timezone display name:', error);
            return this.userTimezone;
        }
    }

    /**
     * Get short timezone abbreviation (e.g., 'JST', 'EST', 'PST')
     * @returns {string} Timezone abbreviation
     */
    getTimezoneAbbreviation() {
        try {
            // Use both summer and winter dates to get the most appropriate abbreviation
            const now = new Date();
            const summer = new Date(now.getFullYear(), 6, 1); // July 1st
            const winter = new Date(now.getFullYear(), 0, 1); // January 1st
            
            // Try current date first
            let formatter = new Intl.DateTimeFormat('en', {
                timeZoneName: 'short',
                timeZone: this.userTimezone
            });
            
            let parts = formatter.formatToParts(now);
            let timeZonePart = parts.find(part => part.type === 'timeZoneName');
            
            if (timeZonePart && timeZonePart.value && !timeZonePart.value.startsWith('GMT')) {
                return timeZonePart.value;
            }
            
            // If current date gives GMT+X format, try summer date
            parts = formatter.formatToParts(summer);
            timeZonePart = parts.find(part => part.type === 'timeZoneName');
            
            if (timeZonePart && timeZonePart.value && !timeZonePart.value.startsWith('GMT')) {
                return timeZonePart.value;
            }
            
            // If still GMT+X format, try winter date
            parts = formatter.formatToParts(winter);
            timeZonePart = parts.find(part => part.type === 'timeZoneName');
            
            if (timeZonePart && timeZonePart.value && !timeZonePart.value.startsWith('GMT')) {
                return timeZonePart.value;
            }
            
            // If all attempts return GMT+X format, try to map common timezones
            return this.mapTimezoneToAbbreviation();
            
        } catch (error) {
            console.warn('Failed to get timezone abbreviation:', error);
            return 'UTC';
        }
    }

    /**
     * Map common timezone identifiers to their abbreviations
     * @returns {string} Timezone abbreviation
     */
    mapTimezoneToAbbreviation() {
        const timezoneMap = {
            // Asia
            'Asia/Tokyo': 'JST',
            'Asia/Seoul': 'KST',
            'Asia/Shanghai': 'CST',
            'Asia/Hong_Kong': 'HKT',
            'Asia/Singapore': 'SGT',
            'Asia/Bangkok': 'ICT',
            'Asia/Jakarta': 'WIB',
            'Asia/Manila': 'PHT',
            'Asia/Taipei': 'CST',
            'Asia/Kolkata': 'IST',
            'Asia/Dubai': 'GST',
            
            // Europe
            'Europe/London': 'GMT',
            'Europe/Paris': 'CET',
            'Europe/Berlin': 'CET',
            'Europe/Rome': 'CET',
            'Europe/Madrid': 'CET',
            'Europe/Amsterdam': 'CET',
            'Europe/Brussels': 'CET',
            'Europe/Vienna': 'CET',
            'Europe/Zurich': 'CET',
            'Europe/Stockholm': 'CET',
            'Europe/Moscow': 'MSK',
            
            // Americas (will be adjusted for DST)
            'America/New_York': 'ET',
            'America/Chicago': 'CT', 
            'America/Denver': 'MT',
            'America/Los_Angeles': 'PT',
            'America/Toronto': 'ET',
            'America/Vancouver': 'PT',
            'America/Mexico_City': 'CST',
            'America/Sao_Paulo': 'BRT',
            'America/Argentina/Buenos_Aires': 'ART',
            
            // Pacific
            'Pacific/Auckland': 'NZST',
            'Pacific/Sydney': 'AEST',
            'Pacific/Melbourne': 'AEST',
            'Australia/Sydney': 'AEST',
            'Australia/Melbourne': 'AEST',
            'Australia/Perth': 'AWST',
            
            // Africa
            'Africa/Cairo': 'EET',
            'Africa/Johannesburg': 'SAST',
            
            // UTC
            'UTC': 'UTC',
            'GMT': 'GMT'
        };
        
        const baseAbbr = timezoneMap[this.userTimezone];
        if (baseAbbr) {
            // For US timezones, check if we need DST adjustment
            if (this.userTimezone.startsWith('America/') && ['ET', 'CT', 'MT', 'PT'].includes(baseAbbr)) {
                return this.getDSTAwareAbbreviation(baseAbbr);
            }
            return baseAbbr;
        }
        
        return this.getOffsetBasedAbbreviation();
    }

    /**
     * Get DST-aware abbreviation for US timezones
     * @param {string} baseAbbr - Base abbreviation (ET, CT, MT, PT)
     * @returns {string} DST-aware timezone abbreviation
     */
    getDSTAwareAbbreviation(baseAbbr) {
        try {
            const now = new Date();
            const january = new Date(now.getFullYear(), 0, 1);
            const july = new Date(now.getFullYear(), 6, 1);
            
            const janOffset = this.getTimezoneOffsetForDate(january);
            const julyOffset = this.getTimezoneOffsetForDate(july);
            const currentOffset = this.getTimezoneOffsetForDate(now);
            
            // If July offset is different from January, timezone observes DST
            const observesDST = janOffset !== julyOffset;
            
            if (!observesDST) {
                // No DST, return standard time abbreviation
                const standardMap = { 'ET': 'EST', 'CT': 'CST', 'MT': 'MST', 'PT': 'PST' };
                return standardMap[baseAbbr] || baseAbbr;
            }
            
            // Determine if currently in DST (summer time has smaller offset)
            const isDST = currentOffset < janOffset;
            
            if (isDST) {
                const dstMap = { 'ET': 'EDT', 'CT': 'CDT', 'MT': 'MDT', 'PT': 'PDT' };
                return dstMap[baseAbbr] || baseAbbr;
            } else {
                const standardMap = { 'ET': 'EST', 'CT': 'CST', 'MT': 'MST', 'PT': 'PST' };
                return standardMap[baseAbbr] || baseAbbr;
            }
            
        } catch (error) {
            console.warn('Failed to determine DST status:', error);
            return baseAbbr;
        }
    }

    /**
     * Get timezone offset for a specific date
     * @param {Date} date - Date to get offset for
     * @returns {number} Timezone offset in minutes
     */
    getTimezoneOffsetForDate(date) {
        try {
            const utc = new Date(date.getTime() + (date.getTimezoneOffset() * 60000));
            const local = new Date(utc.toLocaleString('en-US', { timeZone: this.userTimezone }));
            return (utc.getTime() - local.getTime()) / 60000;
        } catch (error) {
            return date.getTimezoneOffset();
        }
    }

    /**
     * Get timezone abbreviation based on offset as fallback
     * @returns {string} Offset-based timezone abbreviation
     */
    getOffsetBasedAbbreviation() {
        const offsetMinutes = this.timezoneOffset;
        const offsetHours = Math.floor(Math.abs(offsetMinutes) / 60);
        const offsetMins = Math.abs(offsetMinutes) % 60;
        
        if (offsetMinutes === 0) {
            return 'UTC';
        }
        
        const sign = offsetMinutes > 0 ? '+' : '-';
        const hoursStr = String(offsetHours).padStart(2, '0');
        const minsStr = offsetMins > 0 ? `:${String(offsetMins).padStart(2, '0')}` : '';
        
        return `UTC${sign}${hoursStr}${minsStr}`;
    }

    /**
     * Convert UTC date to user's local timezone
     * @param {Date} utcDate - UTC date to convert
     * @returns {Date} Date in user's timezone
     */
    convertUTCToLocal(utcDate) {
        if (!utcDate || !(utcDate instanceof Date)) {
            return new Date();
        }
        
        try {
            // Create a new date in the user's timezone
            return new Date(utcDate.toLocaleString('en-US', { timeZone: this.userTimezone }));
        } catch (error) {
            console.warn('Failed to convert UTC to local time:', error);
            return utcDate;
        }
    }

    /**
     * Format date for display in user's timezone
     * @param {Date} utcDate - UTC date to format
     * @param {Object} options - Formatting options
     * @returns {string} Formatted date string
     */
    formatDateInTimezone(utcDate, options = {}) {
        if (!utcDate || !(utcDate instanceof Date)) {
            return '';
        }

        const defaultOptions = {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            timeZone: this.userTimezone,
            hour12: false
        };

        const formatOptions = { ...defaultOptions, ...options };

        try {
            return utcDate.toLocaleString('en-CA', formatOptions);
        } catch (error) {
            console.warn('Failed to format date in timezone:', error);
            return utcDate.toISOString().slice(0, 16).replace('T', ' ');
        }
    }

    /**
     * Format time for display with timezone abbreviation
     * @param {Date} utcDate - UTC date to format
     * @param {boolean} includeSeconds - Whether to include seconds
     * @returns {string} Formatted time string with timezone
     */
    formatTimeWithTimezone(utcDate, includeSeconds = false) {
        if (!utcDate || !(utcDate instanceof Date)) {
            return '';
        }

        const timeOptions = {
            hour: '2-digit',
            minute: '2-digit',
            timeZone: this.userTimezone,
            hour12: false
        };

        if (includeSeconds) {
            timeOptions.second = '2-digit';
        }

        try {
            const timeStr = utcDate.toLocaleString('en-CA', timeOptions);
            const timezone = this.getTimezoneAbbreviation();
            return `${timeStr} ${timezone}`;
        } catch (error) {
            console.warn('Failed to format time with timezone:', error);
            const fallbackTime = utcDate.toISOString().slice(11, includeSeconds ? 19 : 16);
            return `${fallbackTime} UTC`;
        }
    }

    /**
     * Format date range for display
     * @param {Date} startUTC - Start date in UTC
     * @param {Date} endUTC - End date in UTC
     * @returns {string} Formatted date range string
     */
    formatDateRange(startUTC, endUTC) {
        if (!startUTC || !endUTC) {
            return '';
        }

        try {
            const formatWithoutYear = (utcDate) => {
                const localDate = this.convertUTCToLocal(utcDate);
                const month = String(localDate.getMonth() + 1).padStart(2, '0');
                const day = String(localDate.getDate()).padStart(2, '0');
                const hour = String(localDate.getHours()).padStart(2, '0');
                const minute = String(localDate.getMinutes()).padStart(2, '0');
                return `${month}/${day} ${hour}:${minute}`;
            };

            const startStr = formatWithoutYear(startUTC);
            const endStr = formatWithoutYear(endUTC);
            
            const timezone = this.getTimezoneAbbreviation();
            return `${startStr} - ${endStr} ${timezone}`;
        } catch (error) {
            console.warn('Failed to format date range:', error);
            return `${startUTC.toISOString().slice(5, 16)} - ${endUTC.toISOString().slice(5, 16)} UTC`;
        }
    }

    /**
     * Check if user is in UTC timezone
     * @returns {boolean} True if user is in UTC
     */
    isUTC() {
        return this.userTimezone === 'UTC' || this.timezoneOffset === 0;
    }

    /**
     * Get timezone info for display
     * @returns {Object} Timezone information
     */
    getTimezoneInfo() {
        return {
            timezone: this.userTimezone,
            offset: this.timezoneOffset,
            displayName: this.timezoneDisplayName,
            abbreviation: this.getTimezoneAbbreviation(),
            isUTC: this.isUTC()
        };
    }

    /**
     * Convert local date back to UTC for API calls
     * @param {Date} localDate - Local date to convert
     * @returns {Date} UTC date
     */
    convertLocalToUTC(localDate) {
        if (!localDate || !(localDate instanceof Date)) {
            return new Date();
        }

        try {
            // Get the time in milliseconds and adjust for timezone offset
            const utcTime = localDate.getTime() - (this.timezoneOffset * 60000);
            return new Date(utcTime);
        } catch (error) {
            console.warn('Failed to convert local to UTC time:', error);
            return localDate;
        }
    }
}

// Make TimezoneManager available globally
window.TimezoneManager = TimezoneManager;
