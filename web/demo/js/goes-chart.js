// GOES X-ray Flux Chart Manager

class GOESChartManager {
    constructor() {
        this.chartInstance = null;
        this.basePath = this.getBasePath();
        this.xrsDataMap = {};
        this.currentBaseTime = null;
        this.animationTimer = null;
        this.currentFrame = 0;
        this.isAnimating = false;
        
        // Flare class thresholds and colors
        this.flareThresholds = {
            X: 1e-4,
            M: 1e-5,
            C: 1e-6,
            O: 1e-9
        };
        
        this.flareColors = {
            X: 'rgba(255, 0, 0, 0.15)',
            M: 'rgba(255, 165, 0, 0.15)',
            C: 'rgba(0, 255, 0, 0.15)',
            O: 'rgba(0, 0, 255, 0.15)'
        };
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
    
    async loadXRSData() {
        try {
            const response = await fetch(`${this.basePath}/data/xrs.json`);
            if (response.ok) {
                this.xrsDataMap = await response.json();
                console.log('XRS data loaded:', Object.keys(this.xrsDataMap).length, 'entries');
                return true;
            } else {
                console.warn('Failed to load XRS data:', response.status);
                this.xrsDataMap = {};
                return false;
            }
        } catch (error) {
            console.warn('Error loading XRS data:', error);
            this.xrsDataMap = {};
            return false;
        }
    }
    
    computeClassFromFlux(flux) {
        if (flux < 1e-6) return 0; // O
        if (flux < 1e-5) return 1; // C
        if (flux < 1e-4) return 2; // M
        return 3; // X
    }
    
    getXRSDataForTimeRange(baseTime) {
        const data = [];
        console.log('Getting XRS data for base time:', baseTime.toISOString());
        console.log('Available XRS keys:', Object.keys(this.xrsDataMap).slice(0, 10), '... (showing first 10)');
        
        // Load data for 24 hours from selected time
        for (let i = 0; i < 24; i++) {
            const t = new Date(baseTime.getTime() + i * 3600 * 1000);
            const key = `${t.getUTCFullYear()}${String(t.getUTCMonth() + 1).padStart(2, '0')}` +
                       `${String(t.getUTCDate()).padStart(2, '0')}${String(t.getUTCHours()).padStart(2, '0')}`;
            const flux = this.xrsDataMap[key];
            data.push(flux != null ? flux : null);
            
            if (i < 3) { // Log first few entries for debugging
                console.log(`Hour ${i}: key=${key}, flux=${flux}`);
            }
        }
        
        console.log('XRS data array length:', data.length, 'non-null entries:', data.filter(d => d !== null).length);
        return data;
    }
    
    isMobileView() {
        return window.innerWidth <= 600;
    }
    
    getYAxisTicksDisplay() {
        return !this.isMobileView();
    }
    
    async updateChart(baseTime) {
        this.currentBaseTime = baseTime;
        
        // Load XRS data if not already loaded
        if (Object.keys(this.xrsDataMap).length === 0) {
            await this.loadXRSData();
        }
        
        const flareData = this.getXRSDataForTimeRange(baseTime);
        const labels = Array.from({ length: 24 }, (_, i) => `+${i}h`);
        
        // Generate point colors based on flux values
        const pointColors = flareData.map(v => {
            if (v == null) return 'gray';
            if (v < 1e-6) return 'blue';   // O class
            if (v < 1e-5) return 'green';  // C class
            if (v < 1e-4) return 'orange'; // M class
            return 'red'; // X class
        });
        
        const ctx = document.getElementById('goesChart');
        if (!ctx) {
            console.error('GOES chart canvas not found');
            return;
        }
        
        const formattedTime = `${baseTime.getUTCFullYear()}-${String(baseTime.getUTCMonth() + 1).padStart(2, '0')}-` +
                             `${String(baseTime.getUTCDate()).padStart(2, '0')} ${String(baseTime.getUTCHours()).padStart(2, '0')}:00 UTC`;
        
        if (this.chartInstance) {
            // Update existing chart
            this.chartInstance.data.labels = labels;
            this.chartInstance.data.datasets[0].data = flareData;
            this.chartInstance.data.datasets[0].pointBackgroundColor = pointColors;
            this.chartInstance.options.plugins.annotation.annotations.startTimeLine.label.content = formattedTime;
            this.chartInstance.options.scales.y.ticks = this.chartInstance.options.scales.y.ticks || {};
            this.chartInstance.options.scales.y.ticks.display = this.getYAxisTicksDisplay();
            this.chartInstance.update();
            
            // Start animation
            this.startAnimation(flareData);
        } else {
            // Create new chart
            this.chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'X-ray Flux',
                        data: flareData,
                        borderColor: 'black',
                        pointBackgroundColor: pointColors,
                        fill: false,
                        pointRadius: 2,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: 'logarithmic',
                            min: 1e-9,
                            max: 1e-3,
                            title: { 
                                display: true, 
                                text: 'Flux (W/m²)',
                                color: document.body.classList.contains('dark-theme') ? '#f1f5f9' : '#333'
                            },
                            ticks: {
                                display: this.getYAxisTicksDisplay(),
                                color: document.body.classList.contains('dark-theme') ? '#cbd5e1' : '#666',
                                callback: function(value, index, values) {
                                    // Convert to scientific notation format like 10^-4
                                    const exponent = Math.log10(value);
                                    if (Number.isInteger(exponent)) {
                                        return `10^${exponent}`;
                                    }
                                    return '';
                                }
                            },
                            grid: {
                                color: document.body.classList.contains('dark-theme') ? 'rgba(148, 163, 184, 0.2)' : 'rgba(0, 0, 0, 0.1)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time (hours from current)',
                                color: document.body.classList.contains('dark-theme') ? '#f1f5f9' : '#333'
                            },
                            ticks: {
                                color: document.body.classList.contains('dark-theme') ? '#cbd5e1' : '#666'
                            },
                            grid: {
                                color: document.body.classList.contains('dark-theme') ? 'rgba(148, 163, 184, 0.2)' : 'rgba(0, 0, 0, 0.1)'
                            }
                        }
                    },
                    plugins: {
                        legend: { 
                            display: false 
                        },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => {
                                    const v = ctx.raw;
                                    if (v == null) return '欠損';
                                    const cls = v >= 1e-4 ? 'X'
                                              : v >= 1e-5 ? 'M'
                                              : v >= 1e-6 ? 'C'
                                              : 'O';
                                    return `Flux: ${v.toExponential(2)} W/m² (Class ${cls})`;
                                }
                            },
                            backgroundColor: document.body.classList.contains('dark-theme') ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                            titleColor: document.body.classList.contains('dark-theme') ? '#f1f5f9' : '#333',
                            bodyColor: document.body.classList.contains('dark-theme') ? '#e2e8f0' : '#666'
                        },
                        annotation: {
                            annotations: {
                                // Flare class background bands
                                flareBandX: {
                                    type: 'box',
                                    yMin: 1e-4, yMax: 1e-3,
                                    backgroundColor: this.flareColors.X,
                                    label: { 
                                        enabled: true, 
                                        content: 'X', 
                                        position: 'start', 
                                        xAdjust: 50, 
                                        backgroundColor: 'transparent', 
                                        color: 'red', 
                                        font: { weight: 'bold', size: 14 } 
                                    }
                                },
                                flareBandM: {
                                    type: 'box',
                                    yMin: 1e-5, yMax: 1e-4,
                                    backgroundColor: this.flareColors.M,
                                    label: { 
                                        enabled: true, 
                                        content: 'M', 
                                        position: 'start', 
                                        xAdjust: 50, 
                                        backgroundColor: 'transparent', 
                                        color: 'orange', 
                                        font: { weight: 'bold', size: 14 } 
                                    }
                                },
                                flareBandC: {
                                    type: 'box',
                                    yMin: 1e-6, yMax: 1e-5,
                                    backgroundColor: this.flareColors.C,
                                    label: { 
                                        enabled: true, 
                                        content: 'C', 
                                        position: 'start', 
                                        xAdjust: 50, 
                                        backgroundColor: 'transparent', 
                                        color: 'green', 
                                        font: { weight: 'bold', size: 14 } 
                                    }
                                },
                                flareBandO: {
                                    type: 'box',
                                    yMin: 1e-9, yMax: 1e-6,
                                    backgroundColor: this.flareColors.O,
                                    label: { 
                                        enabled: true, 
                                        content: 'O', 
                                        position: 'start', 
                                        xAdjust: 50, 
                                        backgroundColor: 'transparent', 
                                        color: 'blue', 
                                        font: { weight: 'bold', size: 14 } 
                                    }
                                },
                                // Current time line (start time)
                                startTimeLine: {
                                    type: 'line',
                                    scaleID: 'x', 
                                    value: 0,
                                    borderColor: 'black', 
                                    borderWidth: 3,
                                    label: { 
                                        enabled: true, 
                                        content: formattedTime, 
                                        position: 'start', 
                                        backgroundColor: 'black', 
                                        color: 'white', 
                                        font: { weight: 'bold', size: 12 } 
                                    }
                                }
                            }
                        }
                    }
                },
                plugins: [{
                    id: 'backgroundZones',
                    beforeDraw: (chart) => {
                        const { ctx, chartArea, scales } = chart;
                        const zones = [
                            { from: 1e-4, to: 1e-3, color: this.flareColors.X },
                            { from: 1e-5, to: 1e-4, color: this.flareColors.M },
                            { from: 1e-6, to: 1e-5, color: this.flareColors.C },
                            { from: 1e-9, to: 1e-6, color: this.flareColors.O }
                        ];
                        zones.forEach(z => {
                            const y1 = scales.y.getPixelForValue(z.from);
                            const y2 = scales.y.getPixelForValue(z.to);
                            ctx.fillStyle = z.color;
                            ctx.fillRect(chartArea.left, y2, chartArea.right - chartArea.left, y1 - y2);
                        });
                    }
                }]
            });
            
            // Start animation after chart creation
            this.startAnimation(flareData);
        }
        
        // Setup resize handler if not already done
        if (!window._goesChartResizeHandler) {
            window._goesChartResizeHandler = () => {
                if (this.chartInstance && this.chartInstance.options && this.chartInstance.options.scales && this.chartInstance.options.scales.y) {
                    this.chartInstance.options.scales.y.ticks = this.chartInstance.options.scales.y.ticks || {};
                    this.chartInstance.options.scales.y.ticks.display = this.getYAxisTicksDisplay();
                    this.chartInstance.update();
                }
            };
            window.addEventListener('resize', window._goesChartResizeHandler);
        }
    }
    
    startAnimation(flareData) {
        // Stop any existing animation
        this.stopAnimation();
        
        console.log('Starting GOES animation with data length:', flareData.length);
        
        if (!this.chartInstance || flareData.length === 0) {
            console.warn('Cannot start animation: chartInstance=', !!this.chartInstance, 'dataLength=', flareData.length);
            return;
        }
        
        this.currentFrame = 0;
        this.isAnimating = true;
        
        // Reset point radius to show animation
        const dataset = this.chartInstance.data.datasets[0];
        dataset.pointRadius = Array(24).fill(2);
        
        console.log('GOES animation started');
        
        this.animationTimer = setInterval(() => {
            // Highlight current frame
            dataset.pointRadius = Array(24).fill(2);
            if (this.currentFrame < 24) {
                dataset.pointRadius[this.currentFrame] = 8;
            }
            
            this.chartInstance.update('none');
            
            this.currentFrame++;
            
            // Loop animation
            if (this.currentFrame >= 24) {
                this.currentFrame = 0;
            }
        }, 500); // 500ms per frame
    }
    
    stopAnimation() {
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
            this.animationTimer = null;
        }
        this.isAnimating = false;
        
        // Reset point radius
        if (this.chartInstance) {
            const dataset = this.chartInstance.data.datasets[0];
            dataset.pointRadius = Array(24).fill(2);
            this.chartInstance.update('none');
        }
    }
    
    updateTheme() {
        if (!this.chartInstance) return;
        
        const isDark = document.body.classList.contains('dark-theme');
        const textColor = isDark ? '#f1f5f9' : '#333';
        const tickColor = isDark ? '#cbd5e1' : '#666';
        const gridColor = isDark ? 'rgba(148, 163, 184, 0.2)' : 'rgba(0, 0, 0, 0.1)';
        const tooltipBg = isDark ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.9)';
        const tooltipTitle = isDark ? '#f1f5f9' : '#333';
        const tooltipBody = isDark ? '#e2e8f0' : '#666';
        
        // Update chart colors
        this.chartInstance.options.scales.y.title.color = textColor;
        this.chartInstance.options.scales.y.ticks.color = tickColor;
        this.chartInstance.options.scales.y.ticks.callback = function(value, index, values) {
            // Convert to scientific notation format like 10^-4
            const exponent = Math.log10(value);
            if (Number.isInteger(exponent)) {
                return `10^${exponent}`;
            }
            return '';
        };
        this.chartInstance.options.scales.y.grid.color = gridColor;
        this.chartInstance.options.scales.x.title.color = textColor;
        this.chartInstance.options.scales.x.ticks.color = tickColor;
        this.chartInstance.options.scales.x.grid.color = gridColor;
        this.chartInstance.options.plugins.tooltip.backgroundColor = tooltipBg;
        this.chartInstance.options.plugins.tooltip.titleColor = tooltipTitle;
        this.chartInstance.options.plugins.tooltip.bodyColor = tooltipBody;
        
        this.chartInstance.update();
    }
    
    destroy() {
        // Stop animation
        this.stopAnimation();
        
        if (this.chartInstance) {
            this.chartInstance.destroy();
            this.chartInstance = null;
        }
        
        if (window._goesChartResizeHandler) {
            window.removeEventListener('resize', window._goesChartResizeHandler);
            window._goesChartResizeHandler = null;
        }
    }
}

// Export for use in other modules
window.GOESChartManager = GOESChartManager;
