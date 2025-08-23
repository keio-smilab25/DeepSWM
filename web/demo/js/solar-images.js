// Solar images loading and display

class SolarImagesManager {
    constructor() {
        this.wavelengths = [
            { name: 'AIA 94 Å', code: '0094', filename: 'aia_0094' },
            { name: 'AIA 131 Å', code: '0131', filename: 'aia_0131' },
            { name: 'AIA 171 Å', code: '0171', filename: 'aia_0171' },
            { name: 'AIA 193 Å', code: '0193', filename: 'aia_0193' },
            { name: 'AIA 211 Å', code: '0211', filename: 'aia_0211' },
            { name: 'AIA 304 Å', code: '0304', filename: 'aia_0304' },
            { name: 'AIA 335 Å', code: '0335', filename: 'aia_0335' },
            { name: 'AIA 1600 Å', code: '1600', filename: 'aia_1600' },
            { name: 'AIA 4500 Å', code: '4500', filename: 'aia_4500' },
            { name: 'HMI', code: 'hmi', filename: 'hmi' }
        ];
        
        this.basePath = this.getBasePath();
        this.animationFrames = [];
        this.currentFrame = 0;
        this.isPlaying = true;
        this.animationInterval = null;
        this.frameCount = 6; // 6 hours of data
        
        // Remove animation controls setup
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
    
    // Animation controls removed
    
    startAnimation() {
        if (this.animationInterval) return;
        
        this.animationInterval = setInterval(() => {
            this.nextFrame();
        }, 800); // 800ms per frame
    }
    
    pauseAnimation() {
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
            this.animationInterval = null;
        }
    }
    
    nextFrame() {
        if (this.animationFrames.length === 0) return;
        
        this.currentFrame = (this.currentFrame + 1) % this.animationFrames.length;
        this.displayFrame(this.currentFrame);
        this.updateFrameCounter();
    }
    
    displayFrame(frameIndex) {
        const grid = document.getElementById('solar-grid');
        if (!grid) return;
        
        const channels = grid.querySelectorAll('.channel');
        channels.forEach((channel, wavelengthIndex) => {
            const canvas = channel.querySelector('canvas');
            if (canvas && this.animationFrames[frameIndex] && this.animationFrames[frameIndex][wavelengthIndex]) {
                const frameCanvas = this.animationFrames[frameIndex][wavelengthIndex];
                const ctx = canvas.getContext('2d');
                
                // Set canvas size to match container
                const containerRect = canvas.getBoundingClientRect();
                const size = Math.min(containerRect.width, containerRect.height);
                
                canvas.width = size;
                canvas.height = size;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(frameCanvas, 0, 0, canvas.width, canvas.height);
            }
        });
    }
    
    // Frame counter removed

    async loadImages(date, hour) {
        const dateStr = `${String(date.getMonth() + 1).padStart(2, '0')}${String(date.getDate()).padStart(2, '0')}`;
        
        const grid = document.getElementById('solar-grid');
        if (!grid) return;
        
        grid.innerHTML = '';
        this.animationFrames = [];
        this.loadedTimeRange = null;
        
        // Timestamp display removed
        
        // Create grid structure first
        for (const wavelength of this.wavelengths) {
            const container = document.createElement('div');
            container.className = 'channel';
            
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 512;
            
            const label = document.createElement('div');
            label.className = 'channel-label';
            label.textContent = wavelength.name;
            
            const copyright = document.createElement('div');
            copyright.className = 'channel-copyright';
            copyright.textContent = 'SDO©NASA';
            
            container.appendChild(canvas);
            container.appendChild(copyright);
            container.appendChild(label);
            grid.appendChild(container);
        }
        
        // Load frames for animation (6 hours back from selected time)
        for (let frameIndex = 0; frameIndex < this.frameCount; frameIndex++) {
            const frameHour = hour - (this.frameCount - 1 - frameIndex);
            const frameDate = new Date(date);
            
            if (frameHour < 0) {
                frameDate.setDate(frameDate.getDate() - 1);
                frameDate.setHours(24 + frameHour);
            } else {
                frameDate.setHours(frameHour);
            }
            
            const frameDateStr = `${String(frameDate.getMonth() + 1).padStart(2, '0')}${String(frameDate.getDate()).padStart(2, '0')}`;
            const frameHourStr = String(frameDate.getHours()).padStart(2, '0');
            
            const frameCanvases = [];
            
            for (let wavelengthIndex = 0; wavelengthIndex < this.wavelengths.length; wavelengthIndex++) {
                const wavelength = this.wavelengths[wavelengthIndex];
                const imageUrl = `${this.basePath}/data/images/${frameDateStr}/${frameHourStr}_${wavelength.filename}.png`;
                
                try {
                    const canvas = await this.loadAndProcessImage(imageUrl, wavelength.code);
                    frameCanvases.push(canvas);
                } catch (error) {
                    console.log(`Image not found: ${imageUrl}, using placeholder...`);
                    const placeholderCanvas = await this.loadPlaceholderImage(wavelength);
                    frameCanvases.push(placeholderCanvas);
                    
                    // Asynchronously try to load fallback image and replace placeholder
                    this.loadFallbackAndReplace(frameIndex, wavelengthIndex, frameDate, wavelength);
                }
            }
            
            this.animationFrames.push(frameCanvases);
        }
        
        // Calculate and store time range
        const startHour = hour - (this.frameCount - 1);
        const startDate = new Date(date);
        if (startHour < 0) {
            startDate.setDate(startDate.getDate() - 1);
            startDate.setHours(24 + startHour);
        } else {
            startDate.setHours(startHour);
        }
        
        const startTime = `${String(startDate.getMonth() + 1).padStart(2, '0')}/${String(startDate.getDate()).padStart(2, '0')} ${String(startDate.getHours()).padStart(2, '0')}:00`;
        const endTime = `${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')} ${String(hour).padStart(2, '0')}:00`;
        
        this.loadedTimeRange = { startTime, endTime };
        
        // Display the latest frame initially
        this.currentFrame = this.animationFrames.length - 1;
        this.displayFrame(this.currentFrame);
        
        // Start animation automatically
        this.startAnimation();
    }
    
    async loadPlaceholderImage(wavelength) {
        const placeholderUrl = `${this.basePath}/data/images/placeholder/${wavelength.filename}.png`;
        
        try {
            console.log(`Loading placeholder: ${placeholderUrl}`);
            const canvas = await this.loadAndProcessImage(placeholderUrl, wavelength.code);
            return canvas;
        } catch (error) {
            console.log(`Failed to load placeholder, creating default: ${error.message}`);
            return this.createDefaultPlaceholder(wavelength.name);
        }
    }
    
    async loadFallbackAndReplace(frameIndex, wavelengthIndex, targetDate, wavelength) {
        const fallbackCanvas = await this.findFallbackImage(targetDate, wavelength);
        
        if (fallbackCanvas && this.animationFrames[frameIndex]) {
            // Replace the placeholder with the fallback image
            this.animationFrames[frameIndex][wavelengthIndex] = fallbackCanvas;
            
            // If this is the currently displayed frame, update the display
            if (frameIndex === this.currentFrame) {
                this.displayFrame(this.currentFrame);
            }
            
            console.log(`Replaced placeholder with fallback for ${wavelength.name}`);
        }
    }

    async findFallbackImage(targetDate, wavelength) {
        const maxHoursBack = 24; // 最大24時間前まで探索
        
        for (let hoursBack = 1; hoursBack <= maxHoursBack; hoursBack++) {
            const fallbackDate = new Date(targetDate);
            fallbackDate.setHours(fallbackDate.getHours() - hoursBack);
            
            const fallbackDateStr = `${String(fallbackDate.getMonth() + 1).padStart(2, '0')}${String(fallbackDate.getDate()).padStart(2, '0')}`;
            const fallbackHourStr = String(fallbackDate.getHours()).padStart(2, '0');
            const fallbackUrl = `${this.basePath}/data/images/${fallbackDateStr}/${fallbackHourStr}_${wavelength.filename}.png`;
            
            try {
                console.log(`Trying fallback: ${fallbackUrl}`);
                const canvas = await this.loadAndProcessImage(fallbackUrl, wavelength.code);
                console.log(`Found fallback image: ${fallbackUrl}`);
                return canvas;
            } catch (error) {
                // Continue searching
            }
        }
        
        console.log(`No fallback found for ${wavelength.name}`);
        return null;
    }
    
    createDefaultPlaceholder(wavelengthName) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 512;
        canvas.height = 512;
        
        // Create a simple placeholder
        ctx.fillStyle = '#2c2c2c';
        ctx.fillRect(0, 0, 512, 512);
        
        ctx.fillStyle = '#666';
        ctx.font = '24px "Kanit", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('No Data Available', 256, 230);
        ctx.fillText(wavelengthName, 256, 270);
        
        return canvas;
    }

    loadAndProcessImage(url, wavelengthCode) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            
            img.onload = () => {
                try {
                    if (wavelengthCode !== 'hmi' && wavelengthCode !== '4500') {
                        // Apply AIA colormap
                        const coloredCanvas = window.AIAColormaps.apply(img, wavelengthCode);
                        resolve(coloredCanvas);
                    } else {
                        // For HMI and 4500, just use the original image
                        const canvas = document.createElement('canvas');
                        const ctx = canvas.getContext('2d');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        resolve(canvas);
                    }
                } catch (error) {
                    reject(error);
                }
            };
            
            img.onerror = () => {
                reject(new Error(`Failed to load image: ${url}`));
            };
            
            img.src = url;
        });
    }
}

// Export for use in other modules
window.SolarImagesManager = SolarImagesManager;
