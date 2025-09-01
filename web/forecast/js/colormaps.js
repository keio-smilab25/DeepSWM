// AIA colormap definitions and processing

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

function applyAIAColormap(imageData, wavelength) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Create a new canvas with the same dimensions
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    
    // Draw the original image
    if (imageData instanceof HTMLImageElement) {
        ctx.drawImage(imageData, 0, 0);
    } else if (imageData instanceof ImageData) {
        ctx.putImageData(imageData, 0, 0);
    }
    
    // Get image data
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    
    // Apply colormap
    for (let i = 0; i < data.length; i += 4) {
        const gray = data[i]; // Use red channel as grayscale
        const norm = gray / 255;
        const [r, g, b] = getAIAColorForWavelength(norm, wavelength);
        data[i] = r;
        data[i+1] = g;
        data[i+2] = b;
        // Alpha channel remains unchanged
    }
    
    // Put the modified image data back
    ctx.putImageData(imgData, 0, 0);
    
    return canvas;
}

function createFallbackSolarImage(wavelength, size = 512) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = size;
    canvas.height = size;
    
    // Create synthetic solar disk
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, size, size);
    
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size * 0.4;
    
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
            if (dist < radius) {
                const intensity = Math.max(0, 1 - dist / radius);
                const noise = Math.random() * 0.3;
                const value = Math.min(1, intensity * 0.8 + noise * 0.2);
                
                let color;
                if (wavelength === 'hmi') {
                    // Magnetogram style
                    if (value < 0.5) {
                        const t = value * 2;
                        color = [Math.floor(255 * (0.3 + 0.7 * t)), Math.floor(255 * t), Math.floor(255 * t)];
                    } else {
                        const t = (value - 0.5) * 2;
                        color = [Math.floor(255 * (1 - t)), Math.floor(255 * (1 - t)), Math.floor(255 * (0.3 + 0.7 * t))];
                    }
                } else if (colormaps[wavelength]) {
                    color = getAIAColorForWavelength(value, wavelength);
                } else {
                    const gray = Math.floor(255 * value);
                    color = [gray, gray, gray];
                }
                
                ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
    }
    
    return canvas;
}

// Export for use in other modules
window.AIAColormaps = {
    apply: applyAIAColormap,
    createFallback: createFallbackSolarImage
};
