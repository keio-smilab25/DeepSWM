// ========== 定数定義 ==========
const wavelengths = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '4500'];

// ========== 波長ごとのカラーマップ定義 ==========
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

// ========== カラーマップ補間関数 ==========
function getAIAColorForWavelength(normValue, wavelength) {
  const stops = colormaps[wavelength];
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

// ========== AIA画像にカラーマップを適用 ==========
function applyAIAColormap(image, wavelength) {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    const gray = data[i];
    const norm = gray / 255;
    const [r, g, b] = getAIAColorForWavelength(norm, wavelength);
    data[i] = r;
    data[i+1] = g;
    data[i+2] = b;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}

// ========== フレアクラス判定 ==========
function computeClassFromFlux(flux) {
  if (flux < 1e-6)    return 0; // O
  if (flux < 1e-5)    return 1; // C
  if (flux < 1e-4)    return 2; // M
  return 3;                    // X
}

// ========== グローバル変数 ==========
let xrsFullDataMap = {};
let fp, preloadedImages = {}, timestamps = [], imageElements = {}, frameIndex = 0, animationTimer = null;

// ========== 初期化処理 ==========
window.addEventListener('DOMContentLoaded', () => {
  // XRSデータをマップで読み込み
  fetch('data/xrs.json')
    .then(res => res.json())
    .then(data => { xrsFullDataMap = data; })
    .catch(err => console.error("XRS読み込みエラー:", err));

  const now = new Date();
  const utcNow = new Date(Date.UTC(
    now.getUTCFullYear(), now.getUTCMonth(),
    now.getUTCDate(), now.getUTCHours(), 0, 0
  ));

  // 日付ピッカー
  fp = flatpickr("#date-picker", {
    inline: true,
    enableTime: false,
    dateFormat: "Y-m-d",
    defaultDate: utcNow.toISOString().slice(0,10),
    maxDate:     utcNow.toISOString().slice(0,10),
    minDate:     "2025-04-01",
  });

  // 時刻セレクト初期化（1時間間隔）
  const hourSelect = document.getElementById("utc-hour");
  hourSelect.innerHTML = "";
  for (let h = 0; h < 24; h++) {
    const opt = document.createElement("option");
    opt.value = h;
    opt.textContent = String(h).padStart(2,"0") + ":00";
    hourSelect.appendChild(opt);
  }
  hourSelect.value = utcNow.getUTCHours();

  document.getElementById('load-button')
    .addEventListener('click', loadImagesFromSelectedTime);

  loadImagesFromSelectedTime();
});

// ========== チャート用XRS取得 ==========
function loadXRSData(baseTime) {
  return fetch('data/xrs.json')
    .then(res => res.json())
    .then(data => {
      const arr = [];
      for (let i = -24; i < 72; i++) {
        const t = new Date(baseTime.getTime() + i*3600*1000);
        const key = `${t.getUTCFullYear()}${String(t.getUTCMonth()+1).padStart(2,'0')}`
                  + `${String(t.getUTCDate()).padStart(2,'0')}${String(t.getUTCHours()).padStart(2,'0')}`;
        arr.push(data[key] != null ? data[key] : null);
      }
      return arr;
    })
    .catch(err => {
      console.error("チャート用XRS取得エラー:", err);
      return Array(96).fill(null);
    });
}

// ========== メイン処理（1時間間隔対応） ==========
function loadImagesFromSelectedTime() {
  if (animationTimer) {
    clearInterval(animationTimer);
    animationTimer = null;
  }

  const dateStr = fp.input.value;
  if (!dateStr) { console.error("日付が選択されていません"); return; }
  const [Y, M, D] = dateStr.split("-").map(s=>+s);
  const H = +document.getElementById("utc-hour").value;
  const baseTime = new Date(Date.UTC(Y, M-1, D, H, 0, 0));

  // タイムスタンプ生成（1時間間隔で24時間分）
  timestamps = [];
  for (let h = 23; h >= 0; h--) {
    timestamps.push(new Date(baseTime.getTime() - h*3600*1000));
  }

  // URL生成
  const aiaUrls = {};
  wavelengths.forEach(wl => {
    aiaUrls[wl] = timestamps.map(t => {
      const m = String(t.getUTCMonth()+1).padStart(2,'0');
      const d = String(t.getUTCDate()).padStart(2,'0');
      const h = String(t.getUTCHours()).padStart(2,'0');
      return `data/images/${m}${d}/${h}_aia_${wl}.png`;
    });
  });
  const hmiUrls = timestamps.map(t => {
    const m = String(t.getUTCMonth()+1).padStart(2,'0');
    const d = String(t.getUTCDate()).padStart(2,'0');
    const h = String(t.getUTCHours()).padStart(2,'0');
    return `data/images/${m}${d}/${h}_hmi.png`;
  });

  // 画像プリロード
  preloadedImages = {};
  const transparentURL = createTransparentImageURL();
  wavelengths.forEach(wl => {
    aiaUrls[wl].forEach((url,i) => {
      const key = `${wl}-${i}`;
      const img = new Image();
      img.onload = () => {
        const png = applyAIAColormap(img, wl);
        const cimg = new Image();
        cimg.onload = () => { preloadedImages[key] = cimg; };
        cimg.src = png;
      };
      img.onerror = () => {
        const fb = new Image();
        fb.src = transparentURL;
        preloadedImages[key] = fb;
        console.warn(`❌ AIA画像読み込み失敗: ${url}`);
      };
      img.src = url;
    });
  });
  hmiUrls.forEach((url,i) => {
    const key = `HMI-${i}`;
    const img = new Image();
    img.onload = () => { preloadedImages[key] = img; };
    img.onerror = () => {
      const fb = new Image();
      fb.src = transparentURL;
      preloadedImages[key] = fb;
      console.warn(`❌ HMI画像読み込み失敗: ${url}`);
    };
    img.src = url;
  });

  renderImages();

  // 24時間先予測データ取得 & 表示
  const keyTime = `${Y}${String(M).padStart(2,'0')}${String(D).padStart(2,'0')}${String(H).padStart(2,'0')}`;
  
  fetch(`data/pred_24.json`)
    .then(res => res.json())
    .then(predData => {
      const container = document.getElementById('prediction-result');
      container.innerHTML = '';

      const labels = ['O class','C class','M class','X class'];
      if (predData[keyTime]) {
        const probs = predData[keyTime];
        const maxIdx = probs.indexOf(Math.max(...probs));
        const maxProb = probs[maxIdx];
        
        container.innerHTML = `
          <div class="prediction-display">
            <h2 class="prediction-title">24-Hour Solar Flare Prediction</h2>
            <div class="prediction-time">${Y}-${String(M).padStart(2,'0')}-${String(D).padStart(2,'0')} ${String(H).padStart(2,'0')}:00 UTC</div>
            <div class="prediction-class ${labels[maxIdx][0].toLowerCase()}-class">
              <span class="class-label">${labels[maxIdx][0]}</span>
              <span class="class-description">${labels[maxIdx].slice(1)}</span>
              <div class="probability">${(maxProb*100).toFixed(1)}%</div>
            </div>
            <div class="prediction-details">
              ${probs.map((p, i) => `
                <div class="detail-item ${i === maxIdx ? 'active' : ''}">
                  ${labels[i][0]}: ${(p*100).toFixed(1)}%
                </div>
              `).join('')}
            </div>
          </div>
        `;
      } else {
        container.innerHTML = `
          <div class="prediction-display">
            <h2 class="prediction-title">24-Hour Solar Flare Prediction</h2>
            <div class="prediction-time">${Y}-${String(M).padStart(2,'0')}-${String(D).padStart(2,'0')} ${String(H).padStart(2,'0')}:00 UTC</div>
            <div class="no-data">No prediction data available</div>
          </div>
        `;
      }
    })
    .catch(err => {
      console.error("Prediction fetch error:", err);
      document.getElementById('prediction-result').innerHTML = `
        <div class="prediction-display">
          <h2 class="prediction-title">24-Hour Solar Flare Prediction</h2>
          <div class="no-data">Error loading prediction data</div>
        </div>
      `;
    });
}

// ========== No-data用透過画像 ==========
function createTransparentImageURL(width = 200, height = 200) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.font = "20px sans-serif";
  ctx.fillStyle = "rgba(255, 0, 0, 0.7)";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("No data", width/2, height/2);
  return canvas.toDataURL('image/png');
}

// ========== 画像レンダリング & アニメーション ==========
function renderImages() {
  const grid = document.getElementById('aia-grid');
  grid.innerHTML = '';
  imageElements = {};

  [...wavelengths, 'HMI'].forEach(type => {
    const container = document.createElement('div');
    container.className = 'channel';
    const label = document.createElement('div');
    label.textContent = type === 'HMI' ? 'HMI' : `AIA ${parseInt(type,10)}Å`;
    const img = document.createElement('img');
    img.id = `img-${type}`;
    container.appendChild(label);
    container.appendChild(img);
    grid.appendChild(container);
    imageElements[type] = img;
  });

  frameIndex = 0;
  const timestampLabel = document.getElementById('timestamp');
  animationTimer = setInterval(() => {
    wavelengths.forEach(wl => {
      const key = `${wl}-${frameIndex % timestamps.length}`;
      if (preloadedImages[key]) imageElements[wl].src = preloadedImages[key].src;
    });
    const hmiKey = `HMI-${frameIndex % timestamps.length}`;
    if (preloadedImages[hmiKey]) imageElements['HMI'].src = preloadedImages[hmiKey].src;

    const t = timestamps[frameIndex % timestamps.length];
    const timeStr = `${t.getUTCFullYear()}-${String(t.getUTCMonth()+1).padStart(2,'0')}`
                  + `-${String(t.getUTCDate()).padStart(2,'0')} ${String(t.getUTCHours()).padStart(2,'0')}:00 UTC`;
    timestampLabel.textContent = `Current Time: ${timeStr}`;

    frameIndex++;
  }, 500);
} 