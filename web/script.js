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

// ========== 的中率計算 ==========
function computeAccuracy(predData, xrsMap, rangeHours) {
  let total = 0, correct = 0;
  for (const key in predData) {
    if (!predData.hasOwnProperty(key)) continue;
    // key="YYYYMMDDHH"
    const year  = +key.slice(0,4);
    const month = +key.slice(4,6) - 1;
    const day   = +key.slice(6,8);
    const hour  = +key.slice(8,10);
    const baseUTC = Date.UTC(year, month, day, hour);

    // t から t+rangeHours-1 の最大フラックスを取得
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
    if (maxFlux === null) continue; // データ不足

    const trueCls = computeClassFromFlux(maxFlux);
    const probs   = predData[key];
    if (!Array.isArray(probs) || probs.length < 4) continue;
    const predCls = probs.indexOf(Math.max(...probs));
    if (Math.floor(predCls / 2) === Math.floor(trueCls / 2)) correct++;
    total++;
  }
  return total > 0 ? (correct / total) : null;
}

// ========== Recall≥M計算 ==========
function computeRecallM(predData, xrsMap, rangeHours) {
  let totalM = 0, detectedM = 0;
  for (const key in predData) {
    if (!predData.hasOwnProperty(key)) continue;
    // key="YYYYMMDDHH"
    const year  = +key.slice(0,4);
    const month = +key.slice(4,6) - 1;
    const day   = +key.slice(6,8);
    const hour  = +key.slice(8,10);
    const baseUTC = Date.UTC(year, month, day, hour);

    // t から t+rangeHours-1 の最大フラックスを取得
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
    if (maxFlux === null) continue; // データ不足

    const trueCls = computeClassFromFlux(maxFlux);
    if (trueCls < 2) continue; // Mクラス未満は除外
    totalM++;

    const probs   = predData[key];
    if (!Array.isArray(probs) || probs.length < 4) continue;
    const predCls = probs.indexOf(Math.max(...probs));
    if (predCls >= 2) detectedM++;
  }
  return totalM > 0 ? (detectedM / totalM) : null;
}

// ========== 的中率表示 ==========
function displayAccuracy(acc, recallM = null) {
  const el = document.getElementById('accuracyDisplay');
  // どのrangeが選択されているか取得
  const rangeElem = document.querySelector('input[name="prediction-range"]:checked');
  const range = rangeElem ? +rangeElem.value : 24;
  let header = '';
  if (range === 24) header = '24時間予測性能';
  else if (range === 48) header = '48時間予測性能';
  else if (range === 72) header = '72時間予測性能';
  else header = `${range}時間予測的中率`;

  el.innerHTML = `<h3>${header}</h3>`;
  // if (acc == null) {
  //   el.innerHTML += '的中率を計算できるデータがありません';
  // } else {
  //   el.innerHTML += `Accuracy≥M: ${(acc * 100).toFixed(2)}%`;
  // }
  if (recallM == null) {
    el.innerHTML += `<br>Recall≥M: データ不足`;
  } else {
    el.innerHTML += `<br>Recall≥M: ${(recallM * 100).toFixed(2)}%`;
  }
  
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

  // 時刻セレクト初期化
  const hourSelect = document.getElementById("utc-hour");
  hourSelect.innerHTML = "";
  for (let h = 0; h < 24; h += 2) {
    const opt = document.createElement("option");
    opt.value = h;
    opt.textContent = String(h).padStart(2,"0") + ":00";
    hourSelect.appendChild(opt);
  }
  hourSelect.value = Math.floor(utcNow.getUTCHours()/2)*2;

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

// ========== メイン処理 ==========
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

  // タイムスタンプ生成
  timestamps = [];
  for (let h = 22; h >= 0; h -= 2) {
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

  // 予測データ取得 & カード & 的中率
  const keyTime = `${Y}${String(M).padStart(2,'0')}${String(D).padStart(2,'0')}${String(H).padStart(2,'0')}`;
  const rangeElem = document.querySelector('input[name="prediction-range"]:checked');
  const predictionRange = rangeElem ? +rangeElem.value : 24;

  fetch(`data/pred_${predictionRange}.json`)
    .then(res => res.json())
    .then(predData => {
      const container = document.getElementById('prediction-cards');
      container.innerHTML = '';
      document.getElementById('prediction-header').textContent
        = `${Y}-${String(M).padStart(2,'0')}-${String(D).padStart(2,'0')} ${String(H).padStart(2,'0')}:00 UTC の ${predictionRange}時間先 推論結果`;

      const labels = ['O class','C class','M class','X class'];
      if (predData[keyTime]) {
        const probs = predData[keyTime];
        const maxP = Math.max(...probs);
        probs.forEach((p,i) => {
          const card = document.createElement('div');
          card.className = 'prediction-card' + (p===maxP ? ' highlight' : '');
          card.innerHTML = `
            <h3><span class="class-label">${labels[i][0]}</span>
                <span class="class-small">${labels[i].slice(1)}</span></h3>
            <p>Probability: ${(p*100).toFixed(2)}%</p>
          `;
          container.appendChild(card);
        });
      } else {
        console.warn(`❌ Key ${keyTime} の予測データ無し`);
        labels.forEach(lab => {
          const card = document.createElement('div');
          card.className = 'prediction-card';
          card.innerHTML = `
            <h3><span class="class-label">${lab[0]}</span>
                <span class="class-small">${lab.slice(1)}</span></h3>
            <p>Probability: --</p>
          `;
          container.appendChild(card);
        });
        const err = document.createElement('p');
        err.className = 'error-message';
        err.textContent = "❌ 推論データが存在しません";
        container.appendChild(err);
      }

      // 的中率計算＆表示
      const acc = computeAccuracy(predData, xrsFullDataMap, predictionRange);
      const recallM = computeRecallM(predData, xrsFullDataMap, predictionRange);
      displayAccuracy(acc, recallM);
    })
    .catch(err => {
      console.error("Prediction fetch error:", err);
    });

  // フレアチャート更新
  loadXRSData(baseTime).then(flareData => {
    const labels = Array.from({ length: 96 }, (_, i) => `${i > 24 ? '+' : ''}${i-24}h`);
    const ctx = document.getElementById('flareChart').getContext('2d');
    const pointColors = flareData.map(v => {
      if (v == null) return 'gray';
      if (v < 1e-6)     return 'blue';
      if (v < 1e-5)     return 'green';
      if (v < 1e-4)     return 'orange';
      return 'red';
    });

    if (window.flareChartInstance) {
      // 最新タイムスタンプをフォーマット
      const lastT = timestamps[timestamps.length - 1];
      const formattedTime =
        `${lastT.getUTCFullYear()}-${String(lastT.getUTCMonth()+1).padStart(2,'0')}` +
        `-${String(lastT.getUTCDate()).padStart(2,'0')} ` +
        `${String(lastT.getUTCHours()).padStart(2,'0')}:00 UTC`;
    
      // label.content に定義済みの formattedTime をセット
      window.flareChartInstance.data.labels = labels;
      window.flareChartInstance.data.datasets[0].data = flareData;
      window.flareChartInstance.data.datasets[0].pointBackgroundColor = pointColors;
      window.flareChartInstance.options.plugins.annotation.annotations
        .zeroHourLine.label.content = formattedTime;
    
      window.flareChartInstance.update();
    } else {
      const lastT = timestamps[timestamps.length - 1];
      const formattedTime =
        `${lastT.getUTCFullYear()}-${String(lastT.getUTCMonth()+1).padStart(2,'0')}` +
        `-${String(lastT.getUTCDate()).padStart(2,'0')} ` +
        `${String(lastT.getUTCHours()).padStart(2,'0')}:00 UTC`;
      window.flareChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'X-ray Flux (0.1–0.8 nm)',
            data: flareData,
            borderColor: 'black',
            pointBackgroundColor: pointColors,
            fill: false
          }]
        },
        options: {
          scales: {
            y: {
              type: 'logarithmic',
              min: 1e-9,
              max: 1e-3,
              title: { display: true, text: 'Flux (W/m²)' }
            }
          },
          plugins: {
            tooltip: {
              callbacks: {
                label: ctx => {
                  const v = ctx.raw;
                  if (v == null) return '欠損';
                  const cls = v >= 1e-4 ? 'X'
                            : v >= 1e-5 ? 'M'
                            : v >= 1e-6 ? 'C'
                            : 'O';
                  return `Flux: ${v} W/m² (Class ${cls})`;
                }
              }
            },
            annotation: {
              annotations: {
                flareBands: {
                  type: 'box',
                  yMin: 1e-4, yMax: 1e-3,
                  backgroundColor: 'rgba(255,0,0,0.05)',
                  label: { enabled: true, content: 'X', position: 'start', xAdjust: 50, backgroundColor: 'transparent', color: 'red', font: { weight: 'bold', size: 14 } }
                },
                flareBandM: {
                  type: 'box',
                  yMin: 1e-5, yMax: 1e-4,
                  backgroundColor: 'rgba(255,165,0,0.05)',
                  label: { enabled: true, content: 'M', position: 'start', xAdjust: 50, backgroundColor: 'transparent', color: 'orange', font: { weight: 'bold', size: 14 } }
                },
                flareBandC: {
                  type: 'box',
                  yMin: 1e-6, yMax: 1e-5,
                  backgroundColor: 'rgba(0,255,0,0.05)',
                  label: { enabled: true, content: 'C', position: 'start', xAdjust: 50, backgroundColor: 'transparent', color: 'green', font: { weight: 'bold', size: 14 } }
                },
                flareBandO: {
                  type: 'box',
                  yMin: 1e-9, yMax: 1e-6,
                  backgroundColor: 'rgba(0,0,255,0.05)',
                  label: { enabled: true, content: 'O', position: 'start', xAdjust: 50, backgroundColor: 'transparent', color: 'blue', font: { weight: 'bold', size: 14 } }
                },
                zeroHourLine: {
                  type: 'line',
                  scaleID: 'x', value: 24,
                  borderColor: 'black', borderWidth: 3,
                  label: { enabled: true, content: formattedTime, position: 'end', backgroundColor: 'black', color: 'white', font: { weight: 'bold', size: 12 } }
                }
              }
            }
          }
        },
        plugins: [{
          id: 'backgroundZones',
          beforeDraw: chart => {
            const { ctx, chartArea, scales } = chart;
            const zones = [
              { from: 1e-4, to: 1e-3, color: 'rgba(255,0,0,0.15)' },
              { from: 1e-5, to: 1e-4, color: 'rgba(255,165,0,0.15)' },
              { from: 1e-6, to: 1e-5, color: 'rgba(0,255,0,0.15)' },
              { from: 1e-9, to: 1e-6, color: 'rgba(0,0,255,0.15)' }
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
    }
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
    label.textContent = type === 'HMI' ? 'HMI' : `AIA ${parseInt(type,10)}Å`;
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
    timestampLabel.textContent = `現在表示中の時刻: ${timeStr}`;

    if (window.flareChartInstance) {
      const ds = window.flareChartInstance.data.datasets[0];
      ds.pointRadius = Array(96).fill(2);
      const offset = Math.floor((t - timestamps[timestamps.length-1])/(3600*1000));
      if (offset >= -24 && offset < 72) {
        ds.pointRadius[offset+24] = 6;
      }
      window.flareChartInstance.update('none');
    }

    frameIndex++;
  }, 400);
}