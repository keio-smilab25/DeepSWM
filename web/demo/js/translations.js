// Translation system for multi-language support

const translations = {
    en: {
        title: "Deep Space Weather Model",
        subtitle: "Interactive Solar Flare Prediction Demo",
        date_selection: "Date & Time Selection",
        solar_images: "Multi-wavelength Solar Images",
        prediction: "24-Hour Flare Prediction",
        select_date_msg: "Select a date to view solar images",
        loading_prediction: "Loading prediction...",
        performance: "Model Performance",
        since_date: "Since 2025-04-01",
        accuracy: "Binary Accuracy",
        classification: "O+C vs M+X",
        flare_desc_x: "Major solar flare - significant impact",
        flare_desc_m: "Moderate solar flare - possible effects",
        flare_desc_c: "Minor solar flare - minimal space weather impact",
        flare_desc_o: "No significant solar flare activity",
        binary_accuracy: "Binary Accuracy",
        no_prediction: "No prediction data available for this date",
        time_label: "Time (UTC)",
        confidence: "Confidence",
        prediction_results: "Prediction Results",
        prediction_probabilities: "Prediction Probabilities", 
        prediction_performance: "Prediction Performance",
        since_april_2025: "Since April 2025",
        japanese: "日本語",
        english: "English"
    },
    ja: {
        title: "深宇宙天気モデル",
        subtitle: "インタラクティブ太陽フレア予測デモ",
        date_selection: "日時選択",
        solar_images: "多波長太陽画像",
        prediction: "24時間フレア予測",
        select_date_msg: "日付を選択して太陽画像を表示",
        loading_prediction: "予測データを読み込み中...",
        performance: "モデル性能",
        since_date: "2025年4月1日以降",
        accuracy: "二値精度",
        classification: "O+C vs M+X",
        flare_desc_x: "大規模太陽フレア - 重大な影響",
        flare_desc_m: "中規模太陽フレア - 影響の可能性",
        flare_desc_c: "小規模太陽フレア - 最小限の宇宙天気への影響",
        flare_desc_o: "重大な太陽フレア活動なし",
        binary_accuracy: "二値分類精度",
        no_prediction: "この日付の予測データがありません",
        time_label: "時刻 (UTC)",
        confidence: "信頼度",
        prediction_results: "予測結果",
        prediction_probabilities: "予測確率",
        prediction_performance: "予測性能",
        since_april_2025: "2025年4月から",
        japanese: "日本語",
        english: "English"
    }
};

class TranslationManager {
    constructor() {
        this.currentLang = 'en';
        this.init();
    }
    
    init() {
        // Check URL parameter for language
        const urlParams = new URLSearchParams(window.location.search);
        const lang = urlParams.get('lang');
        if (lang && translations[lang]) {
            this.currentLang = lang;
        }
        
        // Apply translations
        this.updateTranslations();
    }
    
    getCurrentLang() {
        return this.currentLang;
    }
    
    setLanguage(lang) {
        if (translations[lang]) {
            this.currentLang = lang;
            this.updateTranslations();
        }
    }
    
    updateTranslations() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (translations[this.currentLang] && translations[this.currentLang][key]) {
                el.textContent = translations[this.currentLang][key];
            }
        });
    }
    
    t(key) {
        return translations[this.currentLang] && translations[this.currentLang][key] 
            ? translations[this.currentLang][key] 
            : key;
    }
}

// Export for use in other modules
window.TranslationManager = TranslationManager;
