// Translation system for multi-language support

const translations = {
    en: {
        title: "Deep Space Weather Model",
        subtitle: "Operational Solar Flare AI Prediction Model",
        date_selection: "Date & Time Selection",
        solar_images: "Multi-wavelength Solar Images",
        prediction: "24-Hour Flare Prediction",
        select_date_msg: "Select a date to view solar images",
        loading_prediction: "Loading prediction...",
        performance: "Model Performance",
        since_date: "Since 2025-04-01",
        accuracy: "Binary Accuracy",
        classification: "O+C vs M+X",
        binary_accuracy: "Binary Accuracy",
        no_prediction: "No prediction data available for this date",
        time_label: "Time (UTC)",
        confidence: "Confidence",
        prediction_results: "Forecast Result",
        prediction_probabilities: "Probability of Flares", 
        prediction_performance: "Model Performance",
        model_performance_week: "Model Performance (Past Week)",
        model_performance_month: "Model Performance (Past Month)",
        model_performance_all: "Model Performance (All Period)",
        since_may_2025: "Since May 2025",
        disclaimer_title: "Disclaimer",
        disclaimer_text: "This system is provided for research and educational purposes only. The solar flare predictions are experimental and should not be used as the sole basis for operational decisions. The authors and affiliated institutions assume no responsibility for any damages or losses that may result from the use of this information.",
        japanese: "日本語",
        english: "English",
        about_solar_flares: "About Solar Flares",
        about_deepswm: "About Deep Space Weather Model (DeepSWM)",
        current_solar_surface: "Current Solar Surface",
        current_solar_flare_status: "Current Solar Flare Status",
        solar_flare_forecast_24h: "Solar Flare Forecast Over the Next 24 Hours",
        archive: "Archive",
        xray_flux_transition: "X-ray Flux Transition (GOES)",
        solar_flare: "Solar Flare",
        data_time_utc: "Data Time (UTC)",
        author_names: "Shunya Nagashima and Komei Sugiura<br/><span class=\"university-name\">Keio University</span>",
        loading: "Loading...",
        m_accuracy: "M≥ Accuracy",
        months: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        weekdays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        flare_status_major: "Major Flares",
        flare_status_active: "Active", 
        flare_status_eruptive: "Eruptive",
        flare_status_quiet: "Quiet",
        flare_level_x: "Lv.4 (X-Class)",
        flare_level_m: "Lv.3 (M-Class)",
        flare_level_c: "Lv.2 (C-Class)",
        flare_level_o: "Lv.1 (O-Class)",
        predictions: "predictions"
    },
    ja: {
        title: "深宇宙天気モデル",
        subtitle: "太陽フレアAI予報システム",
        date_selection: "日時選択",
        solar_images: "多波長太陽画像",
        prediction: "24時間フレア予測",
        select_date_msg: "日付を選択して太陽画像を表示",
        loading_prediction: "読み込み中...",
        performance: "予報的中率",
        since_date: "2025年4月1日以降",
        accuracy: "二値精度",
        classification: "M≥での精度",
        binary_accuracy: "二値分類精度",
        no_prediction: "この日付の予測データがありません",
        time_label: "時刻 (UTC)",
        confidence: "信頼度",
        prediction_results: "予報結果",
        prediction_probabilities: "予報確率",
        prediction_performance: "予報的中率",
        model_performance_week: "予報的中率 (直近1週間)",
        model_performance_month: "予報的中率<br/>(直近1ヶ月)",
        model_performance_all: "予報的中率<br/>(全期間)",
        since_may_2025: "2025年5月〜",
        disclaimer_title: "免責事項",
        disclaimer_text: "本システムは研究・教育目的で提供されています。太陽フレアの予測は実験的なものであり、実用的な判断の根拠として使用しないでください。本情報の利用により生じた損害について、開発者および関連機関は一切の責任を負いかねます。",
        japanese: "日本語",
        english: "English",
        about_solar_flares: "太陽フレアについて",
        about_deepswm: "Deep Space Weather Model (DeepSWM)について",
        current_solar_surface: "現在の太陽表面",
        current_solar_flare_status: "現在の太陽フレア状況",
        solar_flare_forecast_24h: "今後24時間の太陽フレア予報",
        archive: "アーカイブ",
        xray_flux_transition: "X線フラックス推移 (GOES)",
        solar_flare: "太陽フレア",
        data_time_utc: "データ時刻 (UTC)",
        author_names: "長嶋隼矢, 杉浦孔明<br/><span class=\"university-name\">慶應義塾大学</span>",
        loading: "読み込み中...",
        m_accuracy: "M≥での精度",
        months: ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"],
        weekdays: ["日", "月", "火", "水", "木", "金", "土"],
        flare_status_major: "非常に活発",
        flare_status_active: "活発", 
        flare_status_eruptive: "やや活発",
        flare_status_quiet: "静穏",
        flare_level_x: "Lv.4 (Xクラス)",
        flare_level_m: "Lv.3 (Mクラス)",
        flare_level_c: "Lv.2 (Cクラス)",
        flare_level_o: "Lv.1 (Oクラス)",
        predictions: "件の予測"
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
        } else if (!lang) {
            // Default to English if no language parameter is specified
            this.currentLang = 'en';
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
                const translation = translations[this.currentLang][key];
                if (translation.includes('<br/>')) {
                    el.innerHTML = translation;
                } else {
                    el.textContent = translation;
                }
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
