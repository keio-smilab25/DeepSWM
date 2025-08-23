// Language toggle functionality for the new UI

class LanguageToggleManager {
    constructor(translationManager) {
        this.translationManager = translationManager;
        this.init();
    }
    
    init() {
        const languageToggle = document.querySelector('.language-toggle');
        if (!languageToggle) return;
        
        // Set initial state
        const currentLang = this.translationManager.getCurrentLang();
        this.updateLanguageToggleState(currentLang);
        
        // Add click listeners to language buttons
        const langButtons = languageToggle.querySelectorAll('.lang-btn');
        langButtons.forEach(button => {
            button.addEventListener('click', () => {
                const lang = button.getAttribute('data-lang');
                if (lang && lang !== this.translationManager.getCurrentLang()) {
                    this.setLanguage(lang);
                }
            });
        });
    }
    
    setLanguage(lang) {
        this.translationManager.setLanguage(lang);
        this.updateLanguageToggleState(lang);
        this.updateURLLanguage(lang);
        
        // Trigger custom event for other components to update
        window.dispatchEvent(new CustomEvent('languageChanged', { detail: { lang } }));
    }
    
    updateLanguageToggleState(lang) {
        const languageToggle = document.querySelector('.language-toggle');
        if (!languageToggle) return;
        
        // Update active state
        const langButtons = languageToggle.querySelectorAll('.lang-btn');
        langButtons.forEach(button => {
            button.classList.toggle('active', button.getAttribute('data-lang') === lang);
        });
    }
    
    updateURLLanguage(lang) {
        const url = new URL(window.location);
        if (lang === 'ja') {
            url.searchParams.set('lang', 'ja');
        } else {
            url.searchParams.delete('lang');
        }
        window.history.replaceState({}, '', url);
    }
}

// Export for use in other modules
window.LanguageToggleManager = LanguageToggleManager;
