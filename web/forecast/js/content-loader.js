// Content loader for language-specific HTML content

class ContentLoader {
    constructor(translationManager) {
        this.translationManager = translationManager;
        this.contentCache = new Map();
        this.init();
    }
    
    init() {
        // Listen for language change events
        window.addEventListener('languageChanged', (event) => {
            this.loadAllContent(event.detail.lang);
        });
        
        // Load initial content
        this.loadAllContent(this.translationManager.getCurrentLang());
    }
    
    async loadContent(contentId, language) {
        const cacheKey = `${contentId}-${language}`;
        
        // Check cache first
        if (this.contentCache.has(cacheKey)) {
            return this.contentCache.get(cacheKey);
        }
        
        try {
            const response = await fetch(`web/forecast/content/${contentId}-${language}.html`);
            if (!response.ok) {
                throw new Error(`Failed to load content: ${response.status}`);
            }
            
            const content = await response.text();
            this.contentCache.set(cacheKey, content);
            return content;
        } catch (error) {
            console.warn(`Failed to load content ${contentId}-${language}:`, error);
            
            // Fallback to English if available
            if (language !== 'en') {
                return this.loadContent(contentId, 'en');
            }
            
            return null;
        }
    }
    
    async loadAllContent(language) {
        const contentMappings = [
            { contentId: 'solar-flares', targetId: 'solar-flares-content-text' },
            { contentId: 'deepswm', targetId: 'deepswm-content-text' }
        ];
        
        for (const mapping of contentMappings) {
            try {
                const content = await this.loadContent(mapping.contentId, language);
                if (content) {
                    const targetElement = document.getElementById(mapping.targetId);
                    if (targetElement) {
                        targetElement.innerHTML = content;
                    }
                }
            } catch (error) {
                console.error(`Error loading content ${mapping.contentId}:`, error);
            }
        }
    }
    
    async updateContent(contentId, targetId, language = null) {
        const lang = language || this.translationManager.getCurrentLang();
        const content = await this.loadContent(contentId, lang);
        
        if (content) {
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.innerHTML = content;
            }
        }
    }
}

// Export for use in other modules
window.ContentLoader = ContentLoader;
