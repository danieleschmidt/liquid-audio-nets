//! Internationalization (i18n) support for Liquid Audio Nets
//!
//! Provides multi-language support for error messages, documentation,
//! and user-facing content for global deployment.

#[cfg(feature = "std")]
use std::{string::String, collections::BTreeMap, vec::Vec, sync::{OnceLock, Mutex}};

#[cfg(not(feature = "std"))]
use alloc::{string::String, collections::BTreeMap, vec::Vec};

/// Supported languages for internationalization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Language {
    /// English (default)
    English,
    /// Spanish
    Spanish,
    /// French
    French,
    /// German
    German,
    /// Japanese
    Japanese,
    /// Chinese (Simplified)
    ChineseSimplified,
    /// Portuguese
    Portuguese,
    /// Russian
    Russian,
    /// Korean
    Korean,
    /// Arabic
    Arabic,
}

impl Language {
    /// Get language code (ISO 639-1)
    pub fn code(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Japanese => "ja",
            Language::ChineseSimplified => "zh",
            Language::Portuguese => "pt",
            Language::Russian => "ru",
            Language::Korean => "ko",
            Language::Arabic => "ar",
        }
    }
    
    /// Get language name in native script
    pub fn native_name(&self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Spanish => "Español",
            Language::French => "Français",
            Language::German => "Deutsch",
            Language::Japanese => "日本語",
            Language::ChineseSimplified => "中文 (简体)",
            Language::Portuguese => "Português",
            Language::Russian => "Русский",
            Language::Korean => "한국어",
            Language::Arabic => "العربية",
        }
    }
    
    /// Parse from language code
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" => Some(Language::English),
            "es" => Some(Language::Spanish),
            "fr" => Some(Language::French),
            "de" => Some(Language::German),
            "ja" => Some(Language::Japanese),
            "zh" => Some(Language::ChineseSimplified),
            "pt" => Some(Language::Portuguese),
            "ru" => Some(Language::Russian),
            "ko" => Some(Language::Korean),
            "ar" => Some(Language::Arabic),
            _ => None,
        }
    }
    
    /// Get all supported languages
    pub fn all() -> Vec<Language> {
        vec![
            Language::English,
            Language::Spanish,
            Language::French,
            Language::German,
            Language::Japanese,
            Language::ChineseSimplified,
            Language::Portuguese,
            Language::Russian,
            Language::Korean,
            Language::Arabic,
        ]
    }
}

/// Message keys for internationalization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MessageKey {
    // Error messages
    ModelNotLoaded,
    InvalidInput,
    InvalidConfiguration,
    ComputationError,
    FileNotFound,
    OutOfMemory,
    
    // Status messages
    ModelLoaded,
    ProcessingComplete,
    SystemReady,
    LowPower,
    HighPerformance,
    
    // User guidance
    CheckInput,
    OptimizeSettings,
    ContactSupport,
    
    // Units and measurements
    Milliwatts,
    Milliseconds,
    Percent,
    DecibelsUnits,
    
    // Features
    AdaptiveTimestep,
    PowerOptimization,
    VoiceActivity,
    KeywordSpotting,
}

/// Internationalization manager
pub struct I18nManager {
    current_language: Language,
    messages: BTreeMap<Language, BTreeMap<MessageKey, &'static str>>,
}

impl I18nManager {
    /// Create new i18n manager with default language (English)
    pub fn new() -> Self {
        let mut manager = Self {
            current_language: Language::English,
            messages: BTreeMap::new(),
        };
        manager.initialize_messages();
        manager
    }
    
    /// Set current language
    pub fn set_language(&mut self, language: Language) {
        self.current_language = language;
    }
    
    /// Get current language
    pub fn current_language(&self) -> Language {
        self.current_language
    }
    
    /// Get localized message
    pub fn get_message(&self, key: MessageKey) -> &'static str {
        self.messages
            .get(&self.current_language)
            .and_then(|lang_messages| lang_messages.get(&key))
            .unwrap_or_else(|| {
                // Fallback to English if translation not available
                self.messages
                    .get(&Language::English)
                    .and_then(|lang_messages| lang_messages.get(&key))
                    .unwrap_or(&"[Missing Translation]")
            })
    }
    
    /// Get formatted error message with context
    pub fn format_error(&self, key: MessageKey, context: Option<&str>) -> String {
        let message = self.get_message(key);
        match context {
            Some(ctx) => format!("{}: {}", message, ctx),
            None => message.to_string(),
        }
    }
    
    /// Initialize all message translations
    fn initialize_messages(&mut self) {
        // English (default)
        let mut en = BTreeMap::new();
        en.insert(MessageKey::ModelNotLoaded, "Model not loaded");
        en.insert(MessageKey::InvalidInput, "Invalid input");
        en.insert(MessageKey::InvalidConfiguration, "Invalid configuration");
        en.insert(MessageKey::ComputationError, "Computation error");
        en.insert(MessageKey::FileNotFound, "File not found");
        en.insert(MessageKey::OutOfMemory, "Out of memory");
        en.insert(MessageKey::ModelLoaded, "Model loaded successfully");
        en.insert(MessageKey::ProcessingComplete, "Processing complete");
        en.insert(MessageKey::SystemReady, "System ready");
        en.insert(MessageKey::LowPower, "Low power mode");
        en.insert(MessageKey::HighPerformance, "High performance mode");
        en.insert(MessageKey::CheckInput, "Please check your input");
        en.insert(MessageKey::OptimizeSettings, "Consider optimizing settings");
        en.insert(MessageKey::ContactSupport, "Please contact support");
        en.insert(MessageKey::Milliwatts, "mW");
        en.insert(MessageKey::Milliseconds, "ms");
        en.insert(MessageKey::Percent, "%");
        en.insert(MessageKey::DecibelsUnits, "dB");
        en.insert(MessageKey::AdaptiveTimestep, "Adaptive Timestep");
        en.insert(MessageKey::PowerOptimization, "Power Optimization");
        en.insert(MessageKey::VoiceActivity, "Voice Activity Detection");
        en.insert(MessageKey::KeywordSpotting, "Keyword Spotting");
        self.messages.insert(Language::English, en);

        // Spanish
        let mut es = BTreeMap::new();
        es.insert(MessageKey::ModelNotLoaded, "Modelo no cargado");
        es.insert(MessageKey::InvalidInput, "Entrada inválida");
        es.insert(MessageKey::InvalidConfiguration, "Configuración inválida");
        es.insert(MessageKey::ComputationError, "Error de cálculo");
        es.insert(MessageKey::FileNotFound, "Archivo no encontrado");
        es.insert(MessageKey::OutOfMemory, "Sin memoria");
        es.insert(MessageKey::ModelLoaded, "Modelo cargado exitosamente");
        es.insert(MessageKey::ProcessingComplete, "Procesamiento completado");
        es.insert(MessageKey::SystemReady, "Sistema listo");
        es.insert(MessageKey::LowPower, "Modo de bajo consumo");
        es.insert(MessageKey::HighPerformance, "Modo de alto rendimiento");
        es.insert(MessageKey::CheckInput, "Por favor verifique su entrada");
        es.insert(MessageKey::OptimizeSettings, "Considere optimizar la configuración");
        es.insert(MessageKey::ContactSupport, "Por favor contacte soporte");
        es.insert(MessageKey::Milliwatts, "mW");
        es.insert(MessageKey::Milliseconds, "ms");
        es.insert(MessageKey::Percent, "%");
        es.insert(MessageKey::DecibelsUnits, "dB");
        es.insert(MessageKey::AdaptiveTimestep, "Paso de Tiempo Adaptativo");
        es.insert(MessageKey::PowerOptimization, "Optimización de Energía");
        es.insert(MessageKey::VoiceActivity, "Detección de Actividad de Voz");
        es.insert(MessageKey::KeywordSpotting, "Detección de Palabras Clave");
        self.messages.insert(Language::Spanish, es);

        // French
        let mut fr = BTreeMap::new();
        fr.insert(MessageKey::ModelNotLoaded, "Modèle non chargé");
        fr.insert(MessageKey::InvalidInput, "Entrée invalide");
        fr.insert(MessageKey::InvalidConfiguration, "Configuration invalide");
        fr.insert(MessageKey::ComputationError, "Erreur de calcul");
        fr.insert(MessageKey::FileNotFound, "Fichier introuvable");
        fr.insert(MessageKey::OutOfMemory, "Mémoire insuffisante");
        fr.insert(MessageKey::ModelLoaded, "Modèle chargé avec succès");
        fr.insert(MessageKey::ProcessingComplete, "Traitement terminé");
        fr.insert(MessageKey::SystemReady, "Système prêt");
        fr.insert(MessageKey::LowPower, "Mode basse consommation");
        fr.insert(MessageKey::HighPerformance, "Mode haute performance");
        fr.insert(MessageKey::CheckInput, "Veuillez vérifier votre entrée");
        fr.insert(MessageKey::OptimizeSettings, "Considérez optimiser les paramètres");
        fr.insert(MessageKey::ContactSupport, "Veuillez contacter le support");
        fr.insert(MessageKey::Milliwatts, "mW");
        fr.insert(MessageKey::Milliseconds, "ms");
        fr.insert(MessageKey::Percent, "%");
        fr.insert(MessageKey::DecibelsUnits, "dB");
        fr.insert(MessageKey::AdaptiveTimestep, "Pas de Temps Adaptatif");
        fr.insert(MessageKey::PowerOptimization, "Optimisation d'Énergie");
        fr.insert(MessageKey::VoiceActivity, "Détection d'Activité Vocale");
        fr.insert(MessageKey::KeywordSpotting, "Détection de Mots-Clés");
        self.messages.insert(Language::French, fr);

        // German
        let mut de = BTreeMap::new();
        de.insert(MessageKey::ModelNotLoaded, "Modell nicht geladen");
        de.insert(MessageKey::InvalidInput, "Ungültige Eingabe");
        de.insert(MessageKey::InvalidConfiguration, "Ungültige Konfiguration");
        de.insert(MessageKey::ComputationError, "Berechnungsfehler");
        de.insert(MessageKey::FileNotFound, "Datei nicht gefunden");
        de.insert(MessageKey::OutOfMemory, "Speicher erschöpft");
        de.insert(MessageKey::ModelLoaded, "Modell erfolgreich geladen");
        de.insert(MessageKey::ProcessingComplete, "Verarbeitung abgeschlossen");
        de.insert(MessageKey::SystemReady, "System bereit");
        de.insert(MessageKey::LowPower, "Energiesparmodus");
        de.insert(MessageKey::HighPerformance, "Hochleistungsmodus");
        de.insert(MessageKey::CheckInput, "Bitte überprüfen Sie Ihre Eingabe");
        de.insert(MessageKey::OptimizeSettings, "Erwägen Sie Einstellungen zu optimieren");
        de.insert(MessageKey::ContactSupport, "Bitte kontaktieren Sie den Support");
        de.insert(MessageKey::Milliwatts, "mW");
        de.insert(MessageKey::Milliseconds, "ms");
        de.insert(MessageKey::Percent, "%");
        de.insert(MessageKey::DecibelsUnits, "dB");
        de.insert(MessageKey::AdaptiveTimestep, "Adaptive Zeitschritte");
        de.insert(MessageKey::PowerOptimization, "Energieoptimierung");
        de.insert(MessageKey::VoiceActivity, "Sprachaktivitätserkennung");
        de.insert(MessageKey::KeywordSpotting, "Schlüsselworterkennung");
        self.messages.insert(Language::German, de);

        // Japanese
        let mut ja = BTreeMap::new();
        ja.insert(MessageKey::ModelNotLoaded, "モデルが読み込まれていません");
        ja.insert(MessageKey::InvalidInput, "無効な入力");
        ja.insert(MessageKey::InvalidConfiguration, "無効な設定");
        ja.insert(MessageKey::ComputationError, "計算エラー");
        ja.insert(MessageKey::FileNotFound, "ファイルが見つかりません");
        ja.insert(MessageKey::OutOfMemory, "メモリ不足");
        ja.insert(MessageKey::ModelLoaded, "モデルの読み込み完了");
        ja.insert(MessageKey::ProcessingComplete, "処理完了");
        ja.insert(MessageKey::SystemReady, "システム準備完了");
        ja.insert(MessageKey::LowPower, "低消費電力モード");
        ja.insert(MessageKey::HighPerformance, "高性能モード");
        ja.insert(MessageKey::CheckInput, "入力を確認してください");
        ja.insert(MessageKey::OptimizeSettings, "設定の最適化を検討してください");
        ja.insert(MessageKey::ContactSupport, "サポートにお問い合わせください");
        ja.insert(MessageKey::Milliwatts, "mW");
        ja.insert(MessageKey::Milliseconds, "ms");
        ja.insert(MessageKey::Percent, "%");
        ja.insert(MessageKey::DecibelsUnits, "dB");
        ja.insert(MessageKey::AdaptiveTimestep, "適応的時間ステップ");
        ja.insert(MessageKey::PowerOptimization, "電力最適化");
        ja.insert(MessageKey::VoiceActivity, "音声活動検出");
        ja.insert(MessageKey::KeywordSpotting, "キーワード検出");
        self.messages.insert(Language::Japanese, ja);

        // Chinese (Simplified)
        let mut zh = BTreeMap::new();
        zh.insert(MessageKey::ModelNotLoaded, "模型未加载");
        zh.insert(MessageKey::InvalidInput, "无效输入");
        zh.insert(MessageKey::InvalidConfiguration, "无效配置");
        zh.insert(MessageKey::ComputationError, "计算错误");
        zh.insert(MessageKey::FileNotFound, "文件未找到");
        zh.insert(MessageKey::OutOfMemory, "内存不足");
        zh.insert(MessageKey::ModelLoaded, "模型加载成功");
        zh.insert(MessageKey::ProcessingComplete, "处理完成");
        zh.insert(MessageKey::SystemReady, "系统就绪");
        zh.insert(MessageKey::LowPower, "低功耗模式");
        zh.insert(MessageKey::HighPerformance, "高性能模式");
        zh.insert(MessageKey::CheckInput, "请检查您的输入");
        zh.insert(MessageKey::OptimizeSettings, "请考虑优化设置");
        zh.insert(MessageKey::ContactSupport, "请联系技术支持");
        zh.insert(MessageKey::Milliwatts, "毫瓦");
        zh.insert(MessageKey::Milliseconds, "毫秒");
        zh.insert(MessageKey::Percent, "%");
        zh.insert(MessageKey::DecibelsUnits, "分贝");
        zh.insert(MessageKey::AdaptiveTimestep, "自适应时间步长");
        zh.insert(MessageKey::PowerOptimization, "功耗优化");
        zh.insert(MessageKey::VoiceActivity, "语音活动检测");
        zh.insert(MessageKey::KeywordSpotting, "关键词识别");
        self.messages.insert(Language::ChineseSimplified, zh);
    }
}

impl Default for I18nManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global i18n instance
#[cfg(feature = "std")]
static GLOBAL_I18N: OnceLock<Mutex<I18nManager>> = OnceLock::new();

#[cfg(not(feature = "std"))]
static mut GLOBAL_I18N: Option<I18nManager> = None;

/// Get global i18n manager (lazy initialization)
#[cfg(feature = "std")]
pub fn get_i18n() -> &'static Mutex<I18nManager> {
    GLOBAL_I18N.get_or_init(|| Mutex::new(I18nManager::new()))
}

#[cfg(not(feature = "std"))]
pub fn get_i18n() -> &'static mut I18nManager {
    unsafe {
        if GLOBAL_I18N.is_none() {
            GLOBAL_I18N = Some(I18nManager::new());
        }
        GLOBAL_I18N.as_mut().unwrap()
    }
}

/// Set global language
#[cfg(feature = "std")]
pub fn set_global_language(language: Language) {
    if let Ok(mut manager) = get_i18n().lock() {
        manager.set_language(language);
    }
}

#[cfg(not(feature = "std"))]
pub fn set_global_language(language: Language) {
    get_i18n().set_language(language);
}

/// Get localized message using global i18n manager
#[cfg(feature = "std")]
pub fn t(key: MessageKey) -> String {
    get_i18n()
        .lock()
        .map(|manager| manager.get_message(key).to_string())
        .unwrap_or_else(|_| "Translation error".to_string())
}

#[cfg(not(feature = "std"))]
pub fn t(key: MessageKey) -> &'static str {
    get_i18n().get_message(key)
}

/// Format error message using global i18n manager
#[cfg(feature = "std")]
pub fn t_error(key: MessageKey, context: Option<&str>) -> String {
    get_i18n()
        .lock()
        .map(|manager| manager.format_error(key, context))
        .unwrap_or_else(|_| "Translation error".to_string())
}

#[cfg(not(feature = "std"))]
pub fn t_error(key: MessageKey, context: Option<&str>) -> String {
    get_i18n().format_error(key, context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_codes() {
        assert_eq!(Language::English.code(), "en");
        assert_eq!(Language::Spanish.code(), "es");
        assert_eq!(Language::Japanese.code(), "ja");
    }

    #[test]
    fn test_language_from_code() {
        assert_eq!(Language::from_code("en"), Some(Language::English));
        assert_eq!(Language::from_code("es"), Some(Language::Spanish));
        assert_eq!(Language::from_code("invalid"), None);
    }

    #[test]
    fn test_i18n_manager() {
        let mut manager = I18nManager::new();
        
        // Test English (default)
        assert_eq!(manager.get_message(MessageKey::ModelNotLoaded), "Model not loaded");
        
        // Test Spanish
        manager.set_language(Language::Spanish);
        assert_eq!(manager.get_message(MessageKey::ModelNotLoaded), "Modelo no cargado");
        
        // Test fallback for missing translation
        manager.set_language(Language::Portuguese); // Not fully implemented
        assert!(manager.get_message(MessageKey::ModelNotLoaded) == "Model not loaded"); // Falls back to English
    }

    #[test]
    fn test_format_error() {
        let manager = I18nManager::new();
        let formatted = manager.format_error(MessageKey::InvalidInput, Some("buffer size"));
        assert!(formatted.contains("Invalid input"));
        assert!(formatted.contains("buffer size"));
    }
}