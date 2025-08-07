//! Compliance and privacy module for global deployment
//!
//! Implements GDPR, CCPA, PDPA and other privacy regulations
//! compliance features for international markets.

use crate::{Result, LiquidAudioError};

#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// Privacy regulation frameworks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrivacyFramework {
    /// General Data Protection Regulation (EU)
    GDPR,
    /// California Consumer Privacy Act (US-CA)
    CCPA,
    /// Personal Data Protection Act (Singapore)
    PDPA,
    /// Lei Geral de Proteção de Dados (Brazil)
    LGPD,
    /// Privacy Act (Australia)
    AustralianPrivacyAct,
    /// Personal Information Protection Law (China)
    PIPL,
}

impl PrivacyFramework {
    /// Get framework name
    pub fn name(&self) -> &'static str {
        match self {
            PrivacyFramework::GDPR => "GDPR",
            PrivacyFramework::CCPA => "CCPA",
            PrivacyFramework::PDPA => "PDPA",
            PrivacyFramework::LGPD => "LGPD",
            PrivacyFramework::AustralianPrivacyAct => "Australian Privacy Act",
            PrivacyFramework::PIPL => "PIPL",
        }
    }
    
    /// Get jurisdiction
    pub fn jurisdiction(&self) -> &'static str {
        match self {
            PrivacyFramework::GDPR => "European Union",
            PrivacyFramework::CCPA => "California, USA",
            PrivacyFramework::PDPA => "Singapore",
            PrivacyFramework::LGPD => "Brazil",
            PrivacyFramework::AustralianPrivacyAct => "Australia",
            PrivacyFramework::PIPL => "China",
        }
    }
}

/// Privacy compliance configuration
#[derive(Debug, Clone)]
pub struct ComplianceConfig {
    /// Applicable privacy frameworks
    pub frameworks: Vec<PrivacyFramework>,
    /// Enable data minimization
    pub data_minimization: bool,
    /// Enable purpose limitation
    pub purpose_limitation: bool,
    /// Enable storage limitation
    pub storage_limitation: bool,
    /// Enable data subject rights
    pub data_subject_rights: bool,
    /// Enable audit logging
    pub audit_logging: bool,
    /// Enable consent management
    pub consent_management: bool,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            frameworks: vec![PrivacyFramework::GDPR], // Default to most strict
            data_minimization: true,
            purpose_limitation: true,
            storage_limitation: true,
            data_subject_rights: true,
            audit_logging: true,
            consent_management: true,
        }
    }
}

/// Data processing lawful basis (GDPR Article 6)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LawfulBasis {
    /// Consent of the data subject
    Consent,
    /// Performance of a contract
    Contract,
    /// Compliance with legal obligation
    LegalObligation,
    /// Protection of vital interests
    VitalInterests,
    /// Performance of task in public interest
    PublicTask,
    /// Legitimate interests
    LegitimateInterests,
}

/// Data category for privacy impact assessment
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataCategory {
    /// Audio data
    AudioData,
    /// Processing results
    ProcessingResults,
    /// Performance metrics
    PerformanceMetrics,
    /// Usage statistics
    UsageStatistics,
    /// Error logs
    ErrorLogs,
    /// Configuration data
    ConfigurationData,
}

impl DataCategory {
    /// Check if data category contains personal data
    pub fn is_personal_data(&self) -> bool {
        match self {
            DataCategory::AudioData => true, // Voice is biometric personal data
            DataCategory::ProcessingResults => true, // May contain voice patterns
            DataCategory::PerformanceMetrics => false, // Aggregated technical data
            DataCategory::UsageStatistics => false, // Aggregated usage data
            DataCategory::ErrorLogs => false, // Technical error information
            DataCategory::ConfigurationData => false, // Device settings
        }
    }
    
    /// Get privacy risk level (1-5, 5 being highest risk)
    pub fn privacy_risk_level(&self) -> u8 {
        match self {
            DataCategory::AudioData => 5, // High risk - biometric data
            DataCategory::ProcessingResults => 4, // High risk - derived from biometric
            DataCategory::PerformanceMetrics => 1, // Low risk - technical metrics
            DataCategory::UsageStatistics => 2, // Low risk - aggregated data
            DataCategory::ErrorLogs => 1, // Low risk - technical information
            DataCategory::ConfigurationData => 1, // Low risk - device settings
        }
    }
}

/// Privacy compliance manager
pub struct PrivacyManager {
    config: ComplianceConfig,
    audit_trail: Vec<AuditEvent>,
    consent_records: Vec<ConsentRecord>,
}

impl PrivacyManager {
    /// Create new privacy manager
    pub fn new(config: ComplianceConfig) -> Self {
        Self {
            config,
            audit_trail: Vec::new(),
            consent_records: Vec::new(),
        }
    }
    
    /// Check if data processing is compliant
    pub fn validate_processing(
        &self, 
        category: DataCategory, 
        basis: LawfulBasis,
        purpose: &str
    ) -> Result<bool> {
        // Check if personal data processing has proper basis
        if category.is_personal_data() {
            match basis {
                LawfulBasis::Consent => {
                    // Check if valid consent exists
                    if !self.has_valid_consent(category) {
                        return Err(LiquidAudioError::SecurityError(
                            "No valid consent for personal data processing".to_string()
                        ));
                    }
                }
                LawfulBasis::LegitimateInterests => {
                    // Perform balancing test
                    if category.privacy_risk_level() > 3 {
                        return Err(LiquidAudioError::SecurityError(
                            "Legitimate interests insufficient for high-risk data".to_string()
                        ));
                    }
                }
                _ => {} // Other bases accepted
            }
        }
        
        // Check data minimization
        if self.config.data_minimization {
            if !self.is_data_necessary(category, purpose) {
                return Err(LiquidAudioError::SecurityError(
                    "Data processing violates data minimization principle".to_string()
                ));
            }
        }
        
        // Log audit event
        if self.config.audit_logging {
            self.log_processing_event(category, basis, purpose);
        }
        
        Ok(true)
    }
    
    /// Check if valid consent exists
    fn has_valid_consent(&self, category: DataCategory) -> bool {
        self.consent_records.iter().any(|record| {
            record.category == category && 
            record.is_valid() &&
            !record.is_expired()
        })
    }
    
    /// Check if data processing is necessary
    fn is_data_necessary(&self, category: DataCategory, purpose: &str) -> bool {
        // Define necessary data for common purposes
        match purpose {
            "keyword_spotting" => {
                matches!(category, DataCategory::AudioData | DataCategory::ProcessingResults)
            }
            "voice_activity_detection" => {
                matches!(category, DataCategory::AudioData | DataCategory::ProcessingResults)
            }
            "performance_monitoring" => {
                matches!(category, DataCategory::PerformanceMetrics | DataCategory::UsageStatistics)
            }
            "error_reporting" => {
                matches!(category, DataCategory::ErrorLogs)
            }
            _ => true, // Allow by default for undefined purposes
        }
    }
    
    /// Log processing event
    fn log_processing_event(&self, category: DataCategory, basis: LawfulBasis, purpose: &str) {
        // In real implementation, this would write to secure audit log
        println!("AUDIT: Processing {} under {:?} for purpose: {}", 
                category.is_personal_data().then(|| "personal data").unwrap_or("non-personal data"),
                basis, 
                purpose);
    }
    
    /// Record user consent
    pub fn record_consent(&mut self, category: DataCategory, purpose: String) -> Result<()> {
        let consent = ConsentRecord {
            category,
            purpose,
            timestamp: Self::current_timestamp(),
            expires_at: Self::current_timestamp() + (365 * 24 * 60 * 60), // 1 year
            withdrawn: false,
        };
        
        self.consent_records.push(consent);
        Ok(())
    }
    
    /// Withdraw consent
    pub fn withdraw_consent(&mut self, category: DataCategory) -> Result<()> {
        for record in &mut self.consent_records {
            if record.category == category {
                record.withdrawn = true;
            }
        }
        Ok(())
    }
    
    /// Get privacy notice text
    pub fn get_privacy_notice(&self, framework: PrivacyFramework) -> String {
        match framework {
            PrivacyFramework::GDPR => {
                "This application processes audio data for AI inference. \
                Processing is based on your consent or legitimate interests. \
                You have the right to access, rectify, erase, restrict, \
                object to processing, and data portability. \
                Contact us to exercise your rights.".to_string()
            }
            PrivacyFramework::CCPA => {
                "This application may collect and process personal information \
                including audio data. You have the right to know, delete, \
                opt-out of sale, and non-discrimination. \
                See our privacy policy for details.".to_string()
            }
            _ => {
                "This application processes data in accordance with applicable \
                privacy laws. See our privacy policy for details.".to_string()
            }
        }
    }
    
    /// Get current timestamp (seconds since epoch)
    fn current_timestamp() -> u64 {
        // Simplified timestamp - real implementation would use system time
        12345678_u64
    }
    
    /// Generate compliance report
    pub fn generate_compliance_report(&self) -> ComplianceReport {
        ComplianceReport {
            frameworks: self.config.frameworks.clone(),
            total_consents: self.consent_records.len(),
            active_consents: self.consent_records.iter().filter(|r| r.is_valid()).count(),
            audit_events: self.audit_trail.len(),
            high_risk_processing: self.audit_trail.iter()
                .filter(|event| event.risk_level > 3)
                .count(),
        }
    }
}

/// Consent record
#[derive(Debug, Clone)]
pub struct ConsentRecord {
    pub category: DataCategory,
    pub purpose: String,
    pub timestamp: u64,
    pub expires_at: u64,
    pub withdrawn: bool,
}

impl ConsentRecord {
    /// Check if consent is valid (not withdrawn)
    pub fn is_valid(&self) -> bool {
        !self.withdrawn
    }
    
    /// Check if consent is expired
    pub fn is_expired(&self) -> bool {
        let current_time = PrivacyManager::current_timestamp();
        current_time > self.expires_at
    }
}

/// Audit event
#[derive(Debug, Clone)]
pub struct AuditEvent {
    pub category: DataCategory,
    pub lawful_basis: LawfulBasis,
    pub purpose: String,
    pub timestamp: u64,
    pub risk_level: u8,
}

/// Compliance report
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    pub frameworks: Vec<PrivacyFramework>,
    pub total_consents: usize,
    pub active_consents: usize,
    pub audit_events: usize,
    pub high_risk_processing: usize,
}

/// Privacy by design helper
pub struct PrivacyByDesign;

impl PrivacyByDesign {
    /// Validate configuration for privacy compliance
    pub fn validate_config(config: &crate::ModelConfig) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        // Check for privacy-friendly defaults
        if config.sample_rate > 16000 {
            recommendations.push(
                "Consider reducing sample rate to minimize data collection".to_string()
            );
        }
        
        // Check memory usage (large models store more data)
        let estimated_memory = (config.input_dim * config.hidden_dim) * 4;
        if estimated_memory > 1024 * 1024 {
            recommendations.push(
                "Large model may require more extensive privacy impact assessment".to_string()
            );
        }
        
        Ok(recommendations)
    }
    
    /// Generate data retention policy
    pub fn default_retention_policy() -> String {
        "Audio data: Processed immediately and not stored. \
        Processing results: Retained for 30 days for quality assurance. \
        Performance metrics: Aggregated data retained for 2 years. \
        Error logs: Retained for 90 days for debugging.".to_string()
    }
    
    /// Check if processing can be performed without consent (legitimate interest)
    pub fn can_use_legitimate_interest(category: DataCategory, purpose: &str) -> bool {
        // Conservative approach - only allow for low-risk technical data
        !category.is_personal_data() && 
        category.privacy_risk_level() <= 2 &&
        matches!(purpose, "performance_monitoring" | "error_reporting")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_category_classification() {
        assert!(DataCategory::AudioData.is_personal_data());
        assert!(!DataCategory::PerformanceMetrics.is_personal_data());
        assert_eq!(DataCategory::AudioData.privacy_risk_level(), 5);
        assert_eq!(DataCategory::PerformanceMetrics.privacy_risk_level(), 1);
    }

    #[test]
    fn test_privacy_manager_validation() {
        let config = ComplianceConfig::default();
        let mut manager = PrivacyManager::new(config);
        
        // Record consent
        manager.record_consent(
            DataCategory::AudioData, 
            "keyword_spotting".to_string()
        ).unwrap();
        
        // Test valid processing
        let result = manager.validate_processing(
            DataCategory::AudioData,
            LawfulBasis::Consent,
            "keyword_spotting"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_consent_management() {
        let config = ComplianceConfig::default();
        let mut manager = PrivacyManager::new(config);
        
        // Record and withdraw consent
        manager.record_consent(
            DataCategory::AudioData, 
            "test".to_string()
        ).unwrap();
        
        assert!(manager.has_valid_consent(DataCategory::AudioData));
        
        manager.withdraw_consent(DataCategory::AudioData).unwrap();
        assert!(!manager.has_valid_consent(DataCategory::AudioData));
    }

    #[test]
    fn test_privacy_frameworks() {
        assert_eq!(PrivacyFramework::GDPR.name(), "GDPR");
        assert_eq!(PrivacyFramework::GDPR.jurisdiction(), "European Union");
        assert_eq!(PrivacyFramework::CCPA.jurisdiction(), "California, USA");
    }

    #[test]
    fn test_legitimate_interest() {
        assert!(PrivacyByDesign::can_use_legitimate_interest(
            DataCategory::PerformanceMetrics,
            "performance_monitoring"
        ));
        
        assert!(!PrivacyByDesign::can_use_legitimate_interest(
            DataCategory::AudioData,
            "keyword_spotting"
        ));
    }
}