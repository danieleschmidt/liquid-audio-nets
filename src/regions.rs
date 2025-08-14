//! Multi-region deployment and configuration
//!
//! Supports global deployment with region-specific optimizations,
//! regulatory compliance, and performance tuning.

use crate::{Result, LiquidAudioError, ModelConfig};

#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// Geographic regions for deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Region {
    /// North America
    NorthAmerica,
    /// Europe
    Europe,
    /// Asia Pacific
    AsiaPacific,
    /// Latin America
    LatinAmerica,
    /// Middle East and Africa
    MiddleEastAfrica,
    /// China (separate due to regulatory requirements)
    China,
}

impl Region {
    /// Get region code
    pub fn code(&self) -> &'static str {
        match self {
            Region::NorthAmerica => "na",
            Region::Europe => "eu",
            Region::AsiaPacific => "ap",
            Region::LatinAmerica => "latam",
            Region::MiddleEastAfrica => "mea",
            Region::China => "cn",
        }
    }
    
    /// Get region name
    pub fn name(&self) -> &'static str {
        match self {
            Region::NorthAmerica => "North America",
            Region::Europe => "Europe",
            Region::AsiaPacific => "Asia Pacific",
            Region::LatinAmerica => "Latin America",
            Region::MiddleEastAfrica => "Middle East & Africa",
            Region::China => "China",
        }
    }
    
    /// Get primary languages for region
    pub fn primary_languages(&self) -> Vec<crate::i18n::Language> {
        use crate::i18n::Language;
        match self {
            Region::NorthAmerica => vec![Language::English, Language::Spanish],
            Region::Europe => vec![Language::English, Language::German, Language::French, Language::Spanish],
            Region::AsiaPacific => vec![Language::English, Language::Japanese, Language::Korean],
            Region::LatinAmerica => vec![Language::Spanish, Language::Portuguese],
            Region::MiddleEastAfrica => vec![Language::English, Language::Arabic],
            Region::China => vec![Language::ChineseSimplified],
        }
    }
    
    /// Get applicable privacy frameworks
    pub fn privacy_frameworks(&self) -> Vec<crate::compliance::PrivacyFramework> {
        use crate::compliance::PrivacyFramework;
        match self {
            Region::NorthAmerica => vec![PrivacyFramework::CCPA],
            Region::Europe => vec![PrivacyFramework::GDPR],
            Region::AsiaPacific => vec![PrivacyFramework::PDPA, PrivacyFramework::AustralianPrivacyAct],
            Region::LatinAmerica => vec![PrivacyFramework::LGPD],
            Region::MiddleEastAfrica => vec![PrivacyFramework::GDPR], // Many countries follow GDPR
            Region::China => vec![PrivacyFramework::PIPL],
        }
    }
    
    /// Get timezone offset ranges
    pub fn timezone_ranges(&self) -> (i8, i8) {
        match self {
            Region::NorthAmerica => (-12, -3), // UTC-12 to UTC-3
            Region::Europe => (-1, 4),         // UTC-1 to UTC+4
            Region::AsiaPacific => (5, 12),    // UTC+5 to UTC+12
            Region::LatinAmerica => (-6, -3),  // UTC-6 to UTC-3
            Region::MiddleEastAfrica => (0, 4), // UTC+0 to UTC+4
            Region::China => (8, 8),           // UTC+8
        }
    }
}

/// Region-specific configuration
#[derive(Debug, Clone)]
pub struct RegionalConfig {
    /// Target region
    pub region: Region,
    /// Language preferences
    pub languages: Vec<crate::i18n::Language>,
    /// Privacy compliance settings
    pub privacy_config: crate::compliance::ComplianceConfig,
    /// Performance optimizations
    pub performance_profile: PerformanceProfile,
    /// Network latency target (ms)
    pub latency_target_ms: u32,
    /// Data residency requirements
    pub data_residency: DataResidency,
}

/// Performance optimization profiles
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerformanceProfile {
    /// Optimize for lowest power consumption
    UltraLowPower,
    /// Balanced power and performance
    Balanced,
    /// Optimize for highest performance
    HighPerformance,
    /// Optimize for real-time constraints
    RealTime,
}

impl PerformanceProfile {
    /// Get power budget in milliwatts
    pub fn power_budget_mw(&self) -> f32 {
        match self {
            PerformanceProfile::UltraLowPower => 0.5,
            PerformanceProfile::Balanced => 1.5,
            PerformanceProfile::HighPerformance => 5.0,
            PerformanceProfile::RealTime => 3.0,
        }
    }
    
    /// Get latency budget in milliseconds
    pub fn latency_budget_ms(&self) -> f32 {
        match self {
            PerformanceProfile::UltraLowPower => 50.0,
            PerformanceProfile::Balanced => 20.0,
            PerformanceProfile::HighPerformance => 10.0,
            PerformanceProfile::RealTime => 5.0,
        }
    }
    
    /// Get recommended model size
    pub fn recommended_model_size(&self) -> ModelSize {
        match self {
            PerformanceProfile::UltraLowPower => ModelSize::Micro,
            PerformanceProfile::Balanced => ModelSize::Small,
            PerformanceProfile::HighPerformance => ModelSize::Medium,
            PerformanceProfile::RealTime => ModelSize::Small,
        }
    }
}

/// Model size categories
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    /// <64KB - Ultra-constrained devices
    Micro,
    /// 64-256KB - Embedded devices
    Small,
    /// 256KB-1MB - Edge devices
    Medium,
    /// 1-4MB - Mobile devices
    Large,
}

impl ModelSize {
    /// Get memory budget in bytes
    pub fn memory_budget_bytes(&self) -> usize {
        match self {
            ModelSize::Micro => 64 * 1024,
            ModelSize::Small => 256 * 1024,
            ModelSize::Medium => 1024 * 1024,
            ModelSize::Large => 4 * 1024 * 1024,
        }
    }
    
    /// Get recommended hidden dimension
    pub fn recommended_hidden_dim(&self) -> usize {
        match self {
            ModelSize::Micro => 32,
            ModelSize::Small => 64,
            ModelSize::Medium => 128,
            ModelSize::Large => 256,
        }
    }
}

/// Data residency requirements
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataResidency {
    /// Data can be processed globally
    Global,
    /// Data must remain in region
    Regional,
    /// Data must remain in country
    National,
    /// Data must remain on-device
    OnDevice,
}

/// Regional deployment manager
pub struct RegionalManager {
    configs: Vec<RegionalConfig>,
    default_region: Region,
}

impl RegionalManager {
    /// Create new regional manager
    pub fn new(default_region: Region) -> Self {
        let mut manager = Self {
            configs: Vec::new(),
            default_region,
        };
        manager.initialize_default_configs();
        manager
    }
    
    /// Get configuration for region
    pub fn get_config(&self, region: Region) -> Result<&RegionalConfig> {
        self.configs
            .iter()
            .find(|config| config.region == region)
            .ok_or_else(|| LiquidAudioError::ConfigError(
                format!("No configuration found for region: {:?}", region)
            ))
    }
    
    /// Get optimal model configuration for region
    pub fn get_model_config(&self, region: Region) -> Result<ModelConfig> {
        let regional_config = self.get_config(region)?;
        let model_size = regional_config.performance_profile.recommended_model_size();
        
        Ok(ModelConfig {
            input_dim: 40,  // MFCC features
            hidden_dim: model_size.recommended_hidden_dim(),
            output_dim: 10, // Common number of classes
            sample_rate: self.get_optimal_sample_rate(region),
            frame_size: 512, // Standard frame size
            model_type: format!("{}_optimized", region.code()),
        })
    }
    
    /// Get optimal sample rate for region
    fn get_optimal_sample_rate(&self, region: Region) -> u32 {
        // Optimize sample rate based on regional constraints
        match region {
            Region::China => 8000,         // Lower bandwidth
            Region::MiddleEastAfrica => 8000, // Infrastructure constraints
            Region::LatinAmerica => 8000,  // Mobile-first
            _ => 16000,                    // Standard rate
        }
    }
    
    /// Auto-detect optimal region based on system locale
    pub fn detect_region() -> Region {
        // Simplified detection - real implementation would use system locale,
        // IP geolocation, or other detection mechanisms
        Region::NorthAmerica // Default fallback
    }
    
    /// Get compliance report for region
    pub fn get_compliance_summary(&self, region: Region) -> Result<String> {
        let config = self.get_config(region)?;
        let frameworks = config.privacy_config.frameworks.iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join(", ");
        
        Ok(format!(
            "Region: {}\nApplicable frameworks: {}\nData residency: {:?}\nLanguages: {}",
            region.name(),
            frameworks,
            config.data_residency,
            config.languages.len()
        ))
    }
    
    /// Initialize default configurations for all regions
    fn initialize_default_configs(&mut self) {
        for region in [
            Region::NorthAmerica,
            Region::Europe,
            Region::AsiaPacific,
            Region::LatinAmerica,
            Region::MiddleEastAfrica,
            Region::China,
        ] {
            let config = RegionalConfig {
                region,
                languages: region.primary_languages(),
                privacy_config: crate::compliance::ComplianceConfig {
                    frameworks: region.privacy_frameworks(),
                    data_minimization: true,
                    purpose_limitation: true,
                    storage_limitation: true,
                    data_subject_rights: true,
                    audit_logging: true,
                    consent_management: region != Region::China, // Different consent model
                },
                performance_profile: self.get_default_performance_profile(region),
                latency_target_ms: self.get_default_latency_target(region),
                data_residency: self.get_default_data_residency(region),
            };
            self.configs.push(config);
        }
    }
    
    /// Get default performance profile for region
    fn get_default_performance_profile(&self, region: Region) -> PerformanceProfile {
        match region {
            Region::China => PerformanceProfile::UltraLowPower, // Mobile-first
            Region::MiddleEastAfrica => PerformanceProfile::UltraLowPower, // Power constraints
            Region::LatinAmerica => PerformanceProfile::Balanced, // Mixed infrastructure
            Region::Europe => PerformanceProfile::Balanced, // Efficiency focus
            Region::NorthAmerica => PerformanceProfile::HighPerformance, // High-end devices
            Region::AsiaPacific => PerformanceProfile::Balanced, // Diverse markets
        }
    }
    
    /// Get default latency target for region
    fn get_default_latency_target(&self, region: Region) -> u32 {
        match region {
            Region::China => 100,          // Network constraints
            Region::MiddleEastAfrica => 100, // Infrastructure
            Region::LatinAmerica => 50,    // Mobile networks
            _ => 20,                       // High-speed networks
        }
    }
    
    /// Get default data residency requirements
    fn get_default_data_residency(&self, region: Region) -> DataResidency {
        match region {
            Region::China => DataResidency::National, // Legal requirement
            Region::Europe => DataResidency::Regional, // GDPR preference
            Region::NorthAmerica => DataResidency::Global, // Flexible
            _ => DataResidency::Regional, // Conservative default
        }
    }
}

impl Default for RegionalManager {
    fn default() -> Self {
        Self::new(Region::NorthAmerica)
    }
}

/// Global deployment utilities
pub struct GlobalDeployment;

impl GlobalDeployment {
    /// Validate configuration for global deployment
    pub fn validate_global_config(configs: &[RegionalConfig]) -> Result<Vec<String>> {
        let mut issues = Vec::new();
        
        // Check coverage of major regions
        let covered_regions: Vec<Region> = configs.iter().map(|c| c.region).collect();
        let required_regions = [
            Region::NorthAmerica,
            Region::Europe,
            Region::AsiaPacific,
        ];
        
        for required in &required_regions {
            if !covered_regions.contains(required) {
                issues.push(format!("Missing configuration for required region: {:?}", required));
            }
        }
        
        // Check compliance consistency
        let mut compliance_levels = Vec::new();
        for config in configs {
            compliance_levels.push(config.privacy_config.frameworks.len());
        }
        
        if compliance_levels.iter().max() != compliance_levels.iter().min() {
            issues.push("Inconsistent compliance levels across regions may indicate gaps".to_string());
        }
        
        Ok(issues)
    }
    
    /// Generate deployment manifest
    pub fn generate_manifest(configs: &[RegionalConfig]) -> String {
        let mut manifest = String::from("# Global Liquid Audio Nets Deployment\n\n");
        
        for config in configs {
            manifest.push_str(&format!("## Region: {}\n", config.region.name()));
            manifest.push_str(&format!("- Code: {}\n", config.region.code()));
            manifest.push_str(&format!("- Languages: {}\n", config.languages.len()));
            manifest.push_str(&format!("- Performance: {:?}\n", config.performance_profile));
            manifest.push_str(&format!("- Data Residency: {:?}\n", config.data_residency));
            manifest.push_str(&format!("- Latency Target: {}ms\n\n", config.latency_target_ms));
        }
        
        manifest
    }
    
    /// Estimate global deployment costs
    pub fn estimate_costs(configs: &[RegionalConfig]) -> DeploymentCostEstimate {
        let total_regions = configs.len();
        let mut high_performance_regions = 0;
        let mut compliance_complexity = 0;
        
        for config in configs {
            if matches!(config.performance_profile, PerformanceProfile::HighPerformance) {
                high_performance_regions += 1;
            }
            compliance_complexity += config.privacy_config.frameworks.len();
        }
        
        DeploymentCostEstimate {
            total_regions,
            high_performance_regions,
            compliance_complexity,
            estimated_monthly_cost_usd: Self::calculate_monthly_cost(
                total_regions,
                high_performance_regions,
                compliance_complexity,
            ),
        }
    }
    
    fn calculate_monthly_cost(
        total_regions: usize,
        high_perf_regions: usize,
        compliance_complexity: usize,
    ) -> f32 {
        // Simplified cost model
        let base_cost_per_region = 100.0; // USD
        let high_perf_premium = 200.0;    // USD
        let compliance_overhead = compliance_complexity as f32 * 50.0; // USD
        
        (total_regions as f32 * base_cost_per_region) +
        (high_perf_regions as f32 * high_perf_premium) +
        compliance_overhead
    }
}

/// Deployment cost estimate
#[derive(Debug, Clone)]
pub struct DeploymentCostEstimate {
    pub total_regions: usize,
    pub high_performance_regions: usize,
    pub compliance_complexity: usize,
    pub estimated_monthly_cost_usd: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_properties() {
        assert_eq!(Region::Europe.code(), "eu");
        assert_eq!(Region::Europe.name(), "Europe");
        assert!(!Region::Europe.primary_languages().is_empty());
        assert!(!Region::Europe.privacy_frameworks().is_empty());
    }

    #[test]
    fn test_performance_profiles() {
        let ultra_low = PerformanceProfile::UltraLowPower;
        let high_perf = PerformanceProfile::HighPerformance;
        
        assert!(ultra_low.power_budget_mw() < high_perf.power_budget_mw());
        assert!(ultra_low.latency_budget_ms() > high_perf.latency_budget_ms());
    }

    #[test]
    fn test_regional_manager() {
        let manager = RegionalManager::new(Region::Europe);
        let config = manager.get_config(Region::Europe).unwrap();
        
        assert_eq!(config.region, Region::Europe);
        assert!(!config.languages.is_empty());
    }

    #[test]
    fn test_model_size_budgets() {
        let micro = ModelSize::Micro;
        let large = ModelSize::Large;
        
        assert!(micro.memory_budget_bytes() < large.memory_budget_bytes());
        assert!(micro.recommended_hidden_dim() < large.recommended_hidden_dim());
    }

    #[test]
    fn test_global_deployment_validation() {
        let configs = vec![
            RegionalConfig {
                region: Region::Europe,
                languages: vec![],
                privacy_config: Default::default(),
                performance_profile: PerformanceProfile::Balanced,
                latency_target_ms: 20,
                data_residency: DataResidency::Regional,
            }
        ];
        
        let issues = GlobalDeployment::validate_global_config(&configs).unwrap();
        assert!(!issues.is_empty()); // Should find missing regions
    }
}