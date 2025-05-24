#ifndef PREDIS_AWS_SECRETS_MANAGER_H
#define PREDIS_AWS_SECRETS_MANAGER_H

#include "credential_manager_interface.h"
#include <aws/core/Aws.h>
#include <aws/secretsmanager/SecretsManagerClient.h>
#include <aws/sts/STSClient.h>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>

namespace predis {
namespace credentials {

// Configuration for AWS Secrets Manager
struct AWSSecretsManagerConfig {
    std::string region = "us-east-1";
    std::string role_arn;  // For cross-account access
    std::string external_id;  // For secure cross-account
    std::string kms_key_id;  // Customer managed KMS key
    bool use_vpc_endpoint = false;
    std::string vpc_endpoint_url;
    int max_retries = 3;
    std::chrono::milliseconds retry_delay{1000};
    bool enable_caching = true;
    std::chrono::minutes cache_ttl{5};
    bool enable_metrics = true;
    std::string metrics_namespace = "Predis/Credentials";
};

// Cache entry for credentials
struct CacheEntry {
    std::string value;
    std::chrono::system_clock::time_point expiry;
    int version;
    bool is_valid() const {
        return std::chrono::system_clock::now() < expiry;
    }
};

// AWS Secrets Manager implementation
class AWSSecretsManager : public ICredentialManager {
public:
    explicit AWSSecretsManager(const AWSSecretsManagerConfig& config);
    ~AWSSecretsManager();

    // Core operations
    std::optional<std::string> GetCredential(
        const std::string& name,
        const std::string& version = "latest") override;
    
    bool StoreCredential(
        const std::string& name,
        const std::string& value,
        const CredentialMetadata& metadata,
        const EncryptionConfig& encryption = {}) override;
    
    bool UpdateCredential(
        const std::string& name,
        const std::string& new_value,
        const std::string& reason) override;
    
    bool DeleteCredential(
        const std::string& name,
        bool force = false) override;
    
    // Rotation
    bool RotateCredential(
        const std::string& name,
        std::function<std::string(const std::string&)> rotation_function) override;
    
    bool SetRotationPolicy(
        const std::string& name,
        const RotationPolicy& policy) override;
    
    std::optional<RotationPolicy> GetRotationPolicy(
        const std::string& name) override;
    
    // Access control
    bool GrantAccess(
        const std::string& credential_name,
        const std::string& principal,
        AccessLevel level) override;
    
    bool RevokeAccess(
        const std::string& credential_name,
        const std::string& principal) override;
    
    std::vector<std::pair<std::string, AccessLevel>> ListAccess(
        const std::string& credential_name) override;
    
    // Metadata
    std::optional<CredentialMetadata> GetMetadata(
        const std::string& name) override;
    
    std::vector<std::string> ListCredentials(
        const std::string& prefix = "",
        const std::map<std::string, std::string>& tags = {}) override;
    
    std::vector<int> ListVersions(
        const std::string& name) override;
    
    // Audit
    std::vector<AuditLogEntry> GetAuditLog(
        const std::string& credential_name,
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time) override;
    
    bool EnableCompliance(
        ComplianceStandard standard,
        const std::map<std::string, std::string>& config) override;
    
    std::map<ComplianceStandard, bool> GetComplianceStatus() override;
    
    // Health
    bool IsHealthy() override;
    std::map<std::string, double> GetMetrics() override;
    
    // Backup
    bool BackupCredentials(
        const std::string& backup_location,
        const std::string& encryption_key) override;
    
    bool RestoreCredentials(
        const std::string& backup_location,
        const std::string& decryption_key) override;
    
    // Break-glass
    std::string GenerateBreakGlassToken(
        const std::string& credential_name,
        const std::chrono::hours& validity_period,
        const std::string& justification) override;
    
    bool ValidateBreakGlassToken(
        const std::string& token) override;

private:
    // AWS SDK components
    std::unique_ptr<Aws::SecretsManager::SecretsManagerClient> secrets_client_;
    std::unique_ptr<Aws::STS::STSClient> sts_client_;
    
    // Configuration
    AWSSecretsManagerConfig config_;
    
    // Caching
    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, CacheEntry> cache_;
    
    // Metrics
    struct Metrics {
        std::atomic<uint64_t> get_requests{0};
        std::atomic<uint64_t> get_errors{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
        std::atomic<uint64_t> store_requests{0};
        std::atomic<uint64_t> store_errors{0};
        std::atomic<uint64_t> rotation_requests{0};
        std::atomic<uint64_t> rotation_errors{0};
        std::chrono::system_clock::time_point start_time;
    };
    Metrics metrics_;
    
    // Rotation management
    std::mutex rotation_mutex_;
    std::unordered_map<std::string, RotationPolicy> rotation_policies_;
    std::thread rotation_worker_;
    std::condition_variable rotation_cv_;
    std::atomic<bool> rotation_stop_{false};
    std::queue<std::string> rotation_queue_;
    
    // Compliance tracking
    std::mutex compliance_mutex_;
    std::map<ComplianceStandard, bool> compliance_status_;
    std::map<ComplianceStandard, std::map<std::string, std::string>> compliance_configs_;
    
    // Helper methods
    void InitializeAWSClients();
    void StartRotationWorker();
    void StopRotationWorker();
    void RotationWorkerLoop();
    
    std::string BuildSecretArn(const std::string& name) const;
    std::string GenerateResourcePolicy(const std::string& principal, AccessLevel level) const;
    
    bool ValidateCredentialName(const std::string& name) const;
    bool ValidateEncryption(const EncryptionConfig& encryption) const;
    
    void RecordMetric(const std::string& metric_name, double value);
    void LogAuditEvent(const std::string& action, const std::string& resource, 
                      bool success, const std::map<std::string, std::string>& metadata);
    
    std::optional<CacheEntry> GetFromCache(const std::string& name) const;
    void PutInCache(const std::string& name, const std::string& value, int version);
    void InvalidateCache(const std::string& name);
    
    // AWS SDK helpers
    bool AssumeRole();
    void RefreshCredentials();
    
    // Compliance implementations
    void EnableSOC2Compliance(const std::map<std::string, std::string>& config);
    void EnableGDPRCompliance(const std::map<std::string, std::string>& config);
    void EnablePCIDSSCompliance(const std::map<std::string, std::string>& config);
    void EnableHIPAACompliance(const std::map<std::string, std::string>& config);
};

} // namespace credentials
} // namespace predis

#endif // PREDIS_AWS_SECRETS_MANAGER_H