#ifndef PREDIS_CREDENTIAL_MANAGER_INTERFACE_H
#define PREDIS_CREDENTIAL_MANAGER_INTERFACE_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <optional>
#include <functional>
#include <map>

namespace predis {
namespace credentials {

// Credential types supported by Predis
enum class CredentialType {
    DATABASE_PASSWORD,
    API_KEY,
    CERTIFICATE,
    SSH_KEY,
    OAUTH_TOKEN,
    SERVICE_ACCOUNT,
    ENCRYPTION_KEY,
    TLS_CERTIFICATE,
    HMAC_SECRET
};

// Compliance standards for audit
enum class ComplianceStandard {
    SOC2,
    GDPR,
    PCI_DSS,
    HIPAA,
    ISO27001,
    NIST
};

// Access control levels
enum class AccessLevel {
    READ_ONLY,
    READ_WRITE,
    ADMIN,
    ROTATE_ONLY
};

// Credential metadata
struct CredentialMetadata {
    std::string name;
    CredentialType type;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_rotated;
    std::chrono::system_clock::time_point expires_at;
    int version;
    std::vector<std::string> tags;
    std::map<std::string, std::string> labels;
    bool auto_rotate_enabled;
    std::chrono::hours rotation_interval;
};

// Audit log entry
struct AuditLogEntry {
    std::string event_id;
    std::chrono::system_clock::time_point timestamp;
    std::string user_identity;
    std::string action;
    std::string resource;
    bool success;
    std::string ip_address;
    std::map<std::string, std::string> metadata;
};

// Rotation policy
struct RotationPolicy {
    bool enabled;
    std::chrono::hours interval;
    int max_versions_retained;
    bool notify_before_rotation;
    std::chrono::hours notification_lead_time;
    std::vector<std::string> notification_channels;
};

// Encryption configuration
struct EncryptionConfig {
    std::string algorithm;  // AES-256-GCM, ChaCha20-Poly1305
    std::string kms_key_id;
    bool envelope_encryption;
    std::map<std::string, std::string> additional_context;
};

// Base interface for credential management
class ICredentialManager {
public:
    virtual ~ICredentialManager() = default;

    // Core credential operations
    virtual std::optional<std::string> GetCredential(
        const std::string& name,
        const std::string& version = "latest") = 0;
    
    virtual bool StoreCredential(
        const std::string& name,
        const std::string& value,
        const CredentialMetadata& metadata,
        const EncryptionConfig& encryption = {}) = 0;
    
    virtual bool UpdateCredential(
        const std::string& name,
        const std::string& new_value,
        const std::string& reason) = 0;
    
    virtual bool DeleteCredential(
        const std::string& name,
        bool force = false) = 0;
    
    // Rotation operations
    virtual bool RotateCredential(
        const std::string& name,
        std::function<std::string(const std::string&)> rotation_function) = 0;
    
    virtual bool SetRotationPolicy(
        const std::string& name,
        const RotationPolicy& policy) = 0;
    
    virtual std::optional<RotationPolicy> GetRotationPolicy(
        const std::string& name) = 0;
    
    // Access control
    virtual bool GrantAccess(
        const std::string& credential_name,
        const std::string& principal,
        AccessLevel level) = 0;
    
    virtual bool RevokeAccess(
        const std::string& credential_name,
        const std::string& principal) = 0;
    
    virtual std::vector<std::pair<std::string, AccessLevel>> ListAccess(
        const std::string& credential_name) = 0;
    
    // Metadata and discovery
    virtual std::optional<CredentialMetadata> GetMetadata(
        const std::string& name) = 0;
    
    virtual std::vector<std::string> ListCredentials(
        const std::string& prefix = "",
        const std::map<std::string, std::string>& tags = {}) = 0;
    
    virtual std::vector<int> ListVersions(
        const std::string& name) = 0;
    
    // Audit and compliance
    virtual std::vector<AuditLogEntry> GetAuditLog(
        const std::string& credential_name,
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time) = 0;
    
    virtual bool EnableCompliance(
        ComplianceStandard standard,
        const std::map<std::string, std::string>& config) = 0;
    
    virtual std::map<ComplianceStandard, bool> GetComplianceStatus() = 0;
    
    // Health and monitoring
    virtual bool IsHealthy() = 0;
    virtual std::map<std::string, double> GetMetrics() = 0;
    
    // Backup and recovery
    virtual bool BackupCredentials(
        const std::string& backup_location,
        const std::string& encryption_key) = 0;
    
    virtual bool RestoreCredentials(
        const std::string& backup_location,
        const std::string& decryption_key) = 0;
    
    // Break-glass emergency access
    virtual std::string GenerateBreakGlassToken(
        const std::string& credential_name,
        const std::chrono::hours& validity_period,
        const std::string& justification) = 0;
    
    virtual bool ValidateBreakGlassToken(
        const std::string& token) = 0;
};

// Factory for creating credential managers
class CredentialManagerFactory {
public:
    enum class Provider {
        AWS_SECRETS_MANAGER,
        GCP_SECRET_MANAGER,
        AZURE_KEY_VAULT,
        HASHICORP_VAULT,
        LOCAL_ENCRYPTED
    };
    
    static std::unique_ptr<ICredentialManager> Create(
        Provider provider,
        const std::map<std::string, std::string>& config);
};

} // namespace credentials
} // namespace predis

#endif // PREDIS_CREDENTIAL_MANAGER_INTERFACE_H