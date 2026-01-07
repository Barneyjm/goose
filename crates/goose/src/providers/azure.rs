use anyhow::Result;
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use super::api_client::{ApiClient, AuthMethod, AuthProvider};
use super::azureauth::{AuthError, AzureAuth};
use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use super::formats::openai::{create_request, get_usage, response_to_message};
use super::retry::ProviderRetry;
use super::utils::{get_model, handle_response_openai_compat, ImageFormat};
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use crate::providers::utils::RequestLog;
use rmcp::model::Tool;

pub const AZURE_DEFAULT_MODEL: &str = "gpt-4o";
pub const AZURE_DOC_URL: &str =
    "https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models";
pub const AZURE_DEFAULT_API_VERSION: &str = "2024-10-21";
pub const AZURE_OPENAI_KNOWN_MODELS: &[&str] = &["gpt-4o", "gpt-4o-mini", "gpt-4"];

#[derive(Debug)]
pub struct AzureProvider {
    api_client: ApiClient,
    deployment_name: String,
    api_version: String,
    model: ModelConfig,
    name: String,
}

impl Serialize for AzureProvider {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("AzureProvider", 2)?;
        state.serialize_field("deployment_name", &self.deployment_name)?;
        state.serialize_field("api_version", &self.api_version)?;
        state.end()
    }
}

// Custom auth provider that wraps AzureAuth
struct AzureAuthProvider {
    auth: AzureAuth,
}

#[async_trait]
impl AuthProvider for AzureAuthProvider {
    async fn get_auth_header(&self) -> Result<(String, String)> {
        let auth_token = self
            .auth
            .get_token()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get authentication token: {}", e))?;

        match self.auth.credential_type() {
            super::azureauth::AzureCredentials::ApiKey(_) => {
                Ok(("api-key".to_string(), auth_token.token_value))
            }
            super::azureauth::AzureCredentials::DefaultCredential
            | super::azureauth::AzureCredentials::ClientSecret(_)
            | super::azureauth::AzureCredentials::ClientCertificate(_)
            | super::azureauth::AzureCredentials::ManagedIdentity(_) => Ok((
                "Authorization".to_string(),
                format!("Bearer {}", auth_token.token_value),
            )),
        }
    }
}

impl AzureProvider {
    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let endpoint: String = config.get_param("AZURE_OPENAI_ENDPOINT")?;
        let deployment_name: String = config.get_param("AZURE_OPENAI_DEPLOYMENT_NAME")?;
        let api_version: String = config
            .get_param("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| AZURE_DEFAULT_API_VERSION.to_string());

        // Check for various authentication configurations
        let api_key = config
            .get_secret("AZURE_OPENAI_API_KEY")
            .ok()
            .filter(|key: &String| !key.is_empty());
        let tenant_id: Option<String> = config.get_param("AZURE_OPENAI_TENANT_ID").ok();
        let client_id: Option<String> = config.get_param("AZURE_OPENAI_CLIENT_ID").ok();
        let client_secret: Option<String> = config.get_secret("AZURE_OPENAI_CLIENT_SECRET").ok();
        let certificate_path: Option<String> =
            config.get_param("AZURE_OPENAI_CERTIFICATE_PATH").ok();
        let certificate: Option<String> = config.get_secret("AZURE_OPENAI_CERTIFICATE").ok();
        let token_scope: Option<String> = config.get_param("AZURE_OPENAI_TOKEN_SCOPE").ok();
        let use_managed_identity: bool = config
            .get_param::<String>("AZURE_OPENAI_USE_MANAGED_IDENTITY")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        // Determine auth method priority:
        // 1. Managed Identity (if explicitly enabled)
        // 2. Client Certificate (if certificate path or content provided)
        // 3. Client Secret (if secret provided with tenant/client IDs)
        // 4. API Key (if provided)
        // 5. Default Credential (Azure CLI fallback)
        let auth = if use_managed_identity {
            // Use Managed Identity authentication
            let azure_auth = if let Some(id) = &client_id {
                // User-assigned managed identity
                AzureAuth::with_user_assigned_managed_identity(id.clone(), token_scope)
            } else {
                // System-assigned managed identity
                AzureAuth::with_managed_identity(token_scope)
            }
            .map_err(|e| match e {
                AuthError::Credentials(msg) => {
                    anyhow::anyhow!("Managed identity credentials error: {}", msg)
                }
                AuthError::TokenExchange(msg) => {
                    anyhow::anyhow!("Managed identity token exchange error: {}", msg)
                }
            })?;
            azure_auth
        } else if let Some(cert_path) = &certificate_path {
            // Use Client Certificate authentication from file
            let (t_id, c_id) = match (&tenant_id, &client_id) {
                (Some(t), Some(c)) => (t.clone(), c.clone()),
                _ => {
                    return Err(anyhow::anyhow!(
                        "When using certificate authentication, both AZURE_OPENAI_TENANT_ID \
                         and AZURE_OPENAI_CLIENT_ID must be set."
                    ));
                }
            };
            AzureAuth::with_client_certificate_file(t_id, c_id, cert_path, token_scope).map_err(
                |e| match e {
                    AuthError::Credentials(msg) => {
                        anyhow::anyhow!("Certificate credentials error: {}", msg)
                    }
                    AuthError::TokenExchange(msg) => {
                        anyhow::anyhow!("Certificate token exchange error: {}", msg)
                    }
                },
            )?
        } else if let Some(cert_pem) = &certificate {
            // Use Client Certificate authentication from PEM content
            let (t_id, c_id) = match (&tenant_id, &client_id) {
                (Some(t), Some(c)) => (t.clone(), c.clone()),
                _ => {
                    return Err(anyhow::anyhow!(
                        "When using certificate authentication, both AZURE_OPENAI_TENANT_ID \
                         and AZURE_OPENAI_CLIENT_ID must be set."
                    ));
                }
            };
            AzureAuth::with_client_certificate(t_id, c_id, cert_pem.clone(), token_scope).map_err(
                |e| match e {
                    AuthError::Credentials(msg) => {
                        anyhow::anyhow!("Certificate credentials error: {}", msg)
                    }
                    AuthError::TokenExchange(msg) => {
                        anyhow::anyhow!("Certificate token exchange error: {}", msg)
                    }
                },
            )?
        } else if let Some(secret) = &client_secret {
            // Use Client Secret authentication
            let (t_id, c_id) = match (&tenant_id, &client_id) {
                (Some(t), Some(c)) => (t.clone(), c.clone()),
                _ => {
                    return Err(anyhow::anyhow!(
                        "When using client secret authentication, both AZURE_OPENAI_TENANT_ID \
                         and AZURE_OPENAI_CLIENT_ID must be set."
                    ));
                }
            };
            AzureAuth::with_client_secret(t_id, c_id, secret.clone(), token_scope).map_err(
                |e| match e {
                    AuthError::Credentials(msg) => {
                        anyhow::anyhow!("Client secret credentials error: {}", msg)
                    }
                    AuthError::TokenExchange(msg) => {
                        anyhow::anyhow!("Client secret token exchange error: {}", msg)
                    }
                },
            )?
        } else {
            // Use API Key or Default Credential (Azure CLI)
            AzureAuth::new(api_key).map_err(|e| match e {
                AuthError::Credentials(msg) => anyhow::anyhow!("Credentials error: {}", msg),
                AuthError::TokenExchange(msg) => anyhow::anyhow!("Token exchange error: {}", msg),
            })?
        };

        let auth_provider = AzureAuthProvider { auth };
        let api_client = ApiClient::new(endpoint, AuthMethod::Custom(Box::new(auth_provider)))?;

        Ok(Self {
            api_client,
            deployment_name,
            api_version,
            model,
            name: Self::metadata().name,
        })
    }

    async fn post(&self, payload: &Value) -> Result<Value, ProviderError> {
        // Build the path for Azure OpenAI
        let path = format!(
            "openai/deployments/{}/chat/completions?api-version={}",
            self.deployment_name, self.api_version
        );

        let response = self.api_client.response_post(&path, payload).await?;
        handle_response_openai_compat(response).await
    }
}

#[async_trait]
impl Provider for AzureProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "azure_openai",
            "Azure OpenAI",
            "Models through Azure OpenAI Service. Supports API key, client secret, \
             client certificate, managed identity, or Azure CLI authentication.",
            "gpt-4o",
            AZURE_OPENAI_KNOWN_MODELS.to_vec(),
            AZURE_DOC_URL,
            vec![
                // Required configuration
                ConfigKey::new("AZURE_OPENAI_ENDPOINT", true, false, None),
                ConfigKey::new("AZURE_OPENAI_DEPLOYMENT_NAME", true, false, None),
                ConfigKey::new("AZURE_OPENAI_API_VERSION", false, false, Some("2024-10-21")),
                // API key auth (optional - falls back to Azure CLI if not provided)
                ConfigKey::new("AZURE_OPENAI_API_KEY", false, true, None),
                // Service principal auth (tenant + client ID required for these)
                ConfigKey::new("AZURE_OPENAI_TENANT_ID", false, false, None),
                ConfigKey::new("AZURE_OPENAI_CLIENT_ID", false, false, None),
                // Client secret auth
                ConfigKey::new("AZURE_OPENAI_CLIENT_SECRET", false, true, None),
                // Client certificate auth
                ConfigKey::new("AZURE_OPENAI_CERTIFICATE_PATH", false, false, None),
                ConfigKey::new("AZURE_OPENAI_CERTIFICATE", false, true, None),
                // Managed identity auth
                ConfigKey::new("AZURE_OPENAI_USE_MANAGED_IDENTITY", false, false, None),
                // Token scope (applies to all Entra auth methods)
                ConfigKey::new("AZURE_OPENAI_TOKEN_SCOPE", false, false, None),
            ],
        )
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, model_config, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete_with_model(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let payload = create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            false,
        )?;
        let response = self
            .with_retry(|| async {
                let payload_clone = payload.clone();
                self.post(&payload_clone).await
            })
            .await?;

        let message = response_to_message(&response)?;
        let usage = response.get("usage").map(get_usage).unwrap_or_else(|| {
            tracing::debug!("Failed to get usage data");
            Usage::default()
        });
        let response_model = get_model(&response);
        let mut log = RequestLog::start(model_config, &payload)?;
        log.write(&response, Some(&usage))?;
        Ok((message, ProviderUsage::new(response_model, usage)))
    }
}
