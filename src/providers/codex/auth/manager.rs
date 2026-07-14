use std::sync::Arc;
use std::sync::RwLock;
#[cfg(test)]
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex as AsyncMutex;

use super::constants::{CLIENT_ID, ISSUER, REFRESH_MARGIN_MS};
use super::jwt::{TokenResponse, extract_account_id, validate_token_response};
use super::token_store::{CodexTokenStore, StoredAuth};
use crate::auth::AuthStorage;

// How long a keychain read is trusted before the durable store is consulted
// again. On macOS every `load_auth` spawns a `/usr/bin/security` child to read
// the Keychain; without this cache each incoming request spawns one, so a burst
// of concurrent requests exhausts the process file-descriptor limit (EMFILE),
// fails auth with a spurious 401, and can take down the listener. Holding the
// value for a few seconds collapses a concurrent burst onto a single read while
// still letting an out-of-process re-login be observed shortly after.
const AUTH_CACHE_TTL_MS: u64 = 5_000;

pub struct CodexAuthManager<S: AuthStorage<StoredAuth>> {
    pub store: CodexTokenStore<S>,
    #[cfg(test)]
    test_auth: Arc<Mutex<Option<StoredAuth>>>,
    refresh_lock: Arc<AsyncMutex<()>>,
    // Value plus the timestamp (ms) it was read from the durable store.
    auth_cache: RwLock<Option<(StoredAuth, u64)>>,
    cache_ttl_ms: u64,
    refresh_client: reqwest::Client,
    token_endpoint: String,
}

impl<S: AuthStorage<StoredAuth>> CodexAuthManager<S> {
    pub fn new(store: CodexTokenStore<S>) -> Self {
        Self::new_with_config(store, format!("{ISSUER}/oauth/token"), AUTH_CACHE_TTL_MS)
    }

    #[cfg(test)]
    fn new_with_token_endpoint(store: CodexTokenStore<S>, token_endpoint: String) -> Self {
        Self::new_with_config(store, token_endpoint, AUTH_CACHE_TTL_MS)
    }

    fn new_with_config(
        store: CodexTokenStore<S>,
        token_endpoint: String,
        cache_ttl_ms: u64,
    ) -> Self {
        Self {
            store,
            #[cfg(test)]
            test_auth: Arc::new(Mutex::new(None)),
            refresh_lock: Arc::new(AsyncMutex::new(())),
            auth_cache: RwLock::new(None),
            cache_ttl_ms,
            refresh_client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .timeout(Duration::from_secs(30))
                .build()
                .expect("failed to create Codex OAuth refresh client"),
            token_endpoint,
        }
    }

    fn cache_store(&self, auth: StoredAuth) {
        if let Ok(mut guard) = self.auth_cache.write() {
            *guard = Some((auth, Self::now_ms()));
        }
    }

    fn cache_clear(&self) {
        if let Ok(mut guard) = self.auth_cache.write() {
            *guard = None;
        }
    }

    fn cache_get(&self) -> Option<StoredAuth> {
        if self.cache_ttl_ms == 0 {
            return None;
        }
        let guard = self.auth_cache.read().ok()?;
        let (auth, loaded_at) = guard.as_ref()?;
        (Self::now_ms().saturating_sub(*loaded_at) < self.cache_ttl_ms).then(|| auth.clone())
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub async fn get_auth(&self) -> Result<StoredAuth, anyhow::Error> {
        let stored = self.load_auth()?.ok_or_else(|| {
            anyhow::anyhow!("Not authenticated. Run: claude-code-proxy codex auth login")
        })?;

        if stored.expires > Self::now_ms() + REFRESH_MARGIN_MS {
            return Ok(stored);
        }

        self.refresh(false, None).await
    }

    pub async fn force_refresh(&self, rejected_access: &str) -> Result<StoredAuth, anyhow::Error> {
        self.refresh(true, Some(rejected_access)).await
    }

    fn load_auth(&self) -> Result<Option<StoredAuth>, anyhow::Error> {
        #[cfg(test)]
        if let Some(auth) = self
            .test_auth
            .lock()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .clone()
        {
            return Ok(Some(auth));
        }

        // Serve a recent keychain read from memory so a concurrent request burst
        // does not spawn one `/usr/bin/security` child per request (see
        // AUTH_CACHE_TTL_MS). The cache is write-through on our own rotations, so
        // freshly refreshed tokens are visible immediately regardless of TTL.
        if let Some(auth) = self.cache_get() {
            return Ok(Some(auth));
        }

        let loaded = self.store.load_auth()?;
        match &loaded {
            Some(auth) => self.cache_store(auth.clone()),
            None => self.cache_clear(),
        }
        Ok(loaded)
    }

    async fn refresh(
        &self,
        force: bool,
        rejected_access: Option<&str>,
    ) -> Result<StoredAuth, anyhow::Error> {
        let _refresh_guard = self.refresh_lock.lock().await;

        // Reload from durable storage after acquiring the single-flight lock.
        // Another request may have rotated and persisted the token while this
        // caller was waiting.
        let current = self
            .load_auth()?
            .ok_or_else(|| anyhow::anyhow!("Not authenticated"))?;

        if (!force && current.expires > Self::now_ms() + REFRESH_MARGIN_MS)
            || rejected_access.is_some_and(|access| current.access != access)
        {
            return Ok(current);
        }

        self.refresh_now(&current).await
    }

    async fn refresh_now(&self, current: &StoredAuth) -> Result<StoredAuth, anyhow::Error> {
        if current.refresh.is_empty() {
            anyhow::bail!("No refresh token stored; re-authenticate");
        }

        let form = [
            ("client_id", CLIENT_ID.to_string()),
            ("grant_type", "refresh_token".to_string()),
            ("refresh_token", current.refresh.clone()),
        ];

        let resp = self
            .refresh_client
            .post(&self.token_endpoint)
            .form(&form)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("refresh network error: {e}"))?;

        let status = resp.status().as_u16();
        if status == 401 || status == 403 {
            // Read the durable store directly (not the cache): another actor may
            // have rotated the token out of band, and that is exactly the case
            // this recovery path exists to detect.
            if let Some(latest) = self.store.load_auth()?
                && latest != *current
            {
                self.cache_store(latest.clone());
                return Ok(latest);
            }
            self.store.clear_auth()?;
            self.cache_clear();
            let err_msg = resp
                .text()
                .await
                .unwrap_or_else(|_| "Token refresh unauthorized".to_string());
            anyhow::bail!("{err_msg}");
        }

        if !resp.status().is_success() {
            anyhow::bail!("Token refresh failed: {status}");
        }

        let tokens: TokenResponse = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse token response: {e}"))?;
        validate_token_response(&tokens)?;
        let account_id = extract_account_id(&tokens).or_else(|| current.account_id.clone());
        let expires = Self::now_ms() + (tokens.expires_in.unwrap_or(3600) * 1000);
        let next = StoredAuth {
            access: tokens.access_token,
            refresh: tokens.refresh_token,
            expires,
            account_id,
        };
        self.store.save_auth(next.clone())?;
        self.cache_store(next.clone());
        Ok(next)
    }

    pub fn persist_initial_tokens(
        &self,
        tokens: &TokenResponse,
    ) -> Result<StoredAuth, anyhow::Error> {
        validate_token_response(tokens)?;
        let account_id = extract_account_id(tokens);
        let expires = Self::now_ms() + (tokens.expires_in.unwrap_or(3600) * 1000);
        let auth = StoredAuth {
            access: tokens.access_token.clone(),
            refresh: tokens.refresh_token.clone(),
            expires,
            account_id,
        };
        self.store.save_auth(auth.clone())?;
        self.cache_store(auth.clone());
        Ok(auth)
    }

    #[cfg(test)]
    pub fn set_test_auth(&self, auth: StoredAuth) {
        if let Ok(mut guard) = self.test_auth.lock() {
            *guard = Some(auth);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::InMemoryAuthStore;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    fn test_store() -> CodexTokenStore<InMemoryAuthStore<StoredAuth>> {
        CodexTokenStore::new(InMemoryAuthStore::new())
    }

    #[tokio::test]
    async fn get_auth_returns_stored() {
        let store = test_store();
        let auth = StoredAuth {
            access: "test_access".into(),
            refresh: "test_refresh".into(),
            expires: 9999999999999,
            account_id: Some("acct_1".into()),
        };
        store.save_auth(auth.clone()).unwrap();
        let manager = CodexAuthManager::new(store);
        let result = manager.get_auth().await.unwrap();
        assert_eq!(result.access, "test_access");
        assert_eq!(result.account_id.as_deref(), Some("acct_1"));
    }

    #[tokio::test]
    async fn get_auth_fails_when_no_auth() {
        let store = test_store();
        let manager = CodexAuthManager::new(store);
        assert!(manager.get_auth().await.is_err());
        assert!(
            manager
                .get_auth()
                .await
                .unwrap_err()
                .to_string()
                .contains("Not authenticated")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_expired_auth_refreshes_once() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let refreshes = Arc::new(AtomicUsize::new(0));
        let server_refreshes = refreshes.clone();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let read = stream.read(&mut request).unwrap();
            assert!(read > 0);
            assert!(String::from_utf8_lossy(&request[..read]).contains("refresh_token=stale"));
            server_refreshes.fetch_add(1, Ordering::SeqCst);

            let body = br#"{"access_token":"rotated","refresh_token":"rotated-refresh","expires_in":3600}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });

        let store = test_store();
        store
            .save_auth(StoredAuth {
                access: "expired".into(),
                refresh: "stale".into(),
                expires: 0,
                account_id: Some("acct_1".into()),
            })
            .unwrap();
        let manager = Arc::new(CodexAuthManager::new_with_token_endpoint(
            store,
            format!("http://{addr}/oauth/token"),
        ));
        let (first, second) = tokio::join!(manager.get_auth(), manager.get_auth());
        let results = [first.unwrap(), second.unwrap()];
        server.join().unwrap();

        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        assert!(results.iter().all(|auth| auth.access == "rotated"));
        assert!(results.iter().all(|auth| auth.refresh == "rotated-refresh"));
    }

    #[tokio::test]
    async fn stale_401_reuses_already_rotated_auth() {
        let store = test_store();
        store
            .save_auth(StoredAuth {
                access: "rotated".into(),
                refresh: "rotated-refresh".into(),
                expires: u64::MAX,
                account_id: Some("acct_1".into()),
            })
            .unwrap();
        let manager = CodexAuthManager::new_with_token_endpoint(
            store,
            "http://127.0.0.1:1/should-not-be-called".into(),
        );

        let auth = manager.force_refresh("rejected").await.unwrap();
        assert_eq!(auth.access, "rotated");
        assert_eq!(auth.refresh, "rotated-refresh");
    }

    #[tokio::test]
    async fn unauthorized_refresh_preserves_changed_refresh_token() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let backing = InMemoryAuthStore::new();
        let server_backing = backing.clone();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            assert!(stream.read(&mut request).unwrap() > 0);
            server_backing
                .save(StoredAuth {
                    access: "same-access".into(),
                    refresh: "replacement-refresh".into(),
                    expires: u64::MAX,
                    account_id: Some("acct_1".into()),
                })
                .unwrap();
            let body = b"rejected refresh token";
            let response = format!(
                "HTTP/1.1 401 Unauthorized\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });

        let store = CodexTokenStore::new(backing);
        store
            .save_auth(StoredAuth {
                access: "same-access".into(),
                refresh: "rejected-refresh".into(),
                expires: 0,
                account_id: Some("acct_1".into()),
            })
            .unwrap();
        let manager =
            CodexAuthManager::new_with_token_endpoint(store, format!("http://{addr}/oauth/token"));

        let auth = manager.get_auth().await.unwrap();
        server.join().unwrap();
        assert_eq!(auth.access, "same-access");
        assert_eq!(auth.refresh, "replacement-refresh");
        assert_eq!(manager.store.load_auth().unwrap(), Some(auth));
    }

    #[tokio::test]
    async fn durable_rotation_and_logout_are_observed_by_shared_manager() {
        let store = test_store();
        store
            .save_auth(StoredAuth {
                access: "first".into(),
                refresh: "first-refresh".into(),
                expires: u64::MAX,
                account_id: Some("acct_1".into()),
            })
            .unwrap();
        // TTL 0 disables the in-memory cache so this test can assert that
        // out-of-band durable writes are observed on the very next read.
        let manager =
            CodexAuthManager::new_with_config(store, format!("{ISSUER}/oauth/token"), 0);
        assert_eq!(manager.get_auth().await.unwrap().access, "first");

        manager
            .store
            .save_auth(StoredAuth {
                access: "rotated".into(),
                refresh: "rotated-refresh".into(),
                expires: u64::MAX,
                account_id: Some("acct_2".into()),
            })
            .unwrap();
        let rotated = manager.get_auth().await.unwrap();
        assert_eq!(rotated.access, "rotated");
        assert_eq!(rotated.account_id.as_deref(), Some("acct_2"));

        manager.store.clear_auth().unwrap();
        assert!(manager.get_auth().await.is_err());
    }

    #[tokio::test]
    async fn valid_auth_is_served_from_cache_within_ttl() {
        // With the default TTL the manager must reuse the first read instead of
        // hitting the durable store (a `/usr/bin/security` spawn on macOS) again.
        // We prove the cache is live by rotating the store out of band and
        // confirming the cached value is still returned within the TTL window.
        let store = test_store();
        store
            .save_auth(StoredAuth {
                access: "cached".into(),
                refresh: "cached-refresh".into(),
                expires: u64::MAX,
                account_id: Some("acct_1".into()),
            })
            .unwrap();
        let manager = CodexAuthManager::new(store);
        assert_eq!(manager.get_auth().await.unwrap().access, "cached");

        // Out-of-band durable rotation is intentionally NOT observed within the
        // TTL; the running proxy is the only writer and updates the cache
        // write-through on its own rotations.
        manager
            .store
            .save_auth(StoredAuth {
                access: "rotated-out-of-band".into(),
                refresh: "r".into(),
                expires: u64::MAX,
                account_id: Some("acct_1".into()),
            })
            .unwrap();
        assert_eq!(manager.get_auth().await.unwrap().access, "cached");
    }
}
