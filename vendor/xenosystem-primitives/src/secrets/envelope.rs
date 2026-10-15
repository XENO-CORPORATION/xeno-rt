use super::error::{Result, SecretsError, SecretsErrorCode as C};
use crate::dirs::codec::{base64_decode, base64_encode, hex_encode, sha256};
use crate::dirs::json::{self, Value};
use aes_gcm::aead::{generic_array::GenericArray, Aead, Payload};
use aes_gcm::{Aes256Gcm, KeyInit};

pub const KEY_BYTES: usize = 32;
const NONCE_BYTES: usize = 12;
const TAG_BYTES: usize = 16;

pub fn wrap_aad(area: &str, name: &str) -> Vec<u8> {
    format!("xeno-secrets:v1:wrap:{area}:{name}").into_bytes()
}
pub fn data_aad(area: &str, name: &str) -> Vec<u8> {
    format!("xeno-secrets:v1:data:{area}:{name}").into_bytes()
}

/// Identifier of a master key: first 8 bytes of its SHA-256, hex. Not secret.
pub fn keyid_of(master: &[u8]) -> String {
    hex_encode(&sha256(master)[..8])
}

/// The on-disk envelope; the wire format is shared with the TypeScript implementation and pinned by
/// contract/vectors/secrets-envelope.json. `*_ct` is base64(ciphertext || 16-byte tag).
#[derive(Debug, Clone, PartialEq)]
pub struct EnvelopeFile {
    pub area: String,
    pub name: String,
    pub key_id: String,
    pub wrap_nonce: String,
    pub wrap_ct: String,
    pub data_nonce: String,
    pub data_ct: String,
    pub created_at: String,
}

impl EnvelopeFile {
    pub fn to_json(&self) -> String {
        let s = |x: &str| Value::Str(x.to_string());
        json::to_string(&Value::Obj(vec![
            ("v".into(), Value::Num(1.0)),
            ("alg".into(), s("AES-256-GCM")),
            ("area".into(), s(&self.area)),
            ("name".into(), s(&self.name)),
            ("keyId".into(), s(&self.key_id)),
            ("wrap".into(), Value::Obj(vec![("nonce".into(), s(&self.wrap_nonce)), ("ct".into(), s(&self.wrap_ct))])),
            ("data".into(), Value::Obj(vec![("nonce".into(), s(&self.data_nonce)), ("ct".into(), s(&self.data_ct))])),
            ("createdAt".into(), s(&self.created_at)),
        ]))
    }

    /// Parses and shape-checks an envelope. Never echoes file content.
    pub fn parse(text: &str) -> Result<EnvelopeFile> {
        let bad = || SecretsError::new(C::Malformed, "the secret file is not a version 1 envelope");
        let v = json::parse(text).map_err(|_| SecretsError::new(C::Malformed, "the secret file is not valid JSON"))?;
        let get = |o: &Value, k: &str| o.get(k).and_then(Value::as_str).map(String::from);
        if v.get("v").and_then(Value::as_i64) != Some(1) || get(&v, "alg").as_deref() != Some("AES-256-GCM") {
            return Err(bad());
        }
        let (wrap, data) = (v.get("wrap").ok_or_else(bad)?, v.get("data").ok_or_else(bad)?);
        Ok(EnvelopeFile {
            area: get(&v, "area").ok_or_else(bad)?,
            name: get(&v, "name").ok_or_else(bad)?,
            key_id: get(&v, "keyId").ok_or_else(bad)?,
            wrap_nonce: get(wrap, "nonce").ok_or_else(bad)?,
            wrap_ct: get(wrap, "ct").ok_or_else(bad)?,
            data_nonce: get(data, "nonce").ok_or_else(bad)?,
            data_ct: get(data, "ct").ok_or_else(bad)?,
            created_at: get(&v, "createdAt").unwrap_or_default(),
        })
    }
}

fn seal(key: &[u8], nonce: &[u8], pt: &[u8], aad: &[u8]) -> Vec<u8> {
    let c = Aes256Gcm::new_from_slice(key).expect("32-byte key");
    c.encrypt(GenericArray::from_slice(nonce), Payload { msg: pt, aad }).expect("AES-GCM encryption does not fail for in-memory input")
}

fn open(key: &[u8], nonce: &[u8], sealed: &[u8], aad: &[u8]) -> Option<Vec<u8>> {
    if sealed.len() < TAG_BYTES || nonce.len() != NONCE_BYTES || key.len() != KEY_BYTES {
        return None;
    }
    let c = Aes256Gcm::new_from_slice(key).ok()?;
    c.decrypt(GenericArray::from_slice(nonce), Payload { msg: sealed, aad }).ok()
}

#[derive(Default)]
pub struct SealOptions {
    /// Fixed randomness for known-answer tests only.
    pub dek: Option<[u8; KEY_BYTES]>,
    pub wrap_nonce: Option<[u8; NONCE_BYTES]>,
    pub data_nonce: Option<[u8; NONCE_BYTES]>,
    pub created_at: Option<String>,
}

pub(crate) fn random<const N: usize>() -> Result<[u8; N]> {
    let mut b = [0u8; N];
    getrandom::getrandom(&mut b).map_err(|_| SecretsError::new(C::KeyUnavailable, "the operating system random source failed"))?;
    Ok(b)
}

pub fn seal_secret(master: &[u8], area: &str, name: &str, plaintext: &[u8], opts: SealOptions) -> Result<EnvelopeFile> {
    if master.len() != KEY_BYTES {
        return Err(SecretsError::new(C::InvalidArgument, "the master key must be 32 bytes"));
    }
    let mut dek = match opts.dek {
        Some(d) => d,
        None => random::<KEY_BYTES>()?,
    };
    let wrap_nonce = match opts.wrap_nonce {
        Some(n) => n,
        None => random::<NONCE_BYTES>()?,
    };
    let data_nonce = match opts.data_nonce {
        Some(n) => n,
        None => random::<NONCE_BYTES>()?,
    };
    let file = EnvelopeFile {
        area: area.into(),
        name: name.into(),
        key_id: keyid_of(master),
        wrap_nonce: base64_encode(&wrap_nonce),
        wrap_ct: base64_encode(&seal(master, &wrap_nonce, &dek, &wrap_aad(area, name))),
        data_nonce: base64_encode(&data_nonce),
        data_ct: base64_encode(&seal(&dek, &data_nonce, plaintext, &data_aad(area, name))),
        created_at: opts.created_at.unwrap_or_else(now_iso),
    };
    dek.fill(0);
    Ok(file)
}

pub fn open_secret(master: &[u8], file: &EnvelopeFile, area: &str, name: &str) -> Result<Vec<u8>> {
    if file.area != area || file.name != name {
        return Err(SecretsError::new(C::DecryptFailed, format!("the secret file does not belong at {area}/{name}")));
    }
    let b64 = |s: &str| base64_decode(s).ok_or_else(|| SecretsError::new(C::Malformed, "the secret file holds invalid base64"));
    let mut dek = match open(master, &b64(&file.wrap_nonce)?, &b64(&file.wrap_ct)?, &wrap_aad(area, name)) {
        Some(d) => d,
        None => {
            return Err(if keyid_of(master) != file.key_id {
                SecretsError::new(C::KeyMismatch, format!("{area}/{name} was sealed with a different master key"))
            } else {
                SecretsError::new(C::DecryptFailed, format!("{area}/{name} failed authentication (damaged or tampered)"))
            })
        }
    };
    let pt = open(&dek, &b64(&file.data_nonce)?, &b64(&file.data_ct)?, &data_aad(area, name));
    dek.fill(0);
    pt.ok_or_else(|| SecretsError::new(C::DecryptFailed, format!("{area}/{name} failed authentication (damaged or tampered)")))
}

fn now_iso() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z", rem / 3600, rem % 3600 / 60, rem % 60, d.subsec_millis())
}
