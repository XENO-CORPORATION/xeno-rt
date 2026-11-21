//! Read one named initializer (weight tensor) out of an ONNX model file.
//!
//! Needed for the v2 export, whose `embed_tokens` graph has no
//! `text_conditioning` input: the CFG uncond row must have the TEXT embedding
//! removed while position + emotion are kept (reference: `text_emb[1].zero_()`),
//! and the embedding is additive, so subtracting the model's own
//! `text_emb.weight` rows reproduces the reference exactly. Reading it from the
//! model itself avoids a separately provisioned sidecar that could drift.
//!
//! A minimal protobuf walk: ModelProto.graph(7) → GraphProto.initializer(5) →
//! TensorProto { dims(1), data_type(2), name(8), raw_data(9),
//! external_data(13) {key(1), value(2)}, data_location(14) }.

use std::path::Path;

use crate::AudioError;

const FLOAT: u64 = 1;

pub struct Initializer {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn read_f32(model: &Path, name: &str) -> Result<Initializer, AudioError> {
    let bad = |m: String| AudioError::Inference(format!("{}: {m}", model.display()));
    let bytes = std::fs::read(model).map_err(|e| bad(e.to_string()))?;
    let graph = fields(&bytes)
        .map_err(bad)?
        .into_iter()
        .find(|(f, _)| *f == 7)
        .and_then(|(_, v)| v.bytes())
        .ok_or_else(|| bad("no graph".into()))?;
    for (f, v) in fields(graph).map_err(bad)? {
        if f != 5 {
            continue;
        }
        let Some(t) = v.bytes() else { continue };
        let tf = fields(t).map_err(bad)?;
        let tname = tf
            .iter()
            .find(|(f, _)| *f == 8)
            .and_then(|(_, v)| v.bytes());
        if tname != Some(name.as_bytes()) {
            continue;
        }
        let mut dims = Vec::new();
        for (f, v) in &tf {
            if *f == 1 {
                match v {
                    Val::Int(d) => dims.push(*d as usize),
                    Val::Bytes(packed) => {
                        let mut p = *packed;
                        while !p.is_empty() {
                            let (d, rest) = varint(p).map_err(bad)?;
                            dims.push(d as usize);
                            p = rest;
                        }
                    }
                }
            }
        }
        let dtype = tf
            .iter()
            .find(|(f, _)| *f == 2)
            .and_then(|(_, v)| v.int())
            .unwrap_or(0);
        if dtype != FLOAT {
            return Err(bad(format!(
                "`{name}` is data_type {dtype}, expected float32"
            )));
        }
        let count: usize = dims.iter().product();
        let raw: Vec<u8> = if let Some(r) = tf
            .iter()
            .find(|(f, _)| *f == 9)
            .and_then(|(_, v)| v.bytes())
        {
            r.to_vec()
        } else {
            let (mut loc, mut off, mut len) = (None, 0u64, None);
            for (f, v) in &tf {
                if *f != 13 {
                    continue;
                }
                let Some(kv) = v.bytes() else { continue };
                let kv = fields(kv).map_err(bad)?;
                let k = kv
                    .iter()
                    .find(|(f, _)| *f == 1)
                    .and_then(|(_, v)| v.bytes())
                    .unwrap_or(b"");
                let val = kv
                    .iter()
                    .find(|(f, _)| *f == 2)
                    .and_then(|(_, v)| v.bytes())
                    .unwrap_or(b"");
                let val = std::str::from_utf8(val).map_err(|e| bad(e.to_string()))?;
                match k {
                    b"location" => loc = Some(val.to_string()),
                    b"offset" => off = val.parse().map_err(|_| bad("bad offset".into()))?,
                    b"length" => {
                        len = Some(val.parse::<u64>().map_err(|_| bad("bad length".into()))?)
                    }
                    _ => {}
                }
            }
            let loc =
                loc.ok_or_else(|| bad(format!("`{name}` has no data and no external location")))?;
            // Refuse a location that escapes the model directory.
            if loc.contains("..") || Path::new(&loc).is_absolute() {
                return Err(bad(format!("unsafe external data location `{loc}`")));
            }
            let path = model.parent().unwrap_or(Path::new(".")).join(&loc);
            let want = len.unwrap_or((count * 4) as u64);
            use std::io::{Read, Seek, SeekFrom};
            let mut file =
                std::fs::File::open(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
            file.seek(SeekFrom::Start(off))
                .map_err(|e| bad(e.to_string()))?;
            let mut buf = vec![0u8; want as usize];
            file.read_exact(&mut buf)
                .map_err(|e| bad(format!("reading `{name}`: {e}")))?;
            buf
        };
        if raw.len() != count * 4 {
            return Err(bad(format!(
                "`{name}` holds {} bytes for dims {dims:?}",
                raw.len()
            )));
        }
        let data = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        return Ok(Initializer { dims, data });
    }
    Err(bad(format!("initializer `{name}` not found")))
}

enum Val<'a> {
    Int(u64),
    Bytes(&'a [u8]),
}

impl<'a> Val<'a> {
    fn bytes(&self) -> Option<&'a [u8]> {
        match self {
            Val::Bytes(b) => Some(b),
            Val::Int(_) => None,
        }
    }
    fn int(&self) -> Option<u64> {
        match self {
            Val::Int(i) => Some(*i),
            Val::Bytes(_) => None,
        }
    }
}

fn varint(mut b: &[u8]) -> Result<(u64, &[u8]), String> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let (&byte, rest) = b.split_first().ok_or("truncated varint")?;
        b = rest;
        v |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((v, b));
        }
    }
    Err("varint too long".into())
}

fn fields(mut b: &[u8]) -> Result<Vec<(u64, Val<'_>)>, String> {
    let mut out = Vec::new();
    while !b.is_empty() {
        let (key, rest) = varint(b)?;
        b = rest;
        let (field, wire) = (key >> 3, key & 7);
        match wire {
            0 => {
                let (v, rest) = varint(b)?;
                b = rest;
                out.push((field, Val::Int(v)));
            }
            1 => {
                if b.len() < 8 {
                    return Err("truncated fixed64".into());
                }
                b = &b[8..];
            }
            2 => {
                let (len, rest) = varint(b)?;
                let len = len as usize;
                if rest.len() < len {
                    return Err("truncated length-delimited field".into());
                }
                out.push((field, Val::Bytes(&rest[..len])));
                b = &rest[len..];
            }
            5 => {
                if b.len() < 4 {
                    return Err("truncated fixed32".into());
                }
                b = &b[4..];
            }
            w => return Err(format!("unsupported wire type {w}")),
        }
    }
    Ok(out)
}
