use crate::policy::{PromptSpan, SessionPolicy};
use xrt_core::{KvCache, Result, XrtError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheMode {
    F32,
    Q8,
    KeyQ4ValueQ8,
    AgentAdaptive,
}

impl KvCacheMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" | "float" | "float32" => Some(Self::F32),
            "q8" | "int8" => Some(Self::Q8),
            "kq4_vq8" | "kq4" | "key_q4_value_q8" | "key-q4-value-q8" | "q4_keys_q8_values" => {
                Some(Self::KeyQ4ValueQ8)
            }
            "agent_adaptive" | "agent-adaptive" | "adaptive" => Some(Self::AgentAdaptive),
            _ => None,
        }
    }

    pub fn from_env() -> Self {
        std::env::var("XRT_KV_CACHE_MODE")
            .ok()
            .as_deref()
            .and_then(Self::parse)
            .unwrap_or(Self::F32)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8 => "q8",
            Self::KeyQ4ValueQ8 => "kq4_vq8",
            Self::AgentAdaptive => "agent_adaptive",
        }
    }
}

const KEY_Q4_GROUP_SIZE: usize = 64;

#[derive(Debug, Clone)]
struct KvPage {
    keys: Vec<f32>,
    values: Vec<f32>,
    occupied: Vec<bool>,
    len: usize,
}

impl KvPage {
    fn new(width: usize, page_tokens: usize) -> Self {
        Self {
            keys: vec![0.0; width * page_tokens],
            values: vec![0.0; width * page_tokens],
            occupied: vec![false; page_tokens],
            len: 0,
        }
    }

    fn clear_slot(&mut self, width: usize, slot: usize) {
        self.occupied[slot] = false;
        let offset = slot * width;
        self.keys[offset..offset + width].fill(0.0);
        self.values[offset..offset + width].fill(0.0);
        if slot + 1 == self.len {
            while self.len > 0 && !self.occupied[self.len - 1] {
                self.len -= 1;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct LayerPages {
    pages: Vec<KvPage>,
    len: usize,
}

#[derive(Debug, Clone)]
pub struct PagedKvCache {
    layers: Vec<LayerPages>,
    width: usize,
    page_tokens: usize,
}

impl PagedKvCache {
    pub fn new(layer_count: usize, width: usize, page_tokens: usize) -> Self {
        Self {
            layers: (0..layer_count)
                .map(|_| LayerPages {
                    pages: Vec::new(),
                    len: 0,
                })
                .collect(),
            width,
            page_tokens: page_tokens.max(1),
        }
    }

    fn ensure_page_mut(&mut self, layer: usize, page_index: usize) -> Result<&mut KvPage> {
        let layer = self.layers.get_mut(layer).ok_or_else(|| {
            XrtError::Runtime(format!("layer {layer} is out of range for KV cache"))
        })?;
        while layer.pages.len() <= page_index {
            layer.pages.push(KvPage::new(self.width, self.page_tokens));
        }
        Ok(&mut layer.pages[page_index])
    }

    fn locate(&self, layer: usize, position: usize) -> Option<(&KvPage, usize)> {
        let layer = self.layers.get(layer)?;
        if position >= layer.len {
            return None;
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let page = layer.pages.get(page_index)?;
        page.occupied
            .get(slot)
            .copied()
            .unwrap_or(false)
            .then_some((page, slot))
    }

    pub fn has_position(&self, layer: usize, position: usize) -> bool {
        self.locate(layer, position).is_some()
    }

    pub fn append_at(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        if key.len() != self.width || value.len() != self.width {
            return Err(XrtError::Runtime(format!(
                "KV cache append width mismatch: expected {}, got key {} and value {}",
                self.width,
                key.len(),
                value.len()
            )));
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let width = self.width;
        let page = self.ensure_page_mut(layer, page_index)?;
        let offset = slot * width;
        page.keys[offset..offset + width].copy_from_slice(key);
        page.values[offset..offset + width].copy_from_slice(value);
        page.occupied[slot] = true;
        page.len = page.len.max(slot + 1);
        if let Some(layer_data) = self.layers.get_mut(layer) {
            layer_data.len = layer_data.len.max(position + 1);
        }
        Ok(())
    }

    pub fn remove_at(&mut self, layer: usize, position: usize) {
        let Some(layer_data) = self.layers.get_mut(layer) else {
            return;
        };
        if position >= layer_data.len {
            return;
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let Some(page) = layer_data.pages.get_mut(page_index) else {
            return;
        };
        if !page.occupied.get(slot).copied().unwrap_or(false) {
            return;
        }
        page.clear_slot(self.width, slot);
        if position + 1 == layer_data.len {
            while layer_data.len > 0 {
                let tail = layer_data.len - 1;
                if layer_has_f32_position(layer_data, self.page_tokens, tail) {
                    break;
                }
                layer_data.len -= 1;
            }
            let pages_needed = if layer_data.len == 0 {
                0
            } else {
                (layer_data.len + self.page_tokens - 1) / self.page_tokens
            };
            layer_data.pages.truncate(pages_needed);
        }
    }
}

impl KvCache for PagedKvCache {
    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn width(&self) -> usize {
        self.width
    }

    fn len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .map(|layer| layer.len)
            .unwrap_or_default()
    }

    fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        let position = self.len(layer);
        self.append_at(layer, position, key, value)
    }

    fn key(&self, layer: usize, position: usize) -> Option<&[f32]> {
        let (page, slot) = self.locate(layer, position)?;
        let offset = slot * self.width;
        Some(&page.keys[offset..offset + self.width])
    }

    fn value(&self, layer: usize, position: usize) -> Option<&[f32]> {
        let (page, slot) = self.locate(layer, position)?;
        let offset = slot * self.width;
        Some(&page.values[offset..offset + self.width])
    }

    fn append_batch(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let w = self.width;
        if keys.len() != count * w || values.len() != count * w {
            return Err(XrtError::Runtime(format!(
                "KV cache append_batch size mismatch: expected {} elements, got keys {} values {}",
                count * w,
                keys.len(),
                values.len()
            )));
        }

        let start = self.len(layer);
        for i in 0..count {
            self.append_at(
                layer,
                start + i,
                &keys[i * w..(i + 1) * w],
                &values[i * w..(i + 1) * w],
            )?;
        }
        Ok(())
    }

    fn truncate(&mut self, new_len: usize) {
        for layer_index in 0..self.layers.len() {
            let current_len = self.len(layer_index);
            if new_len >= current_len {
                continue;
            }
            let old_len = current_len;
            if let Some(layer) = self.layers.get_mut(layer_index) {
                layer.len = new_len;
            }
            for position in new_len..old_len {
                let page_index = position / self.page_tokens;
                let slot = position % self.page_tokens;
                if let Some(page) = self.layers[layer_index].pages.get_mut(page_index) {
                    if page.occupied.get(slot).copied().unwrap_or(false) {
                        page.clear_slot(self.width, slot);
                    }
                }
            }
            let pages_needed = if new_len == 0 {
                0
            } else {
                (new_len + self.page_tokens - 1) / self.page_tokens
            };
            self.layers[layer_index].pages.truncate(pages_needed);
        }
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.pages.clear();
            layer.len = 0;
        }
    }
}

#[derive(Debug, Clone)]
struct QuantizedKvPage {
    keys: Vec<i8>,
    values: Vec<i8>,
    key_scales: Vec<f32>,
    value_scales: Vec<f32>,
    occupied: Vec<bool>,
    len: usize,
}

impl QuantizedKvPage {
    fn new(width: usize, page_tokens: usize) -> Self {
        Self {
            keys: vec![0; width * page_tokens],
            values: vec![0; width * page_tokens],
            key_scales: vec![1.0; page_tokens],
            value_scales: vec![1.0; page_tokens],
            occupied: vec![false; page_tokens],
            len: 0,
        }
    }

    fn clear_slot(&mut self, width: usize, slot: usize) {
        self.occupied[slot] = false;
        self.key_scales[slot] = 1.0;
        self.value_scales[slot] = 1.0;
        let offset = slot * width;
        self.keys[offset..offset + width].fill(0);
        self.values[offset..offset + width].fill(0);
        if slot + 1 == self.len {
            while self.len > 0 && !self.occupied[self.len - 1] {
                self.len -= 1;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct QuantizedLayerPages {
    pages: Vec<QuantizedKvPage>,
    len: usize,
}

#[derive(Debug, Clone)]
pub struct QuantizedPagedKvCache {
    layers: Vec<QuantizedLayerPages>,
    width: usize,
    page_tokens: usize,
}

impl QuantizedPagedKvCache {
    pub fn new(layer_count: usize, width: usize, page_tokens: usize) -> Self {
        Self {
            layers: (0..layer_count)
                .map(|_| QuantizedLayerPages {
                    pages: Vec::new(),
                    len: 0,
                })
                .collect(),
            width,
            page_tokens: page_tokens.max(1),
        }
    }

    fn ensure_page_mut(&mut self, layer: usize, page_index: usize) -> Result<&mut QuantizedKvPage> {
        let layer = self.layers.get_mut(layer).ok_or_else(|| {
            XrtError::Runtime(format!("layer {layer} is out of range for KV cache"))
        })?;
        while layer.pages.len() <= page_index {
            layer
                .pages
                .push(QuantizedKvPage::new(self.width, self.page_tokens));
        }
        Ok(&mut layer.pages[page_index])
    }

    fn locate(&self, layer: usize, position: usize) -> Option<(&QuantizedKvPage, usize)> {
        let layer = self.layers.get(layer)?;
        if position >= layer.len {
            return None;
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let page = layer.pages.get(page_index)?;
        page.occupied
            .get(slot)
            .copied()
            .unwrap_or(false)
            .then_some((page, slot))
    }

    pub fn has_position(&self, layer: usize, position: usize) -> bool {
        self.locate(layer, position).is_some()
    }

    pub fn append_at(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        if key.len() != self.width || value.len() != self.width {
            return Err(XrtError::Runtime(format!(
                "KV cache append width mismatch: expected {}, got key {} and value {}",
                self.width,
                key.len(),
                value.len()
            )));
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let width = self.width;
        let page = self.ensure_page_mut(layer, page_index)?;
        let offset = slot * width;
        page.key_scales[slot] = quantize_row(key, &mut page.keys[offset..offset + width]);
        page.value_scales[slot] = quantize_row(value, &mut page.values[offset..offset + width]);
        page.occupied[slot] = true;
        page.len = page.len.max(slot + 1);
        if let Some(layer_data) = self.layers.get_mut(layer) {
            layer_data.len = layer_data.len.max(position + 1);
        }
        Ok(())
    }

    pub fn remove_at(&mut self, layer: usize, position: usize) {
        let Some(layer_data) = self.layers.get_mut(layer) else {
            return;
        };
        if position >= layer_data.len {
            return;
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let Some(page) = layer_data.pages.get_mut(page_index) else {
            return;
        };
        if !page.occupied.get(slot).copied().unwrap_or(false) {
            return;
        }
        page.clear_slot(self.width, slot);
        if position + 1 == layer_data.len {
            while layer_data.len > 0 {
                let tail = layer_data.len - 1;
                if layer_has_q8_position(layer_data, self.page_tokens, tail) {
                    break;
                }
                layer_data.len -= 1;
            }
            let pages_needed = if layer_data.len == 0 {
                0
            } else {
                (layer_data.len + self.page_tokens - 1) / self.page_tokens
            };
            layer_data.pages.truncate(pages_needed);
        }
    }
}

impl KvCache for QuantizedPagedKvCache {
    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn width(&self) -> usize {
        self.width
    }

    fn len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .map(|layer| layer.len)
            .unwrap_or_default()
    }

    fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        let position = self.len(layer);
        self.append_at(layer, position, key, value)
    }

    fn key(&self, _layer: usize, _position: usize) -> Option<&[f32]> {
        None
    }

    fn value(&self, _layer: usize, _position: usize) -> Option<&[f32]> {
        None
    }

    fn copy_key_into(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<()> {
        if out.len() != self.width {
            return Err(XrtError::Runtime(format!(
                "KV cache key read width mismatch: expected {}, got {}",
                self.width,
                out.len()
            )));
        }
        let (page, slot) = self.locate(layer, position).ok_or_else(|| {
            XrtError::Runtime(format!(
                "missing quantized key cache entry at layer {layer} position {position}"
            ))
        })?;
        let offset = slot * self.width;
        dequantize_row(
            &page.keys[offset..offset + self.width],
            page.key_scales[slot],
            out,
        );
        Ok(())
    }

    fn copy_value_into(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<()> {
        if out.len() != self.width {
            return Err(XrtError::Runtime(format!(
                "KV cache value read width mismatch: expected {}, got {}",
                self.width,
                out.len()
            )));
        }
        let (page, slot) = self.locate(layer, position).ok_or_else(|| {
            XrtError::Runtime(format!(
                "missing quantized value cache entry at layer {layer} position {position}"
            ))
        })?;
        let offset = slot * self.width;
        dequantize_row(
            &page.values[offset..offset + self.width],
            page.value_scales[slot],
            out,
        );
        Ok(())
    }

    fn append_batch(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let w = self.width;
        if keys.len() != count * w || values.len() != count * w {
            return Err(XrtError::Runtime(format!(
                "KV cache append_batch size mismatch: expected {} elements, got keys {} values {}",
                count * w,
                keys.len(),
                values.len()
            )));
        }

        let start = self.len(layer);
        for i in 0..count {
            self.append_at(
                layer,
                start + i,
                &keys[i * w..(i + 1) * w],
                &values[i * w..(i + 1) * w],
            )?;
        }
        Ok(())
    }

    fn truncate(&mut self, new_len: usize) {
        for layer_index in 0..self.layers.len() {
            let current_len = self.len(layer_index);
            if new_len >= current_len {
                continue;
            }
            let old_len = current_len;
            if let Some(layer) = self.layers.get_mut(layer_index) {
                layer.len = new_len;
            }
            for position in new_len..old_len {
                let page_index = position / self.page_tokens;
                let slot = position % self.page_tokens;
                if let Some(page) = self.layers[layer_index].pages.get_mut(page_index) {
                    if page.occupied.get(slot).copied().unwrap_or(false) {
                        page.clear_slot(self.width, slot);
                    }
                }
            }
            let pages_needed = if new_len == 0 {
                0
            } else {
                (new_len + self.page_tokens - 1) / self.page_tokens
            };
            self.layers[layer_index].pages.truncate(pages_needed);
        }
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.pages.clear();
            layer.len = 0;
        }
    }
}

#[derive(Debug, Clone)]
struct KeyQ4ValueQ8KvPage {
    keys: Vec<u8>,
    values: Vec<i8>,
    key_scales: Vec<f32>,
    value_scales: Vec<f32>,
    occupied: Vec<bool>,
    len: usize,
}

impl KeyQ4ValueQ8KvPage {
    fn new(width: usize, page_tokens: usize) -> Self {
        let key_row_bytes = packed_q4_row_bytes(width);
        let key_groups = q4_groups_for_width(width);
        Self {
            keys: vec![0; key_row_bytes * page_tokens],
            values: vec![0; width * page_tokens],
            key_scales: vec![1.0; key_groups * page_tokens],
            value_scales: vec![1.0; page_tokens],
            occupied: vec![false; page_tokens],
            len: 0,
        }
    }

    fn clear_slot(&mut self, width: usize, slot: usize) {
        self.occupied[slot] = false;
        self.value_scales[slot] = 1.0;
        let key_row_bytes = packed_q4_row_bytes(width);
        let key_groups = q4_groups_for_width(width);
        let key_offset = slot * key_row_bytes;
        let scale_offset = slot * key_groups;
        let value_offset = slot * width;
        self.keys[key_offset..key_offset + key_row_bytes].fill(0);
        self.key_scales[scale_offset..scale_offset + key_groups].fill(1.0);
        self.values[value_offset..value_offset + width].fill(0);
        if slot + 1 == self.len {
            while self.len > 0 && !self.occupied[self.len - 1] {
                self.len -= 1;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct KeyQ4ValueQ8LayerPages {
    pages: Vec<KeyQ4ValueQ8KvPage>,
    len: usize,
}

#[derive(Debug, Clone)]
pub struct KeyQ4ValueQ8PagedKvCache {
    layers: Vec<KeyQ4ValueQ8LayerPages>,
    width: usize,
    page_tokens: usize,
}

impl KeyQ4ValueQ8PagedKvCache {
    pub fn new(layer_count: usize, width: usize, page_tokens: usize) -> Self {
        Self {
            layers: (0..layer_count)
                .map(|_| KeyQ4ValueQ8LayerPages {
                    pages: Vec::new(),
                    len: 0,
                })
                .collect(),
            width,
            page_tokens: page_tokens.max(1),
        }
    }

    fn ensure_page_mut(
        &mut self,
        layer: usize,
        page_index: usize,
    ) -> Result<&mut KeyQ4ValueQ8KvPage> {
        let layer = self.layers.get_mut(layer).ok_or_else(|| {
            XrtError::Runtime(format!("layer {layer} is out of range for KV cache"))
        })?;
        while layer.pages.len() <= page_index {
            layer
                .pages
                .push(KeyQ4ValueQ8KvPage::new(self.width, self.page_tokens));
        }
        Ok(&mut layer.pages[page_index])
    }

    fn locate(&self, layer: usize, position: usize) -> Option<(&KeyQ4ValueQ8KvPage, usize)> {
        let layer = self.layers.get(layer)?;
        if position >= layer.len {
            return None;
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let page = layer.pages.get(page_index)?;
        page.occupied
            .get(slot)
            .copied()
            .unwrap_or(false)
            .then_some((page, slot))
    }

    pub fn has_position(&self, layer: usize, position: usize) -> bool {
        self.locate(layer, position).is_some()
    }

    pub fn append_at(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        if key.len() != self.width || value.len() != self.width {
            return Err(XrtError::Runtime(format!(
                "KV cache append width mismatch: expected {}, got key {} and value {}",
                self.width,
                key.len(),
                value.len()
            )));
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let width = self.width;
        let key_row_bytes = packed_q4_row_bytes(width);
        let key_groups = q4_groups_for_width(width);
        let page = self.ensure_page_mut(layer, page_index)?;
        let key_offset = slot * key_row_bytes;
        let scale_offset = slot * key_groups;
        let value_offset = slot * width;

        quantize_row_q4(
            key,
            &mut page.keys[key_offset..key_offset + key_row_bytes],
            &mut page.key_scales[scale_offset..scale_offset + key_groups],
        );
        page.value_scales[slot] =
            quantize_row(value, &mut page.values[value_offset..value_offset + width]);
        page.occupied[slot] = true;
        page.len = page.len.max(slot + 1);
        if let Some(layer_data) = self.layers.get_mut(layer) {
            layer_data.len = layer_data.len.max(position + 1);
        }
        Ok(())
    }

    pub fn remove_at(&mut self, layer: usize, position: usize) {
        let Some(layer_data) = self.layers.get_mut(layer) else {
            return;
        };
        if position >= layer_data.len {
            return;
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        let Some(page) = layer_data.pages.get_mut(page_index) else {
            return;
        };
        if !page.occupied.get(slot).copied().unwrap_or(false) {
            return;
        }
        page.clear_slot(self.width, slot);
        if position + 1 == layer_data.len {
            while layer_data.len > 0 {
                let tail = layer_data.len - 1;
                if layer_has_key_q4_value_q8_position(layer_data, self.page_tokens, tail) {
                    break;
                }
                layer_data.len -= 1;
            }
            let pages_needed = if layer_data.len == 0 {
                0
            } else {
                (layer_data.len + self.page_tokens - 1) / self.page_tokens
            };
            layer_data.pages.truncate(pages_needed);
        }
    }
}

impl KvCache for KeyQ4ValueQ8PagedKvCache {
    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn width(&self) -> usize {
        self.width
    }

    fn len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .map(|layer| layer.len)
            .unwrap_or_default()
    }

    fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        let position = self.len(layer);
        self.append_at(layer, position, key, value)
    }

    fn key(&self, _layer: usize, _position: usize) -> Option<&[f32]> {
        None
    }

    fn value(&self, _layer: usize, _position: usize) -> Option<&[f32]> {
        None
    }

    fn copy_key_into(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<()> {
        if out.len() != self.width {
            return Err(XrtError::Runtime(format!(
                "KV cache key read width mismatch: expected {}, got {}",
                self.width,
                out.len()
            )));
        }
        let (page, slot) = self.locate(layer, position).ok_or_else(|| {
            XrtError::Runtime(format!(
                "missing key-q4/value-q8 key cache entry at layer {layer} position {position}"
            ))
        })?;
        let key_row_bytes = packed_q4_row_bytes(self.width);
        let key_groups = q4_groups_for_width(self.width);
        let key_offset = slot * key_row_bytes;
        let scale_offset = slot * key_groups;
        dequantize_row_q4(
            &page.keys[key_offset..key_offset + key_row_bytes],
            &page.key_scales[scale_offset..scale_offset + key_groups],
            out,
        );
        Ok(())
    }

    fn copy_value_into(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<()> {
        if out.len() != self.width {
            return Err(XrtError::Runtime(format!(
                "KV cache value read width mismatch: expected {}, got {}",
                self.width,
                out.len()
            )));
        }
        let (page, slot) = self.locate(layer, position).ok_or_else(|| {
            XrtError::Runtime(format!(
                "missing key-q4/value-q8 value cache entry at layer {layer} position {position}"
            ))
        })?;
        let value_offset = slot * self.width;
        dequantize_row(
            &page.values[value_offset..value_offset + self.width],
            page.value_scales[slot],
            out,
        );
        Ok(())
    }

    fn append_batch(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let w = self.width;
        if keys.len() != count * w || values.len() != count * w {
            return Err(XrtError::Runtime(format!(
                "KV cache append_batch size mismatch: expected {} elements, got keys {} values {}",
                count * w,
                keys.len(),
                values.len()
            )));
        }

        let start = self.len(layer);
        for i in 0..count {
            self.append_at(
                layer,
                start + i,
                &keys[i * w..(i + 1) * w],
                &values[i * w..(i + 1) * w],
            )?;
        }
        Ok(())
    }

    fn truncate(&mut self, new_len: usize) {
        for layer_index in 0..self.layers.len() {
            let current_len = self.len(layer_index);
            if new_len >= current_len {
                continue;
            }
            let old_len = current_len;
            if let Some(layer) = self.layers.get_mut(layer_index) {
                layer.len = new_len;
            }
            for position in new_len..old_len {
                let page_index = position / self.page_tokens;
                let slot = position % self.page_tokens;
                if let Some(page) = self.layers[layer_index].pages.get_mut(page_index) {
                    if page.occupied.get(slot).copied().unwrap_or(false) {
                        page.clear_slot(self.width, slot);
                    }
                }
            }
            let pages_needed = if new_len == 0 {
                0
            } else {
                (new_len + self.page_tokens - 1) / self.page_tokens
            };
            self.layers[layer_index].pages.truncate(pages_needed);
        }
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.pages.clear();
            layer.len = 0;
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveKvCache {
    hot: PagedKvCache,
    cold: QuantizedPagedKvCache,
    policy: SessionPolicy,
    pinned_positions: Vec<bool>,
    layer_lengths: Vec<usize>,
}

impl AdaptiveKvCache {
    fn new(layer_count: usize, width: usize, page_tokens: usize) -> Self {
        Self {
            hot: PagedKvCache::new(layer_count, width, page_tokens),
            cold: QuantizedPagedKvCache::new(layer_count, width, page_tokens),
            policy: SessionPolicy::agent_adaptive(),
            pinned_positions: Vec::new(),
            layer_lengths: vec![0; layer_count],
        }
    }

    fn layers(&self) -> usize {
        self.layer_lengths.len()
    }

    fn width(&self) -> usize {
        self.hot.width()
    }

    fn len(&self, layer: usize) -> usize {
        self.layer_lengths.get(layer).copied().unwrap_or_default()
    }

    fn configure_policy(
        &mut self,
        policy: SessionPolicy,
        prompt_token_count: usize,
        spans: &[PromptSpan],
    ) {
        self.policy = policy;
        self.pinned_positions.clear();
        self.pinned_positions.resize(prompt_token_count, false);
        for span in spans {
            if !self.policy.is_span_pinned(span.kind) {
                continue;
            }
            let end = span.token_end.min(self.pinned_positions.len());
            for position in span.token_start.min(end)..end {
                self.pinned_positions[position] = true;
            }
        }
    }

    fn is_hot_position(&self, position: usize, total_len: usize) -> bool {
        let recent_window = self.policy.recent_window_tokens.max(1);
        let recent_start = total_len.saturating_sub(recent_window);
        position >= recent_start
            || self
                .pinned_positions
                .get(position)
                .copied()
                .unwrap_or(false)
    }

    fn prepare_for_total_len(&mut self, total_len: usize) -> Result<()> {
        let width = self.width();
        let mut key_buf = vec![0.0; width];
        let mut value_buf = vec![0.0; width];
        for layer in 0..self.layers() {
            let current_len = self.len(layer);
            if current_len == 0 {
                continue;
            }
            let target_len = total_len.min(current_len);
            for position in 0..target_len {
                let should_be_hot = self.is_hot_position(position, total_len);
                let in_hot = self.hot.has_position(layer, position);
                let in_cold = self.cold.has_position(layer, position);
                if should_be_hot && in_cold && !in_hot {
                    self.cold.copy_key_into(layer, position, &mut key_buf)?;
                    self.cold.copy_value_into(layer, position, &mut value_buf)?;
                    self.hot.append_at(layer, position, &key_buf, &value_buf)?;
                    self.cold.remove_at(layer, position);
                } else if !should_be_hot && in_hot && !in_cold {
                    self.hot.copy_key_into(layer, position, &mut key_buf)?;
                    self.hot.copy_value_into(layer, position, &mut value_buf)?;
                    self.cold.append_at(layer, position, &key_buf, &value_buf)?;
                    self.hot.remove_at(layer, position);
                }
            }
        }
        Ok(())
    }

    fn append_at(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        let total_len = self
            .layer_lengths
            .get(layer)
            .copied()
            .unwrap_or_default()
            .max(position + 1);
        let route_hot = self.is_hot_position(position, total_len);
        if route_hot {
            self.hot.append_at(layer, position, key, value)?;
            self.cold.remove_at(layer, position);
        } else {
            self.cold.append_at(layer, position, key, value)?;
            self.hot.remove_at(layer, position);
        }
        if let Some(layer_len) = self.layer_lengths.get_mut(layer) {
            *layer_len = (*layer_len).max(position + 1);
        }
        Ok(())
    }

    fn append_batch(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let width = self.width();
        if keys.len() != count * width || values.len() != count * width {
            return Err(XrtError::Runtime(format!(
                "adaptive KV append_batch size mismatch: expected {} elements, got keys {} values {}",
                count * width,
                keys.len(),
                values.len()
            )));
        }
        let start = self.len(layer);
        let final_len = start + count;
        for i in 0..count {
            let position = start + i;
            let route_hot = self.is_hot_position(position, final_len);
            let key = &keys[i * width..(i + 1) * width];
            let value = &values[i * width..(i + 1) * width];
            if route_hot {
                self.hot.append_at(layer, position, key, value)?;
                self.cold.remove_at(layer, position);
            } else {
                self.cold.append_at(layer, position, key, value)?;
                self.hot.remove_at(layer, position);
            }
        }
        if let Some(layer_len) = self.layer_lengths.get_mut(layer) {
            *layer_len = final_len;
        }
        Ok(())
    }

    fn copy_key_into(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<()> {
        if self.hot.has_position(layer, position) {
            self.hot.copy_key_into(layer, position, out)
        } else {
            self.cold.copy_key_into(layer, position, out)
        }
    }

    fn copy_value_into(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<()> {
        if self.hot.has_position(layer, position) {
            self.hot.copy_value_into(layer, position, out)
        } else {
            self.cold.copy_value_into(layer, position, out)
        }
    }

    fn truncate(&mut self, new_len: usize) {
        self.hot.truncate(new_len);
        self.cold.truncate(new_len);
        self.pinned_positions.truncate(new_len);
        for layer_len in &mut self.layer_lengths {
            *layer_len = (*layer_len).min(new_len);
        }
    }

    fn clear(&mut self) {
        self.hot.clear();
        self.cold.clear();
        self.pinned_positions.clear();
        self.layer_lengths.fill(0);
    }
}

#[derive(Debug, Clone)]
pub enum SessionKvCache {
    F32(PagedKvCache),
    Q8(QuantizedPagedKvCache),
    KeyQ4ValueQ8(KeyQ4ValueQ8PagedKvCache),
    AgentAdaptive(AdaptiveKvCache),
}

impl SessionKvCache {
    pub fn new(mode: KvCacheMode, layer_count: usize, width: usize, page_tokens: usize) -> Self {
        match mode {
            KvCacheMode::F32 => Self::F32(PagedKvCache::new(layer_count, width, page_tokens)),
            KvCacheMode::Q8 => {
                Self::Q8(QuantizedPagedKvCache::new(layer_count, width, page_tokens))
            }
            KvCacheMode::KeyQ4ValueQ8 => Self::KeyQ4ValueQ8(KeyQ4ValueQ8PagedKvCache::new(
                layer_count,
                width,
                page_tokens,
            )),
            KvCacheMode::AgentAdaptive => {
                Self::AgentAdaptive(AdaptiveKvCache::new(layer_count, width, page_tokens))
            }
        }
    }

    pub fn mode(&self) -> KvCacheMode {
        match self {
            Self::F32(_) => KvCacheMode::F32,
            Self::Q8(_) => KvCacheMode::Q8,
            Self::KeyQ4ValueQ8(_) => KvCacheMode::KeyQ4ValueQ8,
            Self::AgentAdaptive(_) => KvCacheMode::AgentAdaptive,
        }
    }

    pub fn configure_policy(
        &mut self,
        policy: SessionPolicy,
        prompt_token_count: usize,
        spans: &[PromptSpan],
    ) {
        if let Self::AgentAdaptive(cache) = self {
            cache.configure_policy(policy, prompt_token_count, spans);
        }
    }

    pub fn prepare_for_total_len(&mut self, total_len: usize) -> Result<()> {
        if let Self::AgentAdaptive(cache) = self {
            cache.prepare_for_total_len(total_len)?;
        }
        Ok(())
    }
}

impl KvCache for SessionKvCache {
    fn layers(&self) -> usize {
        match self {
            Self::F32(cache) => cache.layers(),
            Self::Q8(cache) => cache.layers(),
            Self::KeyQ4ValueQ8(cache) => cache.layers(),
            Self::AgentAdaptive(cache) => cache.layers(),
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::F32(cache) => cache.width(),
            Self::Q8(cache) => cache.width(),
            Self::KeyQ4ValueQ8(cache) => cache.width(),
            Self::AgentAdaptive(cache) => cache.width(),
        }
    }

    fn len(&self, layer: usize) -> usize {
        match self {
            Self::F32(cache) => cache.len(layer),
            Self::Q8(cache) => cache.len(layer),
            Self::KeyQ4ValueQ8(cache) => cache.len(layer),
            Self::AgentAdaptive(cache) => cache.len(layer),
        }
    }

    fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        match self {
            Self::F32(cache) => cache.append(layer, key, value),
            Self::Q8(cache) => cache.append(layer, key, value),
            Self::KeyQ4ValueQ8(cache) => cache.append(layer, key, value),
            Self::AgentAdaptive(cache) => cache.append_at(layer, cache.len(layer), key, value),
        }
    }

    fn key(&self, layer: usize, position: usize) -> Option<&[f32]> {
        match self {
            Self::F32(cache) => cache.key(layer, position),
            Self::Q8(cache) => cache.key(layer, position),
            Self::KeyQ4ValueQ8(cache) => cache.key(layer, position),
            Self::AgentAdaptive(_) => None,
        }
    }

    fn value(&self, layer: usize, position: usize) -> Option<&[f32]> {
        match self {
            Self::F32(cache) => cache.value(layer, position),
            Self::Q8(cache) => cache.value(layer, position),
            Self::KeyQ4ValueQ8(cache) => cache.value(layer, position),
            Self::AgentAdaptive(_) => None,
        }
    }

    fn copy_key_into(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<()> {
        match self {
            Self::F32(cache) => cache.copy_key_into(layer, position, out),
            Self::Q8(cache) => cache.copy_key_into(layer, position, out),
            Self::KeyQ4ValueQ8(cache) => cache.copy_key_into(layer, position, out),
            Self::AgentAdaptive(cache) => cache.copy_key_into(layer, position, out),
        }
    }

    fn copy_value_into(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<()> {
        match self {
            Self::F32(cache) => cache.copy_value_into(layer, position, out),
            Self::Q8(cache) => cache.copy_value_into(layer, position, out),
            Self::KeyQ4ValueQ8(cache) => cache.copy_value_into(layer, position, out),
            Self::AgentAdaptive(cache) => cache.copy_value_into(layer, position, out),
        }
    }

    fn append_batch(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        count: usize,
    ) -> Result<()> {
        match self {
            Self::F32(cache) => cache.append_batch(layer, keys, values, count),
            Self::Q8(cache) => cache.append_batch(layer, keys, values, count),
            Self::KeyQ4ValueQ8(cache) => cache.append_batch(layer, keys, values, count),
            Self::AgentAdaptive(cache) => cache.append_batch(layer, keys, values, count),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::F32(cache) => cache.clear(),
            Self::Q8(cache) => cache.clear(),
            Self::KeyQ4ValueQ8(cache) => cache.clear(),
            Self::AgentAdaptive(cache) => cache.clear(),
        }
    }

    fn truncate(&mut self, new_len: usize) {
        match self {
            Self::F32(cache) => cache.truncate(new_len),
            Self::Q8(cache) => cache.truncate(new_len),
            Self::KeyQ4ValueQ8(cache) => cache.truncate(new_len),
            Self::AgentAdaptive(cache) => cache.truncate(new_len),
        }
    }
}

fn packed_q4_row_bytes(width: usize) -> usize {
    width.div_ceil(2)
}

fn q4_groups_for_width(width: usize) -> usize {
    width.div_ceil(KEY_Q4_GROUP_SIZE)
}

fn quantize_row_q4(input: &[f32], out: &mut [u8], scales: &mut [f32]) {
    out.fill(0);
    for (group_index, chunk) in input.chunks(KEY_Q4_GROUP_SIZE).enumerate() {
        let max_abs = chunk.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = if max_abs <= f32::EPSILON {
            1.0
        } else {
            max_abs / 8.0
        };
        scales[group_index] = scale;
        for (lane, value) in chunk.iter().enumerate() {
            let quant = if scale <= f32::EPSILON {
                0i8
            } else {
                (*value / scale).round().clamp(-8.0, 7.0) as i8
            };
            let packed = (quant + 8) as u8;
            let element_index = group_index * KEY_Q4_GROUP_SIZE + lane;
            let byte_index = element_index / 2;
            if element_index % 2 == 0 {
                out[byte_index] = packed;
            } else {
                out[byte_index] |= packed << 4;
            }
        }
    }
}

fn dequantize_row_q4(input: &[u8], scales: &[f32], out: &mut [f32]) {
    for (group_index, chunk) in out.chunks_mut(KEY_Q4_GROUP_SIZE).enumerate() {
        let scale = scales[group_index];
        for lane in 0..chunk.len() {
            let element_index = group_index * KEY_Q4_GROUP_SIZE + lane;
            let byte = input[element_index / 2];
            let nibble = if element_index % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            let quant = nibble as i8 - 8;
            chunk[lane] = f32::from(quant) * scale;
        }
    }
}

fn quantize_row(input: &[f32], out: &mut [i8]) -> f32 {
    let max_abs = input.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
    let scale = if max_abs <= f32::EPSILON {
        1.0
    } else {
        max_abs / 127.0
    };
    for (source, target) in input.iter().zip(out.iter_mut()) {
        *target = (*source / scale).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

fn dequantize_row(input: &[i8], scale: f32, out: &mut [f32]) {
    for (source, target) in input.iter().zip(out.iter_mut()) {
        *target = f32::from(*source) * scale;
    }
}

fn layer_has_f32_position(layer: &LayerPages, page_tokens: usize, position: usize) -> bool {
    if position >= layer.len {
        return false;
    }
    let page_index = position / page_tokens;
    let slot = position % page_tokens;
    layer
        .pages
        .get(page_index)
        .and_then(|page| page.occupied.get(slot))
        .copied()
        .unwrap_or(false)
}

fn layer_has_q8_position(layer: &QuantizedLayerPages, page_tokens: usize, position: usize) -> bool {
    if position >= layer.len {
        return false;
    }
    let page_index = position / page_tokens;
    let slot = position % page_tokens;
    layer
        .pages
        .get(page_index)
        .and_then(|page| page.occupied.get(slot))
        .copied()
        .unwrap_or(false)
}

fn layer_has_key_q4_value_q8_position(
    layer: &KeyQ4ValueQ8LayerPages,
    page_tokens: usize,
    position: usize,
) -> bool {
    if position >= layer.len {
        return false;
    }
    let page_index = position / page_tokens;
    let slot = position % page_tokens;
    layer
        .pages
        .get(page_index)
        .and_then(|page| page.occupied.get(slot))
        .copied()
        .unwrap_or(false)
}
