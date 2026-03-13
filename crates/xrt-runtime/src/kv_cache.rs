use xrt_core::{KvCache, Result, XrtError};

#[derive(Debug, Clone)]
struct KvPage {
    keys: Vec<f32>,
    values: Vec<f32>,
    len: usize,
}

impl KvPage {
    fn new(width: usize, page_tokens: usize) -> Self {
        Self {
            keys: vec![0.0; width * page_tokens],
            values: vec![0.0; width * page_tokens],
            len: 0,
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

    fn locate(&self, layer: usize, position: usize) -> Option<(&KvPage, usize)> {
        let layer = self.layers.get(layer)?;
        if position >= layer.len {
            return None;
        }
        let page_index = position / self.page_tokens;
        let slot = position % self.page_tokens;
        layer.pages.get(page_index).map(|page| (page, slot))
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
        if key.len() != self.width || value.len() != self.width {
            return Err(XrtError::Runtime(format!(
                "KV cache append width mismatch: expected {}, got key {} and value {}",
                self.width,
                key.len(),
                value.len()
            )));
        }
        let layer = self.layers.get_mut(layer).ok_or_else(|| {
            XrtError::Runtime(format!("layer {layer} is out of range for KV cache"))
        })?;
        let page_index = layer.len / self.page_tokens;
        let slot = layer.len % self.page_tokens;
        if page_index == layer.pages.len() {
            layer.pages.push(KvPage::new(self.width, self.page_tokens));
        }
        let page = &mut layer.pages[page_index];
        let offset = slot * self.width;
        page.keys[offset..offset + self.width].copy_from_slice(key);
        page.values[offset..offset + self.width].copy_from_slice(value);
        page.len = page.len.max(slot + 1);
        layer.len += 1;
        Ok(())
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

        let layer_data = self.layers.get_mut(layer).ok_or_else(|| {
            XrtError::Runtime(format!("layer {layer} is out of range for KV cache"))
        })?;

        // Pre-allocate pages for all new tokens
        let final_len = layer_data.len + count;
        let pages_needed = (final_len + self.page_tokens - 1) / self.page_tokens;
        while layer_data.pages.len() < pages_needed {
            layer_data.pages.push(KvPage::new(self.width, self.page_tokens));
        }

        // Copy all K,V vectors into pages
        for i in 0..count {
            let position = layer_data.len + i;
            let page_index = position / self.page_tokens;
            let slot = position % self.page_tokens;
            let page = &mut layer_data.pages[page_index];
            let offset = slot * w;
            page.keys[offset..offset + w].copy_from_slice(&keys[i * w..(i + 1) * w]);
            page.values[offset..offset + w].copy_from_slice(&values[i * w..(i + 1) * w]);
            page.len = page.len.max(slot + 1);
        }
        layer_data.len = final_len;

        Ok(())
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.pages.clear();
            layer.len = 0;
        }
    }
}
