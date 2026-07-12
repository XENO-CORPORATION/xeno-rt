use xrt_core::{DType, Result, XrtError};
use xrt_gguf::{GgufFile, TensorInfo};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidentTensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub dtype: DType,
    pub rank: usize,
    pub rows: usize,
    pub cols: usize,
    pub numel: usize,
    pub byte_len: usize,
}

impl ResidentTensorInfo {
    fn from_gguf(info: &TensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            dimensions: info.dimensions.clone(),
            dtype: info.dtype,
            rank: info.dimensions.len(),
            rows: info.rows(),
            cols: info.row_len(),
            numel: info.numel(),
            byte_len: info.nbytes,
        }
    }
}

pub(crate) trait ResidentTensorSource: Send + Sync {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo>;

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]>;

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo>;

    fn require_tensor(&self, name: &str) -> Result<ResidentTensorInfo> {
        self.tensor_info(name)
            .ok_or_else(|| XrtError::InvalidTensor(format!("missing tensor `{name}`")))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GgufResidentTensorSource<'a> {
    gguf: &'a GgufFile,
}

impl<'a> GgufResidentTensorSource<'a> {
    pub(crate) fn new(gguf: &'a GgufFile) -> Self {
        Self { gguf }
    }
}

impl ResidentTensorSource for GgufResidentTensorSource<'_> {
    fn tensor_info(&self, name: &str) -> Option<ResidentTensorInfo> {
        self.gguf
            .tensor_info(name)
            .map(ResidentTensorInfo::from_gguf)
    }

    fn tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]> {
        self.gguf.tensor_data(name)
    }

    fn tensor_infos(&self) -> Vec<ResidentTensorInfo> {
        self.gguf
            .tensor_infos()
            .iter()
            .map(ResidentTensorInfo::from_gguf)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_metadata_is_normalized_without_changing_row_geometry() {
        let gguf = TensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            dimensions: vec![64, 3],
            strides: vec![1, 64],
            dtype: DType::Q8_0,
            offset: 128,
            nbytes: 204,
        };

        let info = ResidentTensorInfo::from_gguf(&gguf);
        assert_eq!(info.name, gguf.name);
        assert_eq!(info.dimensions, vec![64, 3]);
        assert_eq!(info.dtype, DType::Q8_0);
        assert_eq!(info.rank, 2);
        assert_eq!(info.rows, 3);
        assert_eq!(info.cols, 64);
        assert_eq!(info.numel, 192);
        assert_eq!(info.byte_len, 204);
    }
}
