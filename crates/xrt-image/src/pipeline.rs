use std::sync::Arc;

use crate::{
    DecodedImage, ImageBatchResult, ImageCancellation, ImageCapability, ImageEditRequest,
    ImageError, ImageExecutionPlan, ImageGenerationRequest, ImageProgressSink,
};

#[derive(Debug, Clone)]
pub(crate) enum ImageRequest {
    Generation(ImageGenerationRequest),
    Edit(ImageEditRequest),
}

pub(crate) struct PipelineExecutionPlan {
    pub public: ImageExecutionPlan,
    pub request: ImageRequest,
}

pub(crate) trait ImagePipeline: Send + Sync {
    fn capabilities(&self) -> &[ImageCapability];

    fn backend(&self) -> crate::ImageBackendKind;

    fn default_edit_dimensions(&self, images: &[DecodedImage]) -> Result<(u32, u32), ImageError> {
        images
            .last()
            .map(|image| (image.width(), image.height()))
            .ok_or_else(|| {
                ImageError::InvalidRequest(
                    "default edit dimensions require at least one source image".to_string(),
                )
            })
    }

    fn plan(&self, request: &ImageRequest) -> Result<PipelineExecutionPlan, ImageError>;

    fn execute(
        &self,
        plan: PipelineExecutionPlan,
        cancellation: &ImageCancellation,
        progress: Option<&dyn ImageProgressSink>,
    ) -> Result<ImageBatchResult, ImageError>;
}

pub(crate) type SharedImagePipeline = Arc<dyn ImagePipeline>;
