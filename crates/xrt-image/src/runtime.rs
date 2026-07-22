use std::sync::Arc;

use xrt_runtime::GpuResourceManager;

use crate::{
    pipeline::{ImageRequest, SharedImagePipeline},
    ImageBackendKind, ImageBatchResult, ImageCancellation, ImageCapability, ImageEditRequest,
    ImageError, ImageGenerationRequest, ImageModelBundle, ImageProgressSink,
};

pub struct ImageRuntime {
    pipeline: SharedImagePipeline,
    resources: Arc<GpuResourceManager>,
}

impl std::fmt::Debug for ImageRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImageRuntime")
            .field("capabilities", &self.pipeline.capabilities())
            .field("device_ordinal", &self.resources.config().device_ordinal)
            .finish_non_exhaustive()
    }
}

impl ImageRuntime {
    pub fn load(
        bundle: ImageModelBundle,
        backend: ImageBackendKind,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self, ImageError> {
        #[cfg(feature = "test-util")]
        if bundle.manifest().family == "xrt-synthetic-image" {
            let pipeline = crate::synthetic::SyntheticPipeline::new(bundle, backend)?;
            return Ok(Self {
                pipeline: Arc::new(pipeline),
                resources,
            });
        }

        if bundle.manifest().family == "qwen-image" {
            let pipeline = crate::models::qwen_image::QwenImagePipeline::new(
                bundle,
                backend,
                Arc::clone(&resources),
            )?;
            return Ok(Self {
                pipeline: Arc::new(pipeline),
                resources,
            });
        }

        if bundle.manifest().family == "qwen-image-edit" {
            let pipeline = crate::models::qwen_image::QwenImageEditPipeline::new(
                bundle,
                backend,
                Arc::clone(&resources),
            )?;
            return Ok(Self {
                pipeline: Arc::new(pipeline),
                resources,
            });
        }

        let _ = (backend, resources);
        Err(ImageError::UnsupportedCapability(format!(
            "image adapter `{}` is not implemented yet",
            bundle.manifest().family
        )))
    }

    pub fn capabilities(&self) -> &[ImageCapability] {
        self.pipeline.capabilities()
    }

    pub fn backend(&self) -> ImageBackendKind {
        self.pipeline.backend()
    }

    pub fn resources(&self) -> &Arc<GpuResourceManager> {
        &self.resources
    }

    pub fn default_edit_dimensions(
        &self,
        images: &[crate::DecodedImage],
    ) -> Result<(u32, u32), ImageError> {
        self.pipeline.default_edit_dimensions(images)
    }

    pub fn plan_generation(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<crate::ImageExecutionPlan, ImageError> {
        self.pipeline
            .plan(&ImageRequest::Generation(request.clone()))
            .map(|plan| plan.public)
    }

    pub fn plan_edit(
        &self,
        request: &ImageEditRequest,
    ) -> Result<crate::ImageExecutionPlan, ImageError> {
        self.pipeline
            .plan(&ImageRequest::Edit(request.clone()))
            .map(|plan| plan.public)
    }

    pub fn generate(
        &self,
        request: ImageGenerationRequest,
        cancellation: ImageCancellation,
        progress: Option<Arc<dyn ImageProgressSink>>,
    ) -> Result<ImageBatchResult, ImageError> {
        let plan = self.pipeline.plan(&ImageRequest::Generation(request))?;
        cancellation.check()?;
        self.pipeline
            .execute(plan, &cancellation, progress.as_deref())
    }

    pub fn edit(
        &self,
        request: ImageEditRequest,
        cancellation: ImageCancellation,
        progress: Option<Arc<dyn ImageProgressSink>>,
    ) -> Result<ImageBatchResult, ImageError> {
        let plan = self.pipeline.plan(&ImageRequest::Edit(request))?;
        cancellation.check()?;
        self.pipeline
            .execute(plan, &cancellation, progress.as_deref())
    }
}
