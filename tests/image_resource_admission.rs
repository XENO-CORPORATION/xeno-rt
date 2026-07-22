#![cfg(feature = "image-generation-tests")]

mod common;

use std::sync::Arc;

use common::{build_synthetic_llama_fixture, SyntheticLlamaSpec};
use xrt_image::{synthetic_bundle_for_tests, ImageBackendKind, ImageRuntime};
use xrt_runtime::{BackendKind, GpuResourceConfig, GpuResourceManager, MoeRuntimeConfig, Runtime};

#[test]
fn text_and_image_runtimes_accept_the_same_device_resource_manager() {
    let fixture = build_synthetic_llama_fixture(SyntheticLlamaSpec::tiny()).unwrap();
    let config = GpuResourceConfig::default();
    let manager = Arc::new(GpuResourceManager::new(config));
    let text = Runtime::load_with_backend_configs_and_resource_manager(
        fixture.path(),
        BackendKind::Cpu,
        MoeRuntimeConfig::default(),
        config,
        Arc::clone(&manager),
    )
    .unwrap();
    let image = ImageRuntime::load(
        synthetic_bundle_for_tests(),
        ImageBackendKind::Cpu,
        Arc::clone(&manager),
    )
    .unwrap();
    assert!(Arc::ptr_eq(&text.gpu_resource_manager(), &manager));
    assert!(Arc::ptr_eq(image.resources(), &manager));
}

#[test]
fn injected_text_runtime_rejects_a_mismatched_manager_configuration() {
    let fixture = build_synthetic_llama_fixture(SyntheticLlamaSpec::tiny()).unwrap();
    let requested = GpuResourceConfig::default();
    let mut actual = requested;
    actual.device_ordinal = 1;
    let error = Runtime::load_with_backend_configs_and_resource_manager(
        fixture.path(),
        BackendKind::Cpu,
        MoeRuntimeConfig::default(),
        requested,
        Arc::new(GpuResourceManager::new(actual)),
    )
    .err()
    .expect("mismatched shared manager must fail before loading");
    assert!(error.to_string().contains("configuration mismatch"));
}
