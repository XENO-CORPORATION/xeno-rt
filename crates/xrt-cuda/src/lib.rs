use xrt_core::{DType, Result, XrtError};
use xrt_gguf::GgufFile;

#[cfg(not(feature = "cuda"))]
const CUDA_DISABLED_MESSAGE: &str =
    "CUDA backend requested but the xrt-cuda crate was built without the `cuda` feature";

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use core::ffi::c_void;
    use cudarc::{
        driver::{
            result as driver_result, sys, CudaDevice as DriverCudaDevice, CudaFunction, CudaSlice,
            DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig,
        },
        nvrtc::Ptx,
    };
    use std::{ffi::CString, fmt::Display, mem::MaybeUninit, sync::Arc};
    use tracing::info;
    use xrt_core::{checked_mul, decode_bf16, decode_f16};
    use xrt_kernels::cpu::{dequantize_q4_k_row, dequantize_q5_k_row};

    const BLOCK_SIZE: u32 = 256;
    const MATMUL_TILE: u32 = 16;

    #[derive(Debug, Clone, Copy)]
    struct LoadedModules {
        rmsnorm: &'static str,
        rope: &'static str,
        softmax: &'static str,
        silu: &'static str,
        matmul: &'static str,
        q8_0_matvec: &'static str,
        q4_k_matvec: &'static str,
        q6_k_matvec: &'static str,
        add: &'static str,
        mul: &'static str,
        activation: &'static str,
        repeat_kv: &'static str,
        attention: &'static str,
        embed: &'static str,
    }

    const MODULES: LoadedModules = LoadedModules {
        rmsnorm: "xrt_cuda_rmsnorm",
        rope: "xrt_cuda_rope",
        softmax: "xrt_cuda_softmax",
        silu: "xrt_cuda_silu",
        matmul: "xrt_cuda_matmul",
        q8_0_matvec: "xrt_cuda_q8_0_matvec",
        q4_k_matvec: "xrt_cuda_q4_k_matvec",
        q6_k_matvec: "xrt_cuda_q6_k_matvec",
        add: "xrt_cuda_add",
        mul: "xrt_cuda_mul",
        activation: "xrt_cuda_activation",
        repeat_kv: "xrt_cuda_repeat_kv",
        attention: "xrt_cuda_attention",
        embed: "xrt_cuda_embed",
    };

    // ponytail: scalar row kernel for correctness; replace with block reduction when RMSNorm perf matters.
    const RMSNORM_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry rmsnorm_kernel(
    .param .u64 rmsnorm_kernel_param_0,
    .param .u64 rmsnorm_kernel_param_1,
    .param .u64 rmsnorm_kernel_param_2,
    .param .u32 rmsnorm_kernel_param_3,
    .param .u32 rmsnorm_kernel_param_4,
    .param .f32 rmsnorm_kernel_param_5
)
{
    .reg .pred %p<6>;
    .reg .f32 %f<12>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;

    ld.param.u64 %rd1, [rmsnorm_kernel_param_0];
    ld.param.u64 %rd2, [rmsnorm_kernel_param_1];
    ld.param.u64 %rd3, [rmsnorm_kernel_param_2];
    ld.param.u32 %r1, [rmsnorm_kernel_param_3];
    ld.param.u32 %r2, [rmsnorm_kernel_param_4];
    ld.param.f32 %f1, [rmsnorm_kernel_param_5];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra RMS_DONE;

    mov.u32 %r4, %tid.x;
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra RMS_DONE;

    mul.lo.u32 %r5, %r3, %r2;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r6, 0;

RMS_SUM:
    setp.ge.u32 %p3, %r6, %r2;
    @%p3 bra RMS_SCALE;
    add.u32 %r7, %r5, %r6;
    mul.wide.u32 %rd7, %r7, 4;
    add.s64 %rd8, %rd4, %rd7;
    ld.global.f32 %f3, [%rd8];
    fma.rn.f32 %f2, %f3, %f3, %f2;
    add.u32 %r6, %r6, 1;
    bra RMS_SUM;

RMS_SCALE:
    cvt.rn.f32.u32 %f4, %r2;
    div.rn.f32 %f5, %f2, %f4;
    add.f32 %f6, %f5, %f1;
    sqrt.rn.f32 %f7, %f6;
    mov.f32 %f8, 0f3f800000;
    div.rn.f32 %f9, %f8, %f7;
    mov.u32 %r6, 0;

RMS_WRITE:
    setp.ge.u32 %p4, %r6, %r2;
    @%p4 bra RMS_DONE;
    add.u32 %r7, %r5, %r6;
    mul.wide.u32 %rd7, %r7, 4;
    add.s64 %rd8, %rd4, %rd7;
    add.s64 %rd9, %rd6, %rd7;
    mul.wide.u32 %rd10, %r6, 4;
    add.s64 %rd11, %rd5, %rd10;
    ld.global.f32 %f3, [%rd8];
    ld.global.f32 %f10, [%rd11];
    mul.f32 %f11, %f3, %f9;
    mul.f32 %f11, %f11, %f10;
    st.global.f32 [%rd9], %f11;
    add.u32 %r6, %r6, 1;
    bra RMS_WRITE;

RMS_DONE:
    ret;
}

.visible .entry rmsnorm_unweighted_kernel(
    .param .u64 rmsnorm_unweighted_kernel_param_0,
    .param .u64 rmsnorm_unweighted_kernel_param_1,
    .param .u32 rmsnorm_unweighted_kernel_param_2,
    .param .u32 rmsnorm_unweighted_kernel_param_3,
    .param .f32 rmsnorm_unweighted_kernel_param_4
)
{
    .reg .pred %p<6>;
    .reg .f32 %f<12>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;

    ld.param.u64 %rd1, [rmsnorm_unweighted_kernel_param_0];
    ld.param.u64 %rd2, [rmsnorm_unweighted_kernel_param_1];
    ld.param.u32 %r1, [rmsnorm_unweighted_kernel_param_2];
    ld.param.u32 %r2, [rmsnorm_unweighted_kernel_param_3];
    ld.param.f32 %f1, [rmsnorm_unweighted_kernel_param_4];

    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra RMS_UNWEIGHTED_DONE;

    mov.u32 %r4, %tid.x;
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra RMS_UNWEIGHTED_DONE;

    mul.lo.u32 %r5, %r3, %r2;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r6, 0;

RMS_UNWEIGHTED_SUM:
    setp.ge.u32 %p3, %r6, %r2;
    @%p3 bra RMS_UNWEIGHTED_SCALE;
    add.u32 %r7, %r5, %r6;
    mul.wide.u32 %rd5, %r7, 4;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.f32 %f3, [%rd6];
    fma.rn.f32 %f2, %f3, %f3, %f2;
    add.u32 %r6, %r6, 1;
    bra RMS_UNWEIGHTED_SUM;

RMS_UNWEIGHTED_SCALE:
    cvt.rn.f32.u32 %f4, %r2;
    div.rn.f32 %f5, %f2, %f4;
    add.f32 %f6, %f5, %f1;
    sqrt.rn.f32 %f7, %f6;
    mov.f32 %f8, 0f3f800000;
    div.rn.f32 %f9, %f8, %f7;
    mov.u32 %r6, 0;

RMS_UNWEIGHTED_WRITE:
    setp.ge.u32 %p4, %r6, %r2;
    @%p4 bra RMS_UNWEIGHTED_DONE;
    add.u32 %r7, %r5, %r6;
    mul.wide.u32 %rd5, %r7, 4;
    add.s64 %rd6, %rd3, %rd5;
    add.s64 %rd7, %rd4, %rd5;
    ld.global.f32 %f3, [%rd6];
    mul.f32 %f10, %f3, %f9;
    st.global.f32 [%rd7], %f10;
    add.u32 %r6, %r6, 1;
    bra RMS_UNWEIGHTED_WRITE;

RMS_UNWEIGHTED_DONE:
    ret;
}
"#;
    const ROPE_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry rope_kernel(
    .param .u64 rope_kernel_param_0,
    .param .u32 rope_kernel_param_1,
    .param .u32 rope_kernel_param_2,
    .param .u32 rope_kernel_param_3,
    .param .u32 rope_kernel_param_4,
    .param .f32 rope_kernel_param_5,
    .param .f32 rope_kernel_param_6
)
{
    .reg .pred %p<8>;
    .reg .f32 %f<16>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [rope_kernel_param_0];
    ld.param.u32 %r1, [rope_kernel_param_1];
    ld.param.u32 %r2, [rope_kernel_param_2];
    ld.param.u32 %r3, [rope_kernel_param_3];
    ld.param.u32 %r4, [rope_kernel_param_4];
    ld.param.f32 %f1, [rope_kernel_param_5];
    ld.param.f32 %f2, [rope_kernel_param_6];

    cvta.to.global.u64 %rd2, %rd1;

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mad.lo.s32 %r8, %r6, %r7, %r5;

    shr.u32 %r9, %r4, 1;
    mul.lo.u32 %r10, %r1, %r9;
    setp.ge.u32 %p1, %r8, %r10;
    @%p1 bra ROPE_DONE;

    div.u32 %r11, %r8, %r9;
    mul.lo.u32 %r12, %r11, %r9;
    sub.u32 %r13, %r8, %r12;

    mul.lo.u32 %r14, %r11, %r2;
    add.u32 %r15, %r14, %r13;
    add.u32 %r16, %r15, %r9;

    mul.wide.u32 %rd3, %r15, 4;
    mul.wide.u32 %rd4, %r16, 4;
    add.s64 %rd5, %rd2, %rd3;
    add.s64 %rd6, %rd2, %rd4;

    ld.global.f32 %f3, [%rd5];
    ld.global.f32 %f4, [%rd6];

    cvt.rn.f32.u32 %f5, %r13;
    mov.f32 %f6, 0f40000000;
    mul.f32 %f7, %f5, %f6;
    cvt.rn.f32.u32 %f8, %r4;
    div.rn.f32 %f9, %f7, %f8;
    neg.f32 %f10, %f9;
    lg2.approx.f32 %f11, %f1;
    mul.f32 %f12, %f10, %f11;
    ex2.approx.f32 %f13, %f12;
    cvt.rn.f32.u32 %f14, %r3;
    mul.f32 %f15, %f14, %f2;
    mul.f32 %f5, %f15, %f13;
    sin.approx.f32 %f6, %f5;
    cos.approx.f32 %f7, %f5;

    mul.f32 %f8, %f3, %f7;
    mul.f32 %f9, %f4, %f6;
    sub.f32 %f10, %f8, %f9;
    mul.f32 %f11, %f3, %f6;
    mul.f32 %f12, %f4, %f7;
    add.f32 %f13, %f11, %f12;

    st.global.f32 [%rd5], %f10;
    st.global.f32 [%rd6], %f13;

ROPE_DONE:
    ret;
}
"#;
    const SOFTMAX_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry softmax_kernel(
    .param .u64 softmax_kernel_param_0,
    .param .u32 softmax_kernel_param_1,
    .param .u32 softmax_kernel_param_2
)
{
    .shared .align 4 .b8 reduce_buf[1024];
    .reg .pred %p<18>;
    .reg .f32 %f<24>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<20>;

    ld.param.u64 %rd1, [softmax_kernel_param_0];
    ld.param.u32 %r1, [softmax_kernel_param_1];
    ld.param.u32 %r2, [softmax_kernel_param_2];

    cvta.to.global.u64 %rd2, %rd1;
    cvta.to.shared.u64 %rd3, reduce_buf;

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra SOFTMAX_DONE;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r6, %r3, %r2;
    mul.wide.u32 %rd4, %r4, 4;
    add.s64 %rd5, %rd3, %rd4;

    mov.f32 %f1, 0fFF800000;
    mov.u32 %r7, %r4;

SOFTMAX_MAX_LOOP:
    setp.ge.u32 %p2, %r7, %r2;
    @%p2 bra SOFTMAX_MAX_DONE;
    add.u32 %r8, %r6, %r7;
    mul.wide.u32 %rd6, %r8, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd7];
    max.f32 %f1, %f1, %f2;
    add.u32 %r7, %r7, %r5;
    bra SOFTMAX_MAX_LOOP;

SOFTMAX_MAX_DONE:
    st.shared.f32 [%rd5], %f1;
    bar.sync 0;

    setp.ge.u32 %p3, %r4, 128;
    @%p3 bra SOFTMAX_MAX_128_DONE;
    add.u32 %r9, %r4, 128;
    mul.wide.u32 %rd8, %r9, 4;
    add.s64 %rd9, %rd3, %rd8;
    ld.shared.f32 %f3, [%rd5];
    ld.shared.f32 %f4, [%rd9];
    max.f32 %f5, %f3, %f4;
    st.shared.f32 [%rd5], %f5;
SOFTMAX_MAX_128_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 64;
    @%p4 bra SOFTMAX_MAX_64_DONE;
    add.u32 %r10, %r4, 64;
    mul.wide.u32 %rd10, %r10, 4;
    add.s64 %rd11, %rd3, %rd10;
    ld.shared.f32 %f6, [%rd5];
    ld.shared.f32 %f7, [%rd11];
    max.f32 %f8, %f6, %f7;
    st.shared.f32 [%rd5], %f8;
SOFTMAX_MAX_64_DONE:
    bar.sync 0;

    setp.ge.u32 %p5, %r4, 32;
    @%p5 bra SOFTMAX_MAX_32_DONE;
    add.u32 %r11, %r4, 32;
    mul.wide.u32 %rd12, %r11, 4;
    add.s64 %rd13, %rd3, %rd12;
    ld.shared.f32 %f9, [%rd5];
    ld.shared.f32 %f10, [%rd13];
    max.f32 %f11, %f9, %f10;
    st.shared.f32 [%rd5], %f11;
SOFTMAX_MAX_32_DONE:
    bar.sync 0;

    setp.ge.u32 %p6, %r4, 16;
    @%p6 bra SOFTMAX_MAX_16_DONE;
    add.u32 %r12, %r4, 16;
    mul.wide.u32 %rd14, %r12, 4;
    add.s64 %rd15, %rd3, %rd14;
    ld.shared.f32 %f12, [%rd5];
    ld.shared.f32 %f13, [%rd15];
    max.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd5], %f14;
SOFTMAX_MAX_16_DONE:
    bar.sync 0;

    setp.ge.u32 %p7, %r4, 8;
    @%p7 bra SOFTMAX_MAX_8_DONE;
    add.u32 %r13, %r4, 8;
    mul.wide.u32 %rd16, %r13, 4;
    add.s64 %rd17, %rd3, %rd16;
    ld.shared.f32 %f15, [%rd5];
    ld.shared.f32 %f16, [%rd17];
    max.f32 %f17, %f15, %f16;
    st.shared.f32 [%rd5], %f17;
SOFTMAX_MAX_8_DONE:
    bar.sync 0;

    setp.ge.u32 %p8, %r4, 4;
    @%p8 bra SOFTMAX_MAX_4_DONE;
    add.u32 %r14, %r4, 4;
    mul.wide.u32 %rd18, %r14, 4;
    add.s64 %rd19, %rd3, %rd18;
    ld.shared.f32 %f18, [%rd5];
    ld.shared.f32 %f19, [%rd19];
    max.f32 %f20, %f18, %f19;
    st.shared.f32 [%rd5], %f20;
SOFTMAX_MAX_4_DONE:
    bar.sync 0;

    setp.ge.u32 %p9, %r4, 2;
    @%p9 bra SOFTMAX_MAX_2_DONE;
    add.u32 %r15, %r4, 2;
    mul.wide.u32 %rd8, %r15, 4;
    add.s64 %rd9, %rd3, %rd8;
    ld.shared.f32 %f3, [%rd5];
    ld.shared.f32 %f4, [%rd9];
    max.f32 %f5, %f3, %f4;
    st.shared.f32 [%rd5], %f5;
SOFTMAX_MAX_2_DONE:
    bar.sync 0;

    setp.ge.u32 %p10, %r4, 1;
    @%p10 bra SOFTMAX_MAX_1_DONE;
    add.u32 %r16, %r4, 1;
    mul.wide.u32 %rd10, %r16, 4;
    add.s64 %rd11, %rd3, %rd10;
    ld.shared.f32 %f6, [%rd5];
    ld.shared.f32 %f7, [%rd11];
    max.f32 %f8, %f6, %f7;
    st.shared.f32 [%rd5], %f8;
SOFTMAX_MAX_1_DONE:
    bar.sync 0;

    ld.shared.f32 %f21, [%rd3];
    mov.f32 %f1, 0f00000000;
    mov.u32 %r7, %r4;

SOFTMAX_EXP_LOOP:
    setp.ge.u32 %p11, %r7, %r2;
    @%p11 bra SOFTMAX_EXP_DONE;
    add.u32 %r8, %r6, %r7;
    mul.wide.u32 %rd6, %r8, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd7];
    sub.f32 %f3, %f2, %f21;
    mul.f32 %f4, %f3, 0f3FB8AA3B;
    ex2.approx.f32 %f5, %f4;
    st.global.f32 [%rd7], %f5;
    add.f32 %f1, %f1, %f5;
    add.u32 %r7, %r7, %r5;
    bra SOFTMAX_EXP_LOOP;

SOFTMAX_EXP_DONE:
    st.shared.f32 [%rd5], %f1;
    bar.sync 0;

    setp.ge.u32 %p12, %r4, 128;
    @%p12 bra SOFTMAX_SUM_128_DONE;
    add.u32 %r9, %r4, 128;
    mul.wide.u32 %rd8, %r9, 4;
    add.s64 %rd9, %rd3, %rd8;
    ld.shared.f32 %f6, [%rd5];
    ld.shared.f32 %f7, [%rd9];
    add.f32 %f8, %f6, %f7;
    st.shared.f32 [%rd5], %f8;
SOFTMAX_SUM_128_DONE:
    bar.sync 0;

    setp.ge.u32 %p13, %r4, 64;
    @%p13 bra SOFTMAX_SUM_64_DONE;
    add.u32 %r10, %r4, 64;
    mul.wide.u32 %rd10, %r10, 4;
    add.s64 %rd11, %rd3, %rd10;
    ld.shared.f32 %f9, [%rd5];
    ld.shared.f32 %f10, [%rd11];
    add.f32 %f11, %f9, %f10;
    st.shared.f32 [%rd5], %f11;
SOFTMAX_SUM_64_DONE:
    bar.sync 0;

    setp.ge.u32 %p14, %r4, 32;
    @%p14 bra SOFTMAX_SUM_32_DONE;
    add.u32 %r11, %r4, 32;
    mul.wide.u32 %rd12, %r11, 4;
    add.s64 %rd13, %rd3, %rd12;
    ld.shared.f32 %f12, [%rd5];
    ld.shared.f32 %f13, [%rd13];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd5], %f14;
SOFTMAX_SUM_32_DONE:
    bar.sync 0;

    setp.ge.u32 %p15, %r4, 16;
    @%p15 bra SOFTMAX_SUM_16_DONE;
    add.u32 %r12, %r4, 16;
    mul.wide.u32 %rd14, %r12, 4;
    add.s64 %rd15, %rd3, %rd14;
    ld.shared.f32 %f15, [%rd5];
    ld.shared.f32 %f16, [%rd15];
    add.f32 %f17, %f15, %f16;
    st.shared.f32 [%rd5], %f17;
SOFTMAX_SUM_16_DONE:
    bar.sync 0;

    setp.ge.u32 %p16, %r4, 8;
    @%p16 bra SOFTMAX_SUM_8_DONE;
    add.u32 %r13, %r4, 8;
    mul.wide.u32 %rd16, %r13, 4;
    add.s64 %rd17, %rd3, %rd16;
    ld.shared.f32 %f18, [%rd5];
    ld.shared.f32 %f19, [%rd17];
    add.f32 %f20, %f18, %f19;
    st.shared.f32 [%rd5], %f20;
SOFTMAX_SUM_8_DONE:
    bar.sync 0;

    setp.ge.u32 %p17, %r4, 4;
    @%p17 bra SOFTMAX_SUM_4_DONE;
    add.u32 %r14, %r4, 4;
    mul.wide.u32 %rd18, %r14, 4;
    add.s64 %rd19, %rd3, %rd18;
    ld.shared.f32 %f3, [%rd5];
    ld.shared.f32 %f4, [%rd19];
    add.f32 %f5, %f3, %f4;
    st.shared.f32 [%rd5], %f5;
SOFTMAX_SUM_4_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 2;
    @%p2 bra SOFTMAX_SUM_2_DONE;
    add.u32 %r15, %r4, 2;
    mul.wide.u32 %rd8, %r15, 4;
    add.s64 %rd9, %rd3, %rd8;
    ld.shared.f32 %f6, [%rd5];
    ld.shared.f32 %f7, [%rd9];
    add.f32 %f8, %f6, %f7;
    st.shared.f32 [%rd5], %f8;
SOFTMAX_SUM_2_DONE:
    bar.sync 0;

    setp.ge.u32 %p3, %r4, 1;
    @%p3 bra SOFTMAX_SUM_1_DONE;
    add.u32 %r16, %r4, 1;
    mul.wide.u32 %rd10, %r16, 4;
    add.s64 %rd11, %rd3, %rd10;
    ld.shared.f32 %f9, [%rd5];
    ld.shared.f32 %f10, [%rd11];
    add.f32 %f11, %f9, %f10;
    st.shared.f32 [%rd5], %f11;
SOFTMAX_SUM_1_DONE:
    bar.sync 0;

    ld.shared.f32 %f22, [%rd3];
    mov.u32 %r7, %r4;

SOFTMAX_NORM_LOOP:
    setp.ge.u32 %p4, %r7, %r2;
    @%p4 bra SOFTMAX_DONE;
    add.u32 %r8, %r6, %r7;
    mul.wide.u32 %rd6, %r8, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f12, [%rd7];
    div.rn.f32 %f13, %f12, %f22;
    st.global.f32 [%rd7], %f13;
    add.u32 %r7, %r7, %r5;
    bra SOFTMAX_NORM_LOOP;

SOFTMAX_DONE:
    ret;
}
"#;
    const SILU_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry silu_kernel(
    .param .u64 silu_kernel_param_0,
    .param .u32 silu_kernel_param_1
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<8>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [silu_kernel_param_0];
    ld.param.u32 %r1, [silu_kernel_param_1];
    cvta.to.global.u64 %rd2, %rd1;

    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra SILU_DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    neg.f32 %f2, %f1;
    mul.f32 %f3, %f2, 0f3FB8AA3B;
    ex2.approx.f32 %f4, %f3;
    add.f32 %f5, %f4, 0f3F800000;
    div.rn.f32 %f6, %f1, %f5;
    st.global.f32 [%rd4], %f6;

SILU_DONE:
    ret;
}
"#;
    // ponytail: one thread per output keeps F32 probe reliable; restore tiling after CUDA decode parity is broader.
    const MATMUL_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry matmul_kernel(
    .param .u64 matmul_kernel_param_0,
    .param .u64 matmul_kernel_param_1,
    .param .u64 matmul_kernel_param_2,
    .param .u32 matmul_kernel_param_3,
    .param .u32 matmul_kernel_param_4,
    .param .u32 matmul_kernel_param_5
)
{
    .reg .pred %p<6>;
    .reg .f32 %f<6>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<16>;

    ld.param.u64 %rd1, [matmul_kernel_param_0];
    ld.param.u64 %rd2, [matmul_kernel_param_1];
    ld.param.u64 %rd3, [matmul_kernel_param_2];
    ld.param.u32 %r1, [matmul_kernel_param_3];
    ld.param.u32 %r2, [matmul_kernel_param_4];
    ld.param.u32 %r3, [matmul_kernel_param_5];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ctaid.y;
    mov.u32 %r6, %ntid.x;
    mov.u32 %r7, %ntid.y;
    mov.u32 %r8, %tid.x;
    mov.u32 %r9, %tid.y;
    mad.lo.u32 %r10, %r4, %r6, %r8;
    mad.lo.u32 %r11, %r5, %r7, %r9;

    setp.ge.u32 %p1, %r11, %r1;
    setp.ge.u32 %p2, %r10, %r3;
    or.pred %p3, %p1, %p2;
    @%p3 bra MATMUL_DONE;

    mov.u32 %r12, 0;
    mov.f32 %f1, 0f00000000;

MATMUL_LOOP:
    setp.ge.u32 %p4, %r12, %r2;
    @%p4 bra MATMUL_STORE;

    mul.lo.u32 %r13, %r11, %r2;
    add.u32 %r14, %r13, %r12;
    mul.wide.u32 %rd7, %r14, 4;
    add.s64 %rd8, %rd4, %rd7;

    mul.lo.u32 %r15, %r12, %r3;
    add.u32 %r16, %r15, %r10;
    mul.wide.u32 %rd9, %r16, 4;
    add.s64 %rd10, %rd5, %rd9;

    ld.global.f32 %f2, [%rd8];
    ld.global.f32 %f3, [%rd10];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r12, %r12, 1;
    bra MATMUL_LOOP;

MATMUL_STORE:
    mul.lo.u32 %r17, %r11, %r3;
    add.u32 %r18, %r17, %r10;
    mul.wide.u32 %rd11, %r18, 4;
    add.s64 %rd12, %rd6, %rd11;
    st.global.f32 [%rd12], %f1;

MATMUL_DONE:
    ret;
}
"#;
    const ADD_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry elementwise_add_kernel(
    .param .u64 elementwise_add_kernel_param_0,
    .param .u64 elementwise_add_kernel_param_1,
    .param .u32 elementwise_add_kernel_param_2
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<6>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [elementwise_add_kernel_param_0];
    ld.param.u64 %rd2, [elementwise_add_kernel_param_1];
    ld.param.u32 %r1, [elementwise_add_kernel_param_2];

    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;

    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra ADD_DONE;

    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd3, %rd5;
    add.s64 %rd7, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd6], %f3;

ADD_DONE:
    ret;
}
"#;
    const MUL_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry elementwise_mul_kernel(
    .param .u64 elementwise_mul_kernel_param_0,
    .param .u64 elementwise_mul_kernel_param_1,
    .param .u32 elementwise_mul_kernel_param_2
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<6>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [elementwise_mul_kernel_param_0];
    ld.param.u64 %rd2, [elementwise_mul_kernel_param_1];
    ld.param.u32 %r1, [elementwise_mul_kernel_param_2];

    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;

    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra MUL_DONE;

    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd3, %rd5;
    add.s64 %rd7, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    mul.f32 %f3, %f1, %f2;
    st.global.f32 [%rd6], %f3;

MUL_DONE:
    ret;
}
"#;
    const ACTIVATION_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry scale_assign_kernel(
    .param .u64 scale_assign_kernel_param_0,
    .param .u32 scale_assign_kernel_param_1,
    .param .f32 scale_assign_kernel_param_2
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [scale_assign_kernel_param_0];
    ld.param.u32 %r1, [scale_assign_kernel_param_1];
    ld.param.f32 %f1, [scale_assign_kernel_param_2];
    cvta.to.global.u64 %rd2, %rd1;

    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra SCALE_ASSIGN_DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.f32 %f2, [%rd4];
    mul.f32 %f3, %f2, %f1;
    st.global.f32 [%rd4], %f3;

SCALE_ASSIGN_DONE:
    ret;
}

.visible .entry geglu_pytorch_tanh_kernel(
    .param .u64 geglu_pytorch_tanh_kernel_param_0,
    .param .u64 geglu_pytorch_tanh_kernel_param_1,
    .param .u32 geglu_pytorch_tanh_kernel_param_2
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<24>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [geglu_pytorch_tanh_kernel_param_0];
    ld.param.u64 %rd2, [geglu_pytorch_tanh_kernel_param_1];
    ld.param.u32 %r1, [geglu_pytorch_tanh_kernel_param_2];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;

    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra GEGLU_TANH_DONE;

    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd3, %rd5;
    add.s64 %rd7, %rd4, %rd5;
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];

    mul.f32 %f3, %f1, %f1;
    mul.f32 %f4, %f3, %f1;
    mul.f32 %f5, %f4, 0f3d372713;
    add.f32 %f6, %f1, %f5;
    mul.f32 %f7, %f6, 0f3f4c422a;

    mul.f32 %f8, %f7, 0fc038aa3b;
    ex2.approx.f32 %f9, %f8;
    add.f32 %f10, %f9, 0f3f800000;
    div.rn.f32 %f11, 0f40000000, %f10;
    sub.f32 %f12, %f11, 0f3f800000;

    add.f32 %f13, %f12, 0f3f800000;
    mul.f32 %f14, %f1, 0f3f000000;
    mul.f32 %f15, %f14, %f13;
    mul.f32 %f16, %f15, %f2;
    st.global.f32 [%rd6], %f16;

GEGLU_TANH_DONE:
    ret;
}

.visible .entry logit_softcap_assign_kernel(
    .param .u64 logit_softcap_assign_kernel_param_0,
    .param .u32 logit_softcap_assign_kernel_param_1,
    .param .f32 logit_softcap_assign_kernel_param_2
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<12>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [logit_softcap_assign_kernel_param_0];
    ld.param.u32 %r1, [logit_softcap_assign_kernel_param_1];
    ld.param.f32 %f1, [logit_softcap_assign_kernel_param_2];
    cvta.to.global.u64 %rd2, %rd1;

    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra LOGIT_SOFTCAP_DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.f32 %f2, [%rd4];
    div.rn.f32 %f3, %f2, %f1;
    mul.f32 %f4, %f3, 0fc038aa3b;
    ex2.approx.f32 %f5, %f4;
    add.f32 %f6, %f5, 0f3f800000;
    div.rn.f32 %f7, 0f40000000, %f6;
    sub.f32 %f8, %f7, 0f3f800000;
    mul.f32 %f9, %f8, %f1;
    st.global.f32 [%rd4], %f9;

LOGIT_SOFTCAP_DONE:
    ret;
}
"#;
    const REPEAT_KV_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry repeat_kv_kernel(
    .param .u64 repeat_kv_kernel_param_0,
    .param .u64 repeat_kv_kernel_param_1,
    .param .u32 repeat_kv_kernel_param_2,
    .param .u32 repeat_kv_kernel_param_3,
    .param .u32 repeat_kv_kernel_param_4,
    .param .u32 repeat_kv_kernel_param_5
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [repeat_kv_kernel_param_0];
    ld.param.u64 %rd2, [repeat_kv_kernel_param_1];
    ld.param.u32 %r1, [repeat_kv_kernel_param_2];
    ld.param.u32 %r2, [repeat_kv_kernel_param_3];
    ld.param.u32 %r3, [repeat_kv_kernel_param_4];
    ld.param.u32 %r4, [repeat_kv_kernel_param_5];

    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;

    mov.u32 %r5, %tid.x;
    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mad.lo.u32 %r8, %r6, %r7, %r5;
    setp.ge.u32 %p1, %r8, %r4;
    @%p1 bra REPEAT_KV_DONE;

    div.u32 %r9, %r8, %r3;
    mul.lo.u32 %r10, %r9, %r3;
    sub.u32 %r11, %r8, %r10;
    div.u32 %r12, %r1, %r2;
    div.u32 %r13, %r9, %r12;
    mul.lo.u32 %r14, %r13, %r3;
    add.u32 %r15, %r14, %r11;

    mul.wide.u32 %rd5, %r15, 4;
    add.s64 %rd6, %rd3, %rd5;
    mul.wide.u32 %rd7, %r8, 4;
    add.s64 %rd8, %rd4, %rd7;
    ld.global.f32 %f1, [%rd6];
    st.global.f32 [%rd8], %f1;

REPEAT_KV_DONE:
    ret;
}
"#;
    const ATTENTION_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry adaptive_kv_route_write_kernel(
    .param .u64 adaptive_kv_route_write_kernel_param_0,
    .param .u32 adaptive_kv_route_write_kernel_param_1,
    .param .u32 adaptive_kv_route_write_kernel_param_2
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [adaptive_kv_route_write_kernel_param_0];
    ld.param.u32 %r1, [adaptive_kv_route_write_kernel_param_1];
    ld.param.u32 %r2, [adaptive_kv_route_write_kernel_param_2];
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r2;
    ret;
}

.visible .entry kv_cache_append_kernel(
    .param .u64 kv_cache_append_kernel_param_0,
    .param .u64 kv_cache_append_kernel_param_1,
    .param .u64 kv_cache_append_kernel_param_2,
    .param .u64 kv_cache_append_kernel_param_3,
    .param .u32 kv_cache_append_kernel_param_4,
    .param .u32 kv_cache_append_kernel_param_5
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [kv_cache_append_kernel_param_0];
    ld.param.u64 %rd2, [kv_cache_append_kernel_param_1];
    ld.param.u64 %rd3, [kv_cache_append_kernel_param_2];
    ld.param.u64 %rd4, [kv_cache_append_kernel_param_3];
    ld.param.u32 %r1, [kv_cache_append_kernel_param_4];
    ld.param.u32 %r2, [kv_cache_append_kernel_param_5];

    cvta.to.global.u64 %rd5, %rd1;
    cvta.to.global.u64 %rd6, %rd2;
    cvta.to.global.u64 %rd7, %rd3;
    cvta.to.global.u64 %rd8, %rd4;

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra KV_APPEND_DONE;

    mad.lo.u32 %r7, %r1, %r2, %r6;
    mul.wide.u32 %rd9, %r6, 4;
    mul.wide.u32 %rd10, %r7, 4;
    add.s64 %rd11, %rd7, %rd9;
    add.s64 %rd12, %rd5, %rd10;
    ld.global.f32 %f1, [%rd11];
    st.global.f32 [%rd12], %f1;
    add.s64 %rd11, %rd8, %rd9;
    add.s64 %rd13, %rd6, %rd10;
    ld.global.f32 %f2, [%rd11];
    st.global.f32 [%rd13], %f2;

KV_APPEND_DONE:
    ret;
}

.visible .entry paged_kv_cache_append_kernel(
    .param .u64 paged_kv_cache_append_kernel_param_0,
    .param .u64 paged_kv_cache_append_kernel_param_1,
    .param .u64 paged_kv_cache_append_kernel_param_2,
    .param .u64 paged_kv_cache_append_kernel_param_3,
    .param .u64 paged_kv_cache_append_kernel_param_4,
    .param .u32 paged_kv_cache_append_kernel_param_5,
    .param .u32 paged_kv_cache_append_kernel_param_6,
    .param .u32 paged_kv_cache_append_kernel_param_7
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<18>;

    ld.param.u64 %rd1, [paged_kv_cache_append_kernel_param_0];
    ld.param.u64 %rd2, [paged_kv_cache_append_kernel_param_1];
    ld.param.u64 %rd3, [paged_kv_cache_append_kernel_param_2];
    ld.param.u64 %rd4, [paged_kv_cache_append_kernel_param_3];
    ld.param.u64 %rd5, [paged_kv_cache_append_kernel_param_4];
    ld.param.u32 %r1, [paged_kv_cache_append_kernel_param_5];
    ld.param.u32 %r2, [paged_kv_cache_append_kernel_param_6];
    ld.param.u32 %r3, [paged_kv_cache_append_kernel_param_7];

    cvta.to.global.u64 %rd6, %rd1;
    cvta.to.global.u64 %rd7, %rd2;
    cvta.to.global.u64 %rd8, %rd3;
    cvta.to.global.u64 %rd9, %rd4;
    cvta.to.global.u64 %rd10, %rd5;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mad.lo.u32 %r7, %r5, %r6, %r4;
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra PAGED_KV_APPEND_DONE;

    div.u32 %r8, %r1, %r3;
    rem.u32 %r9, %r1, %r3;
    mul.wide.u32 %rd11, %r8, 4;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.u32 %r10, [%rd12];
    mad.lo.u32 %r11, %r10, %r3, %r9;
    mad.lo.u32 %r12, %r11, %r2, %r7;
    mul.wide.u32 %rd13, %r7, 4;
    mul.wide.u32 %rd14, %r12, 4;
    add.s64 %rd15, %rd9, %rd13;
    add.s64 %rd16, %rd6, %rd14;
    ld.global.f32 %f1, [%rd15];
    st.global.f32 [%rd16], %f1;
    add.s64 %rd15, %rd10, %rd13;
    add.s64 %rd17, %rd7, %rd14;
    ld.global.f32 %f2, [%rd15];
    st.global.f32 [%rd17], %f2;

PAGED_KV_APPEND_DONE:
    ret;
}

.visible .entry paged_kv_cache_gather_kernel(
    .param .u64 paged_kv_cache_gather_kernel_param_0,
    .param .u64 paged_kv_cache_gather_kernel_param_1,
    .param .u64 paged_kv_cache_gather_kernel_param_2,
    .param .u64 paged_kv_cache_gather_kernel_param_3,
    .param .u64 paged_kv_cache_gather_kernel_param_4,
    .param .u32 paged_kv_cache_gather_kernel_param_5,
    .param .u32 paged_kv_cache_gather_kernel_param_6,
    .param .u32 paged_kv_cache_gather_kernel_param_7,
    .param .u32 paged_kv_cache_gather_kernel_param_8
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<20>;

    ld.param.u64 %rd1, [paged_kv_cache_gather_kernel_param_0];
    ld.param.u64 %rd2, [paged_kv_cache_gather_kernel_param_1];
    ld.param.u64 %rd3, [paged_kv_cache_gather_kernel_param_2];
    ld.param.u64 %rd4, [paged_kv_cache_gather_kernel_param_3];
    ld.param.u64 %rd5, [paged_kv_cache_gather_kernel_param_4];
    ld.param.u32 %r1, [paged_kv_cache_gather_kernel_param_5];
    ld.param.u32 %r2, [paged_kv_cache_gather_kernel_param_6];
    ld.param.u32 %r3, [paged_kv_cache_gather_kernel_param_7];
    ld.param.u32 %r16, [paged_kv_cache_gather_kernel_param_8];

    cvta.to.global.u64 %rd6, %rd1;
    cvta.to.global.u64 %rd7, %rd2;
    cvta.to.global.u64 %rd8, %rd3;
    cvta.to.global.u64 %rd9, %rd4;
    cvta.to.global.u64 %rd10, %rd5;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %ntid.x;
    mad.lo.u32 %r7, %r5, %r6, %r4;
    mul.lo.u32 %r8, %r1, %r2;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra PAGED_KV_GATHER_DONE;

    div.u32 %r9, %r7, %r2;
    add.u32 %r9, %r9, %r16;
    rem.u32 %r10, %r7, %r2;
    div.u32 %r11, %r9, %r3;
    rem.u32 %r12, %r9, %r3;
    mul.wide.u32 %rd11, %r11, 4;
    add.s64 %rd12, %rd8, %rd11;
    ld.global.u32 %r13, [%rd12];
    mad.lo.u32 %r14, %r13, %r3, %r12;
    mad.lo.u32 %r15, %r14, %r2, %r10;
    mul.wide.u32 %rd13, %r15, 4;
    add.s64 %rd14, %rd6, %rd13;
    add.s64 %rd15, %rd7, %rd13;
    ld.global.f32 %f1, [%rd14];
    ld.global.f32 %f2, [%rd15];
    mul.wide.u32 %rd16, %r7, 4;
    add.s64 %rd17, %rd9, %rd16;
    add.s64 %rd18, %rd10, %rd16;
    st.global.f32 [%rd17], %f1;
    st.global.f32 [%rd18], %f2;

PAGED_KV_GATHER_DONE:
    ret;
}

.visible .entry q8_kv_cache_append_kernel(
    .param .u64 q8_kv_cache_append_kernel_param_0,
    .param .u64 q8_kv_cache_append_kernel_param_1,
    .param .u64 q8_kv_cache_append_kernel_param_2,
    .param .u64 q8_kv_cache_append_kernel_param_3,
    .param .u64 q8_kv_cache_append_kernel_param_4,
    .param .u64 q8_kv_cache_append_kernel_param_5,
    .param .u32 q8_kv_cache_append_kernel_param_6,
    .param .u32 q8_kv_cache_append_kernel_param_7,
    .param .u64 q8_kv_cache_append_kernel_param_8,
    .param .u32 q8_kv_cache_append_kernel_param_9
)
{
    .reg .pred %p<8>;
    .reg .f32 %f<20>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<34>;

    ld.param.u64 %rd1, [q8_kv_cache_append_kernel_param_0];
    ld.param.u64 %rd2, [q8_kv_cache_append_kernel_param_1];
    ld.param.u64 %rd3, [q8_kv_cache_append_kernel_param_2];
    ld.param.u64 %rd4, [q8_kv_cache_append_kernel_param_3];
    ld.param.u64 %rd5, [q8_kv_cache_append_kernel_param_4];
    ld.param.u64 %rd6, [q8_kv_cache_append_kernel_param_5];
    ld.param.u32 %r1, [q8_kv_cache_append_kernel_param_6];
    ld.param.u32 %r2, [q8_kv_cache_append_kernel_param_7];
    ld.param.u64 %rd25, [q8_kv_cache_append_kernel_param_8];
    ld.param.u32 %r24, [q8_kv_cache_append_kernel_param_9];

    cvta.to.global.u64 %rd7, %rd1;
    cvta.to.global.u64 %rd8, %rd2;
    cvta.to.global.u64 %rd9, %rd3;
    cvta.to.global.u64 %rd10, %rd4;
    cvta.to.global.u64 %rd11, %rd5;
    cvta.to.global.u64 %rd12, %rd6;
    cvta.to.global.u64 %rd26, %rd25;

    mov.u32 %r3, %tid.x;
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra Q8_KV_APPEND_DONE;
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra Q8_KV_APPEND_DONE;

    div.u32 %r25, %r1, %r24;
    mul.lo.u32 %r26, %r25, %r24;
    sub.u32 %r27, %r1, %r26;
    mul.wide.u32 %rd27, %r25, 4;
    add.s64 %rd28, %rd26, %rd27;
    ld.global.u32 %r26, [%rd28];
    mul.lo.u32 %r26, %r26, %r24;
    add.u32 %r1, %r26, %r27;

    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r4, 0;

Q8_KV_APPEND_MAX_LOOP:
    setp.ge.u32 %p3, %r4, %r2;
    @%p3 bra Q8_KV_APPEND_SCALE;
    mul.wide.u32 %rd13, %r4, 4;
    add.s64 %rd14, %rd11, %rd13;
    add.s64 %rd15, %rd12, %rd13;
    ld.global.f32 %f3, [%rd14];
    ld.global.f32 %f4, [%rd15];
    abs.f32 %f5, %f3;
    abs.f32 %f6, %f4;
    max.f32 %f1, %f1, %f5;
    max.f32 %f2, %f2, %f6;
    add.u32 %r4, %r4, 1;
    bra Q8_KV_APPEND_MAX_LOOP;

Q8_KV_APPEND_SCALE:
    mov.f32 %f7, 0f3f800000;
    mov.f32 %f8, 0f3f800000;
    mov.f32 %f9, 0f42FE0000;
    setp.gt.f32 %p4, %f1, 0f00000000;
    @%p4 div.rn.f32 %f7, %f1, %f9;
    setp.gt.f32 %p5, %f2, 0f00000000;
    @%p5 div.rn.f32 %f8, %f2, %f9;

    mul.wide.u32 %rd16, %r1, 4;
    add.s64 %rd17, %rd9, %rd16;
    add.s64 %rd18, %rd10, %rd16;
    st.global.f32 [%rd17], %f7;
    st.global.f32 [%rd18], %f8;

    mul.lo.u32 %r5, %r1, %r2;
    mov.u32 %r4, 0;

Q8_KV_APPEND_STORE_LOOP:
    setp.ge.u32 %p6, %r4, %r2;
    @%p6 bra Q8_KV_APPEND_DONE;
    mul.wide.u32 %rd19, %r4, 4;
    add.s64 %rd20, %rd11, %rd19;
    add.s64 %rd21, %rd12, %rd19;
    ld.global.f32 %f10, [%rd20];
    ld.global.f32 %f11, [%rd21];
    div.rn.f32 %f12, %f10, %f7;
    div.rn.f32 %f13, %f11, %f8;
    cvt.rni.s32.f32 %r6, %f12;
    cvt.rni.s32.f32 %r7, %f13;
    max.s32 %r6, %r6, -127;
    min.s32 %r6, %r6, 127;
    max.s32 %r7, %r7, -127;
    min.s32 %r7, %r7, 127;
    add.u32 %r8, %r5, %r4;
    cvt.u64.u32 %rd22, %r8;
    add.s64 %rd23, %rd7, %rd22;
    add.s64 %rd24, %rd8, %rd22;
    st.global.u8 [%rd23], %r6;
    st.global.u8 [%rd24], %r7;
    add.u32 %r4, %r4, 1;
    bra Q8_KV_APPEND_STORE_LOOP;

Q8_KV_APPEND_DONE:
    ret;
}

.visible .entry q8_kv_cache_dequantize_kernel(
    .param .u64 q8_kv_cache_dequantize_kernel_param_0,
    .param .u64 q8_kv_cache_dequantize_kernel_param_1,
    .param .u64 q8_kv_cache_dequantize_kernel_param_2,
    .param .u64 q8_kv_cache_dequantize_kernel_param_3,
    .param .u64 q8_kv_cache_dequantize_kernel_param_4,
    .param .u64 q8_kv_cache_dequantize_kernel_param_5,
    .param .u32 q8_kv_cache_dequantize_kernel_param_6,
    .param .u32 q8_kv_cache_dequantize_kernel_param_7,
    .param .u64 q8_kv_cache_dequantize_kernel_param_8,
    .param .u32 q8_kv_cache_dequantize_kernel_param_9
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<8>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<24>;

    ld.param.u64 %rd1, [q8_kv_cache_dequantize_kernel_param_0];
    ld.param.u64 %rd2, [q8_kv_cache_dequantize_kernel_param_1];
    ld.param.u64 %rd3, [q8_kv_cache_dequantize_kernel_param_2];
    ld.param.u64 %rd4, [q8_kv_cache_dequantize_kernel_param_3];
    ld.param.u64 %rd5, [q8_kv_cache_dequantize_kernel_param_4];
    ld.param.u64 %rd6, [q8_kv_cache_dequantize_kernel_param_5];
    ld.param.u32 %r1, [q8_kv_cache_dequantize_kernel_param_6];
    ld.param.u32 %r2, [q8_kv_cache_dequantize_kernel_param_7];
    ld.param.u64 %rd22, [q8_kv_cache_dequantize_kernel_param_8];
    ld.param.u32 %r11, [q8_kv_cache_dequantize_kernel_param_9];

    cvta.to.global.u64 %rd7, %rd1;
    cvta.to.global.u64 %rd8, %rd2;
    cvta.to.global.u64 %rd9, %rd3;
    cvta.to.global.u64 %rd10, %rd4;
    cvta.to.global.u64 %rd11, %rd5;
    cvta.to.global.u64 %rd12, %rd6;
    cvta.to.global.u64 %rd23, %rd22;

    div.u32 %r12, %r1, %r11;
    mul.lo.u32 %r13, %r12, %r11;
    sub.u32 %r13, %r1, %r13;
    mul.wide.u32 %rd22, %r12, 4;
    add.s64 %rd22, %rd23, %rd22;
    ld.global.u32 %r12, [%rd22];
    mul.lo.u32 %r12, %r12, %r11;
    add.u32 %r1, %r12, %r13;

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra Q8_KV_DEQUANT_DONE;

    mul.lo.u32 %r7, %r1, %r2;
    add.u32 %r8, %r7, %r6;
    cvt.u64.u32 %rd13, %r8;
    add.s64 %rd14, %rd7, %rd13;
    add.s64 %rd15, %rd8, %rd13;
    ld.global.s8 %r9, [%rd14];
    ld.global.s8 %r10, [%rd15];
    cvt.rn.f32.s32 %f1, %r9;
    cvt.rn.f32.s32 %f2, %r10;

    mul.wide.u32 %rd16, %r1, 4;
    add.s64 %rd17, %rd9, %rd16;
    add.s64 %rd18, %rd10, %rd16;
    ld.global.f32 %f3, [%rd17];
    ld.global.f32 %f4, [%rd18];
    mul.f32 %f5, %f1, %f3;
    mul.f32 %f6, %f2, %f4;

    mul.wide.u32 %rd19, %r6, 4;
    add.s64 %rd20, %rd11, %rd19;
    add.s64 %rd21, %rd12, %rd19;
    st.global.f32 [%rd20], %f5;
    st.global.f32 [%rd21], %f6;

Q8_KV_DEQUANT_DONE:
    ret;
}

.visible .entry kq4_vq8_kv_cache_append_kernel(
    .param .u64 kq4_vq8_kv_cache_append_kernel_param_0,
    .param .u64 kq4_vq8_kv_cache_append_kernel_param_1,
    .param .u64 kq4_vq8_kv_cache_append_kernel_param_2,
    .param .u64 kq4_vq8_kv_cache_append_kernel_param_3,
    .param .u64 kq4_vq8_kv_cache_append_kernel_param_4,
    .param .u64 kq4_vq8_kv_cache_append_kernel_param_5,
    .param .u32 kq4_vq8_kv_cache_append_kernel_param_6,
    .param .u32 kq4_vq8_kv_cache_append_kernel_param_7,
    .param .u64 kq4_vq8_kv_cache_append_kernel_param_8,
    .param .u32 kq4_vq8_kv_cache_append_kernel_param_9
)
{
    .reg .pred %p<12>;
    .reg .f32 %f<28>;
    .reg .b32 %r<46>;
    .reg .b64 %rd<42>;

    ld.param.u64 %rd1, [kq4_vq8_kv_cache_append_kernel_param_0];
    ld.param.u64 %rd2, [kq4_vq8_kv_cache_append_kernel_param_1];
    ld.param.u64 %rd3, [kq4_vq8_kv_cache_append_kernel_param_2];
    ld.param.u64 %rd4, [kq4_vq8_kv_cache_append_kernel_param_3];
    ld.param.u64 %rd5, [kq4_vq8_kv_cache_append_kernel_param_4];
    ld.param.u64 %rd6, [kq4_vq8_kv_cache_append_kernel_param_5];
    ld.param.u32 %r1, [kq4_vq8_kv_cache_append_kernel_param_6];
    ld.param.u32 %r2, [kq4_vq8_kv_cache_append_kernel_param_7];
    ld.param.u64 %rd35, [kq4_vq8_kv_cache_append_kernel_param_8];
    ld.param.u32 %r40, [kq4_vq8_kv_cache_append_kernel_param_9];

    cvta.to.global.u64 %rd7, %rd1;
    cvta.to.global.u64 %rd8, %rd2;
    cvta.to.global.u64 %rd9, %rd3;
    cvta.to.global.u64 %rd10, %rd4;
    cvta.to.global.u64 %rd11, %rd5;
    cvta.to.global.u64 %rd12, %rd6;
    cvta.to.global.u64 %rd36, %rd35;

    mov.u32 %r3, %tid.x;
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra KQ4VQ8_APPEND_DONE;
    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra KQ4VQ8_APPEND_DONE;

    div.u32 %r41, %r1, %r40;
    mul.lo.u32 %r42, %r41, %r40;
    sub.u32 %r43, %r1, %r42;
    mul.wide.u32 %rd37, %r41, 4;
    add.s64 %rd38, %rd36, %rd37;
    ld.global.u32 %r42, [%rd38];
    mul.lo.u32 %r42, %r42, %r40;
    add.u32 %r1, %r42, %r43;

    add.u32 %r4, %r2, 1;
    shr.u32 %r5, %r4, 1;
    add.u32 %r6, %r2, 63;
    shr.u32 %r7, %r6, 6;
    mul.lo.u32 %r8, %r1, %r5;
    mul.lo.u32 %r9, %r1, %r2;
    mul.lo.u32 %r10, %r1, %r7;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r11, 0;

KQ4VQ8_VALUE_MAX_LOOP:
    setp.ge.u32 %p3, %r11, %r2;
    @%p3 bra KQ4VQ8_VALUE_SCALE;
    mul.wide.u32 %rd13, %r11, 4;
    add.s64 %rd14, %rd12, %rd13;
    ld.global.f32 %f2, [%rd14];
    abs.f32 %f3, %f2;
    max.f32 %f1, %f1, %f3;
    add.u32 %r11, %r11, 1;
    bra KQ4VQ8_VALUE_MAX_LOOP;

KQ4VQ8_VALUE_SCALE:
    mov.f32 %f4, 0f3f800000;
    mov.f32 %f5, 0f42FE0000;
    setp.gt.f32 %p4, %f1, 0f00000000;
    @%p4 div.rn.f32 %f4, %f1, %f5;
    mul.wide.u32 %rd15, %r1, 4;
    add.s64 %rd16, %rd10, %rd15;
    st.global.f32 [%rd16], %f4;

    mov.u32 %r11, 0;

KQ4VQ8_VALUE_STORE_LOOP:
    setp.ge.u32 %p5, %r11, %r2;
    @%p5 bra KQ4VQ8_KEY_GROUP_LOOP_INIT;
    mul.wide.u32 %rd17, %r11, 4;
    add.s64 %rd18, %rd12, %rd17;
    ld.global.f32 %f6, [%rd18];
    div.rn.f32 %f7, %f6, %f4;
    cvt.rni.s32.f32 %r12, %f7;
    max.s32 %r12, %r12, -127;
    min.s32 %r12, %r12, 127;
    add.u32 %r13, %r9, %r11;
    cvt.u64.u32 %rd19, %r13;
    add.s64 %rd20, %rd8, %rd19;
    st.global.u8 [%rd20], %r12;
    add.u32 %r11, %r11, 1;
    bra KQ4VQ8_VALUE_STORE_LOOP;

KQ4VQ8_KEY_GROUP_LOOP_INIT:
    mov.u32 %r14, 0;

KQ4VQ8_KEY_GROUP_LOOP:
    setp.ge.u32 %p6, %r14, %r7;
    @%p6 bra KQ4VQ8_APPEND_DONE;
    shl.b32 %r15, %r14, 6;
    add.u32 %r16, %r15, 64;
    min.u32 %r16, %r16, %r2;
    mov.f32 %f8, 0f00000000;
    mov.u32 %r17, %r15;

KQ4VQ8_KEY_MAX_LOOP:
    setp.ge.u32 %p7, %r17, %r16;
    @%p7 bra KQ4VQ8_KEY_SCALE;
    mul.wide.u32 %rd21, %r17, 4;
    add.s64 %rd22, %rd11, %rd21;
    ld.global.f32 %f9, [%rd22];
    abs.f32 %f10, %f9;
    max.f32 %f8, %f8, %f10;
    add.u32 %r17, %r17, 1;
    bra KQ4VQ8_KEY_MAX_LOOP;

KQ4VQ8_KEY_SCALE:
    mov.f32 %f11, 0f3f800000;
    mov.f32 %f12, 0f41000000;
    setp.gt.f32 %p8, %f8, 0f00000000;
    @%p8 div.rn.f32 %f11, %f8, %f12;
    add.u32 %r18, %r10, %r14;
    mul.wide.u32 %rd23, %r18, 4;
    add.s64 %rd24, %rd9, %rd23;
    st.global.f32 [%rd24], %f11;
    mov.u32 %r17, %r15;

KQ4VQ8_KEY_STORE_LOOP:
    setp.ge.u32 %p9, %r17, %r16;
    @%p9 bra KQ4VQ8_KEY_GROUP_NEXT;
    mul.wide.u32 %rd25, %r17, 4;
    add.s64 %rd26, %rd11, %rd25;
    ld.global.f32 %f13, [%rd26];
    div.rn.f32 %f14, %f13, %f11;
    cvt.rni.s32.f32 %r19, %f14;
    max.s32 %r19, %r19, -8;
    min.s32 %r19, %r19, 7;
    add.s32 %r19, %r19, 8;
    shr.u32 %r20, %r17, 1;
    add.u32 %r21, %r8, %r20;
    cvt.u64.u32 %rd27, %r21;
    add.s64 %rd28, %rd7, %rd27;
    ld.global.u8 %r22, [%rd28];
    and.b32 %r23, %r17, 1;
    setp.eq.u32 %p10, %r23, 0;
    @%p10 bra KQ4VQ8_KEY_STORE_LOW;
    and.b32 %r22, %r22, 15;
    shl.b32 %r24, %r19, 4;
    or.b32 %r22, %r22, %r24;
    bra KQ4VQ8_KEY_STORE_WRITE;

KQ4VQ8_KEY_STORE_LOW:
    and.b32 %r22, %r22, 240;
    or.b32 %r22, %r22, %r19;

KQ4VQ8_KEY_STORE_WRITE:
    st.global.u8 [%rd28], %r22;
    add.u32 %r17, %r17, 1;
    bra KQ4VQ8_KEY_STORE_LOOP;

KQ4VQ8_KEY_GROUP_NEXT:
    add.u32 %r14, %r14, 1;
    bra KQ4VQ8_KEY_GROUP_LOOP;

KQ4VQ8_APPEND_DONE:
    ret;
}

.visible .entry kq4_vq8_kv_cache_copy_row_kernel(
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_0,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_1,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_2,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_3,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_4,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_5,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_6,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_7,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_8,
    .param .u64 kq4_vq8_kv_cache_copy_row_kernel_param_9,
    .param .u32 kq4_vq8_kv_cache_copy_row_kernel_param_10,
    .param .u32 kq4_vq8_kv_cache_copy_row_kernel_param_11,
    .param .u32 kq4_vq8_kv_cache_copy_row_kernel_param_12,
    .param .u32 kq4_vq8_kv_cache_copy_row_kernel_param_13,
    .param .u32 kq4_vq8_kv_cache_copy_row_kernel_param_14
)
{
    .reg .pred %p<6>;
    .reg .f32 %f<3>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<48>;

    ld.param.u64 %rd1, [kq4_vq8_kv_cache_copy_row_kernel_param_0];
    ld.param.u64 %rd2, [kq4_vq8_kv_cache_copy_row_kernel_param_1];
    ld.param.u64 %rd3, [kq4_vq8_kv_cache_copy_row_kernel_param_2];
    ld.param.u64 %rd4, [kq4_vq8_kv_cache_copy_row_kernel_param_3];
    ld.param.u64 %rd5, [kq4_vq8_kv_cache_copy_row_kernel_param_4];
    ld.param.u64 %rd6, [kq4_vq8_kv_cache_copy_row_kernel_param_5];
    ld.param.u64 %rd7, [kq4_vq8_kv_cache_copy_row_kernel_param_6];
    ld.param.u64 %rd8, [kq4_vq8_kv_cache_copy_row_kernel_param_7];
    ld.param.u64 %rd9, [kq4_vq8_kv_cache_copy_row_kernel_param_8];
    ld.param.u64 %rd10, [kq4_vq8_kv_cache_copy_row_kernel_param_9];
    ld.param.u32 %r1, [kq4_vq8_kv_cache_copy_row_kernel_param_10];
    ld.param.u32 %r2, [kq4_vq8_kv_cache_copy_row_kernel_param_11];
    ld.param.u32 %r3, [kq4_vq8_kv_cache_copy_row_kernel_param_12];
    ld.param.u32 %r4, [kq4_vq8_kv_cache_copy_row_kernel_param_13];
    ld.param.u32 %r5, [kq4_vq8_kv_cache_copy_row_kernel_param_14];

    cvta.to.global.u64 %rd11, %rd1;
    cvta.to.global.u64 %rd12, %rd2;
    cvta.to.global.u64 %rd13, %rd3;
    cvta.to.global.u64 %rd14, %rd4;
    cvta.to.global.u64 %rd15, %rd5;
    cvta.to.global.u64 %rd16, %rd6;
    cvta.to.global.u64 %rd17, %rd7;
    cvta.to.global.u64 %rd18, %rd8;
    cvta.to.global.u64 %rd19, %rd9;
    cvta.to.global.u64 %rd20, %rd10;

    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mad.lo.u32 %r9, %r7, %r8, %r6;
    setp.ge.u32 %p1, %r9, %r3;
    @%p1 bra KQ4VQ8_COPY_ROW_DONE;

    div.u32 %r10, %r1, %r4;
    rem.u32 %r11, %r1, %r4;
    mul.wide.u32 %rd21, %r10, 4;
    add.s64 %rd22, %rd15, %rd21;
    ld.global.u32 %r12, [%rd22];
    mad.lo.u32 %r13, %r12, %r4, %r11;

    div.u32 %r14, %r2, %r5;
    rem.u32 %r15, %r2, %r5;
    mul.wide.u32 %rd23, %r14, 4;
    add.s64 %rd24, %rd20, %rd23;
    ld.global.u32 %r16, [%rd24];
    mad.lo.u32 %r17, %r16, %r5, %r15;

    add.u32 %r18, %r3, 1;
    shr.u32 %r18, %r18, 1;
    add.u32 %r19, %r3, 63;
    shr.u32 %r19, %r19, 6;

    mad.lo.u32 %r20, %r13, %r3, %r9;
    mad.lo.u32 %r21, %r17, %r3, %r9;
    cvt.u64.u32 %rd25, %r20;
    cvt.u64.u32 %rd26, %r21;
    add.s64 %rd27, %rd12, %rd25;
    add.s64 %rd28, %rd17, %rd26;
    ld.global.u8 %r22, [%rd27];
    st.global.u8 [%rd28], %r22;

    setp.ge.u32 %p2, %r9, %r18;
    @%p2 bra KQ4VQ8_COPY_ROW_KEY_DONE;
    mad.lo.u32 %r23, %r13, %r18, %r9;
    mad.lo.u32 %r24, %r17, %r18, %r9;
    cvt.u64.u32 %rd29, %r23;
    cvt.u64.u32 %rd30, %r24;
    add.s64 %rd31, %rd11, %rd29;
    add.s64 %rd32, %rd16, %rd30;
    ld.global.u8 %r25, [%rd31];
    st.global.u8 [%rd32], %r25;

KQ4VQ8_COPY_ROW_KEY_DONE:
    setp.ge.u32 %p3, %r9, %r19;
    @%p3 bra KQ4VQ8_COPY_ROW_KEY_SCALE_DONE;
    mad.lo.u32 %r26, %r13, %r19, %r9;
    mad.lo.u32 %r27, %r17, %r19, %r9;
    mul.wide.u32 %rd33, %r26, 4;
    mul.wide.u32 %rd34, %r27, 4;
    add.s64 %rd35, %rd13, %rd33;
    add.s64 %rd36, %rd18, %rd34;
    ld.global.f32 %f1, [%rd35];
    st.global.f32 [%rd36], %f1;

KQ4VQ8_COPY_ROW_KEY_SCALE_DONE:
    setp.ne.u32 %p4, %r9, 0;
    @%p4 bra KQ4VQ8_COPY_ROW_DONE;
    mul.wide.u32 %rd37, %r13, 4;
    mul.wide.u32 %rd38, %r17, 4;
    add.s64 %rd39, %rd14, %rd37;
    add.s64 %rd40, %rd19, %rd38;
    ld.global.f32 %f2, [%rd39];
    st.global.f32 [%rd40], %f2;

KQ4VQ8_COPY_ROW_DONE:
    ret;
}

.visible .entry kq4_vq8_kv_cache_dequantize_kernel(
    .param .u64 kq4_vq8_kv_cache_dequantize_kernel_param_0,
    .param .u64 kq4_vq8_kv_cache_dequantize_kernel_param_1,
    .param .u64 kq4_vq8_kv_cache_dequantize_kernel_param_2,
    .param .u64 kq4_vq8_kv_cache_dequantize_kernel_param_3,
    .param .u64 kq4_vq8_kv_cache_dequantize_kernel_param_4,
    .param .u64 kq4_vq8_kv_cache_dequantize_kernel_param_5,
    .param .u32 kq4_vq8_kv_cache_dequantize_kernel_param_6,
    .param .u32 kq4_vq8_kv_cache_dequantize_kernel_param_7,
    .param .u64 kq4_vq8_kv_cache_dequantize_kernel_param_8,
    .param .u32 kq4_vq8_kv_cache_dequantize_kernel_param_9
)
{
    .reg .pred %p<4>;
    .reg .f32 %f<12>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<36>;

    ld.param.u64 %rd1, [kq4_vq8_kv_cache_dequantize_kernel_param_0];
    ld.param.u64 %rd2, [kq4_vq8_kv_cache_dequantize_kernel_param_1];
    ld.param.u64 %rd3, [kq4_vq8_kv_cache_dequantize_kernel_param_2];
    ld.param.u64 %rd4, [kq4_vq8_kv_cache_dequantize_kernel_param_3];
    ld.param.u64 %rd5, [kq4_vq8_kv_cache_dequantize_kernel_param_4];
    ld.param.u64 %rd6, [kq4_vq8_kv_cache_dequantize_kernel_param_5];
    ld.param.u32 %r1, [kq4_vq8_kv_cache_dequantize_kernel_param_6];
    ld.param.u32 %r2, [kq4_vq8_kv_cache_dequantize_kernel_param_7];
    ld.param.u64 %rd30, [kq4_vq8_kv_cache_dequantize_kernel_param_8];
    ld.param.u32 %r24, [kq4_vq8_kv_cache_dequantize_kernel_param_9];

    cvta.to.global.u64 %rd7, %rd1;
    cvta.to.global.u64 %rd8, %rd2;
    cvta.to.global.u64 %rd9, %rd3;
    cvta.to.global.u64 %rd10, %rd4;
    cvta.to.global.u64 %rd11, %rd5;
    cvta.to.global.u64 %rd12, %rd6;
    cvta.to.global.u64 %rd31, %rd30;

    div.u32 %r25, %r1, %r24;
    mul.lo.u32 %r26, %r25, %r24;
    sub.u32 %r27, %r1, %r26;
    mul.wide.u32 %rd32, %r25, 4;
    add.s64 %rd33, %rd31, %rd32;
    ld.global.u32 %r26, [%rd33];
    mul.lo.u32 %r26, %r26, %r24;
    add.u32 %r1, %r26, %r27;

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.u32 %r6, %r4, %r5, %r3;
    setp.ge.u32 %p1, %r6, %r2;
    @%p1 bra KQ4VQ8_DEQUANT_DONE;

    add.u32 %r7, %r2, 1;
    shr.u32 %r8, %r7, 1;
    add.u32 %r9, %r2, 63;
    shr.u32 %r10, %r9, 6;
    mul.lo.u32 %r11, %r1, %r8;
    mul.lo.u32 %r12, %r1, %r2;
    mul.lo.u32 %r13, %r1, %r10;

    shr.u32 %r14, %r6, 1;
    add.u32 %r15, %r11, %r14;
    cvt.u64.u32 %rd13, %r15;
    add.s64 %rd14, %rd7, %rd13;
    ld.global.u8 %r16, [%rd14];
    and.b32 %r17, %r6, 1;
    setp.eq.u32 %p2, %r17, 0;
    @%p2 bra KQ4VQ8_DEQUANT_KEY_LOW;
    shr.u32 %r18, %r16, 4;
    bra KQ4VQ8_DEQUANT_KEY_READY;

KQ4VQ8_DEQUANT_KEY_LOW:
    and.b32 %r18, %r16, 15;

KQ4VQ8_DEQUANT_KEY_READY:
    cvt.s32.u32 %r19, %r18;
    add.s32 %r19, %r19, -8;
    cvt.rn.f32.s32 %f1, %r19;
    shr.u32 %r20, %r6, 6;
    add.u32 %r21, %r13, %r20;
    mul.wide.u32 %rd15, %r21, 4;
    add.s64 %rd16, %rd9, %rd15;
    ld.global.f32 %f2, [%rd16];
    mul.f32 %f3, %f1, %f2;

    add.u32 %r22, %r12, %r6;
    cvt.u64.u32 %rd17, %r22;
    add.s64 %rd18, %rd8, %rd17;
    ld.global.s8 %r23, [%rd18];
    cvt.rn.f32.s32 %f4, %r23;
    mul.wide.u32 %rd19, %r1, 4;
    add.s64 %rd20, %rd10, %rd19;
    ld.global.f32 %f5, [%rd20];
    mul.f32 %f6, %f4, %f5;

    mul.wide.u32 %rd21, %r6, 4;
    add.s64 %rd22, %rd11, %rd21;
    add.s64 %rd23, %rd12, %rd21;
    st.global.f32 [%rd22], %f3;
    st.global.f32 [%rd23], %f6;

KQ4VQ8_DEQUANT_DONE:
    ret;
}

.visible .entry attention_scores_kernel(
    .param .u64 attention_scores_kernel_param_0,
    .param .u64 attention_scores_kernel_param_1,
    .param .u64 attention_scores_kernel_param_2,
    .param .u32 attention_scores_kernel_param_3,
    .param .u32 attention_scores_kernel_param_4,
    .param .u32 attention_scores_kernel_param_5,
    .param .u32 attention_scores_kernel_param_6,
    .param .u32 attention_scores_kernel_param_7,
    .param .f32 attention_scores_kernel_param_8
)
{
    .reg .pred %p<3>;
    .reg .f32 %f<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<18>;

    ld.param.u64 %rd1, [attention_scores_kernel_param_0];
    ld.param.u64 %rd2, [attention_scores_kernel_param_1];
    ld.param.u64 %rd3, [attention_scores_kernel_param_2];
    ld.param.u32 %r1, [attention_scores_kernel_param_3];
    ld.param.u32 %r2, [attention_scores_kernel_param_4];
    ld.param.u32 %r3, [attention_scores_kernel_param_5];
    ld.param.u32 %r4, [attention_scores_kernel_param_6];
    ld.param.u32 %r5, [attention_scores_kernel_param_7];
    ld.param.f32 %f1, [attention_scores_kernel_param_8];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;

    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mad.lo.u32 %r9, %r7, %r8, %r6;
    mul.lo.u32 %r10, %r1, %r4;
    setp.ge.u32 %p1, %r9, %r10;
    @%p1 bra ATTENTION_SCORES_DONE;

    div.u32 %r11, %r9, %r4;
    mul.lo.u32 %r12, %r11, %r4;
    sub.u32 %r13, %r9, %r12;
    div.u32 %r14, %r1, %r2;
    div.u32 %r15, %r11, %r14;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r16, 0;

ATTENTION_SCORE_LOOP:
    setp.ge.u32 %p2, %r16, %r3;
    @%p2 bra ATTENTION_SCORE_LOOP_DONE;
    mad.lo.u32 %r17, %r11, %r3, %r16;
    mul.wide.u32 %rd7, %r17, 4;
    add.s64 %rd8, %rd4, %rd7;
    ld.global.f32 %f3, [%rd8];
    mul.lo.u32 %r18, %r13, %r5;
    mul.lo.u32 %r19, %r15, %r3;
    add.u32 %r20, %r18, %r19;
    add.u32 %r21, %r20, %r16;
    mul.wide.u32 %rd9, %r21, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f4, [%rd10];
    fma.rn.f32 %f2, %f3, %f4, %f2;
    add.u32 %r16, %r16, 1;
    bra ATTENTION_SCORE_LOOP;

ATTENTION_SCORE_LOOP_DONE:
    mul.f32 %f5, %f2, %f1;
    mul.wide.u32 %rd11, %r9, 4;
    add.s64 %rd12, %rd6, %rd11;
    st.global.f32 [%rd12], %f5;

ATTENTION_SCORES_DONE:
    ret;
}

.visible .entry attention_values_kernel(
    .param .u64 attention_values_kernel_param_0,
    .param .u64 attention_values_kernel_param_1,
    .param .u64 attention_values_kernel_param_2,
    .param .u32 attention_values_kernel_param_3,
    .param .u32 attention_values_kernel_param_4,
    .param .u32 attention_values_kernel_param_5,
    .param .u32 attention_values_kernel_param_6,
    .param .u32 attention_values_kernel_param_7,
    .param .u32 attention_values_kernel_param_8
)
{
    .reg .pred %p<3>;
    .reg .f32 %f<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<18>;

    ld.param.u64 %rd1, [attention_values_kernel_param_0];
    ld.param.u64 %rd2, [attention_values_kernel_param_1];
    ld.param.u64 %rd3, [attention_values_kernel_param_2];
    ld.param.u32 %r1, [attention_values_kernel_param_3];
    ld.param.u32 %r2, [attention_values_kernel_param_4];
    ld.param.u32 %r3, [attention_values_kernel_param_5];
    ld.param.u32 %r4, [attention_values_kernel_param_6];
    ld.param.u32 %r5, [attention_values_kernel_param_7];
    ld.param.u32 %r6, [attention_values_kernel_param_8];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;

    mov.u32 %r7, %tid.x;
    mov.u32 %r8, %ctaid.x;
    mov.u32 %r9, %ntid.x;
    mad.lo.u32 %r10, %r8, %r9, %r7;
    setp.ge.u32 %p1, %r10, %r6;
    @%p1 bra ATTENTION_VALUES_DONE;

    div.u32 %r11, %r10, %r3;
    mul.lo.u32 %r12, %r11, %r3;
    sub.u32 %r13, %r10, %r12;
    div.u32 %r14, %r1, %r2;
    div.u32 %r15, %r11, %r14;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r16, 0;

ATTENTION_VALUE_LOOP:
    setp.ge.u32 %p2, %r16, %r4;
    @%p2 bra ATTENTION_VALUE_LOOP_DONE;
    mad.lo.u32 %r17, %r11, %r4, %r16;
    mul.wide.u32 %rd7, %r17, 4;
    add.s64 %rd8, %rd4, %rd7;
    ld.global.f32 %f2, [%rd8];
    mul.lo.u32 %r18, %r16, %r5;
    mul.lo.u32 %r19, %r15, %r3;
    add.u32 %r20, %r18, %r19;
    add.u32 %r21, %r20, %r13;
    mul.wide.u32 %rd9, %r21, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f3, [%rd10];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r16, %r16, 1;
    bra ATTENTION_VALUE_LOOP;

ATTENTION_VALUE_LOOP_DONE:
    mul.wide.u32 %rd11, %r10, 4;
    add.s64 %rd12, %rd6, %rd11;
    st.global.f32 [%rd12], %f1;

ATTENTION_VALUES_DONE:
    ret;
}

.visible .entry single_query_attention_online_kernel(
    .param .u64 single_query_attention_online_kernel_param_0,
    .param .u64 single_query_attention_online_kernel_param_1,
    .param .u64 single_query_attention_online_kernel_param_2,
    .param .u64 single_query_attention_online_kernel_param_3,
    .param .u32 single_query_attention_online_kernel_param_4,
    .param .u32 single_query_attention_online_kernel_param_5,
    .param .u32 single_query_attention_online_kernel_param_6,
    .param .u32 single_query_attention_online_kernel_param_7,
    .param .u32 single_query_attention_online_kernel_param_8,
    .param .u32 single_query_attention_online_kernel_param_9,
    .param .f32 single_query_attention_online_kernel_param_10,
    .param .u64 single_query_attention_online_kernel_param_11,
    .param .u32 single_query_attention_online_kernel_param_12,
    .param .u32 single_query_attention_online_kernel_param_13
)
{
    .shared .align 4 .b8 single_attention_online_reduce[1024];
    .shared .align 4 .b8 single_attention_online_state[16];
    .reg .pred %p<9>;
    .reg .f32 %f<24>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<32>;

    ld.param.u64 %rd1, [single_query_attention_online_kernel_param_0];
    ld.param.u64 %rd2, [single_query_attention_online_kernel_param_1];
    ld.param.u64 %rd3, [single_query_attention_online_kernel_param_2];
    ld.param.u64 %rd4, [single_query_attention_online_kernel_param_3];
    ld.param.u32 %r1, [single_query_attention_online_kernel_param_4];
    ld.param.u32 %r2, [single_query_attention_online_kernel_param_5];
    ld.param.u32 %r3, [single_query_attention_online_kernel_param_6];
    ld.param.u32 %r4, [single_query_attention_online_kernel_param_7];
    ld.param.u32 %r5, [single_query_attention_online_kernel_param_8];
    ld.param.u32 %r6, [single_query_attention_online_kernel_param_9];
    ld.param.f32 %f1, [single_query_attention_online_kernel_param_10];
    ld.param.u64 %rd5, [single_query_attention_online_kernel_param_11];
    ld.param.u32 %r7, [single_query_attention_online_kernel_param_12];
    ld.param.u32 %r8, [single_query_attention_online_kernel_param_13];

    cvta.to.global.u64 %rd6, %rd1;
    cvta.to.global.u64 %rd7, %rd2;
    cvta.to.global.u64 %rd8, %rd3;
    cvta.to.global.u64 %rd9, %rd4;
    cvta.to.global.u64 %rd10, %rd5;
    mov.u64 %rd11, single_attention_online_reduce;
    mov.u64 %rd12, single_attention_online_state;
    add.s64 %rd13, %rd12, 4;
    add.s64 %rd14, %rd12, 8;
    add.s64 %rd15, %rd12, 12;

    mov.u32 %r9, %tid.x;
    mov.u32 %r10, %ctaid.x;
    setp.ge.u32 %p1, %r10, %r1;
    @%p1 bra SINGLE_ATTENTION_ONLINE_DONE;

    div.u32 %r11, %r1, %r2;
    div.u32 %r12, %r10, %r11;
    mov.f32 %f15, 0f00000000;
    setp.ne.u32 %p2, %r9, 0;
    @%p2 bra SINGLE_ATTENTION_ONLINE_INIT_DONE;
    mov.f32 %f5, 0fFF800000;
    mov.f32 %f6, 0f00000000;
    st.shared.f32 [%rd12], %f5;
    st.shared.f32 [%rd13], %f6;

SINGLE_ATTENTION_ONLINE_INIT_DONE:
    bar.sync 0;
    mov.u32 %r22, %r8;

SINGLE_ATTENTION_ONLINE_POS:
    setp.ge.u32 %p3, %r22, %r4;
    @%p3 bra SINGLE_ATTENTION_ONLINE_WRITE;
    mov.f32 %f2, 0f00000000;
    setp.ge.u32 %p4, %r9, %r3;
    @%p4 bra SINGLE_ATTENTION_ONLINE_PARTIAL_DONE;

    mad.lo.u32 %r13, %r10, %r3, %r9;
    mul.wide.u32 %rd16, %r13, 4;
    add.s64 %rd17, %rd6, %rd16;
    ld.global.f32 %f3, [%rd17];

    div.u32 %r14, %r22, %r7;
    rem.u32 %r15, %r22, %r7;
    mul.wide.u32 %rd18, %r14, 4;
    add.s64 %rd19, %rd10, %rd18;
    ld.global.u32 %r16, [%rd19];
    mad.lo.u32 %r17, %r16, %r7, %r15;
    mad.lo.u32 %r18, %r17, %r5, %r9;
    mul.lo.u32 %r19, %r12, %r3;
    add.u32 %r18, %r18, %r19;
    mul.wide.u32 %rd20, %r18, 4;
    add.s64 %rd21, %rd7, %rd20;
    ld.global.f32 %f4, [%rd21];
    mul.f32 %f2, %f3, %f4;

SINGLE_ATTENTION_ONLINE_PARTIAL_DONE:
    mul.wide.u32 %rd22, %r9, 4;
    add.s64 %rd23, %rd11, %rd22;
    st.shared.f32 [%rd23], %f2;
    bar.sync 0;

    mov.u32 %r20, 128;

SINGLE_ATTENTION_ONLINE_REDUCE:
    setp.eq.u32 %p5, %r20, 0;
    @%p5 bra SINGLE_ATTENTION_ONLINE_REDUCE_DONE;
    setp.ge.u32 %p6, %r9, %r20;
    @%p6 bra SINGLE_ATTENTION_ONLINE_REDUCE_SKIP;
    add.u32 %r21, %r9, %r20;
    mul.wide.u32 %rd24, %r21, 4;
    add.s64 %rd25, %rd11, %rd24;
    ld.shared.f32 %f18, [%rd23];
    ld.shared.f32 %f19, [%rd25];
    add.f32 %f20, %f18, %f19;
    st.shared.f32 [%rd23], %f20;

SINGLE_ATTENTION_ONLINE_REDUCE_SKIP:
    bar.sync 0;
    shr.u32 %r20, %r20, 1;
    bra SINGLE_ATTENTION_ONLINE_REDUCE;

SINGLE_ATTENTION_ONLINE_REDUCE_DONE:
    setp.ne.u32 %p7, %r9, 0;
    @%p7 bra SINGLE_ATTENTION_ONLINE_STATE_DONE;
    ld.shared.f32 %f7, [%rd11];
    mul.f32 %f7, %f7, %f1;
    ld.shared.f32 %f5, [%rd12];
    ld.shared.f32 %f6, [%rd13];
    max.f32 %f8, %f5, %f7;
    mov.f32 %f9, 0f3FB8AA3B;
    sub.f32 %f10, %f5, %f8;
    mul.f32 %f10, %f10, %f9;
    ex2.approx.f32 %f11, %f10;
    sub.f32 %f12, %f7, %f8;
    mul.f32 %f12, %f12, %f9;
    ex2.approx.f32 %f13, %f12;
    fma.rn.f32 %f14, %f6, %f11, %f13;
    st.shared.f32 [%rd12], %f8;
    st.shared.f32 [%rd13], %f14;
    st.shared.f32 [%rd14], %f11;
    st.shared.f32 [%rd15], %f13;

SINGLE_ATTENTION_ONLINE_STATE_DONE:
    bar.sync 0;
    setp.ge.u32 %p8, %r9, %r3;
    @%p8 bra SINGLE_ATTENTION_ONLINE_VALUE_DONE;
    ld.shared.f32 %f11, [%rd14];
    ld.shared.f32 %f13, [%rd15];
    mul.wide.u32 %rd26, %r18, 4;
    add.s64 %rd27, %rd8, %rd26;
    ld.global.f32 %f16, [%rd27];
    mul.f32 %f17, %f15, %f11;
    fma.rn.f32 %f15, %f13, %f16, %f17;

SINGLE_ATTENTION_ONLINE_VALUE_DONE:
    bar.sync 0;
    add.u32 %r22, %r22, 1;
    bra SINGLE_ATTENTION_ONLINE_POS;

SINGLE_ATTENTION_ONLINE_WRITE:
    setp.ge.u32 %p8, %r9, %r3;
    @%p8 bra SINGLE_ATTENTION_ONLINE_DONE;
    ld.shared.f32 %f6, [%rd13];
    div.rn.f32 %f21, %f15, %f6;
    mad.lo.u32 %r23, %r10, %r3, %r9;
    mul.wide.u32 %rd28, %r23, 4;
    add.s64 %rd29, %rd9, %rd28;
    st.global.f32 [%rd29], %f21;

SINGLE_ATTENTION_ONLINE_DONE:
    ret;
}

.visible .entry single_query_attention_kernel(
    .param .u64 single_query_attention_kernel_param_0,
    .param .u64 single_query_attention_kernel_param_1,
    .param .u64 single_query_attention_kernel_param_2,
    .param .u64 single_query_attention_kernel_param_3,
    .param .u32 single_query_attention_kernel_param_4,
    .param .u32 single_query_attention_kernel_param_5,
    .param .u32 single_query_attention_kernel_param_6,
    .param .u32 single_query_attention_kernel_param_7,
    .param .u32 single_query_attention_kernel_param_8,
    .param .u32 single_query_attention_kernel_param_9,
    .param .f32 single_query_attention_kernel_param_10,
    .param .u64 single_query_attention_kernel_param_11,
    .param .u32 single_query_attention_kernel_param_12,
    .param .u32 single_query_attention_kernel_param_13
)
{
    .reg .pred %p<8>;
    .reg .f32 %f<24>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<28>;

    ld.param.u64 %rd1, [single_query_attention_kernel_param_0];
    ld.param.u64 %rd2, [single_query_attention_kernel_param_1];
    ld.param.u64 %rd3, [single_query_attention_kernel_param_2];
    ld.param.u64 %rd4, [single_query_attention_kernel_param_3];
    ld.param.u32 %r1, [single_query_attention_kernel_param_4];
    ld.param.u32 %r2, [single_query_attention_kernel_param_5];
    ld.param.u32 %r3, [single_query_attention_kernel_param_6];
    ld.param.u32 %r4, [single_query_attention_kernel_param_7];
    ld.param.u32 %r5, [single_query_attention_kernel_param_8];
    ld.param.u32 %r6, [single_query_attention_kernel_param_9];
    ld.param.f32 %f1, [single_query_attention_kernel_param_10];
    ld.param.u64 %rd9, [single_query_attention_kernel_param_11];
    ld.param.u32 %r34, [single_query_attention_kernel_param_12];
    ld.param.u32 %r41, [single_query_attention_kernel_param_13];

    cvta.to.global.u64 %rd5, %rd1;
    cvta.to.global.u64 %rd6, %rd2;
    cvta.to.global.u64 %rd7, %rd3;
    cvta.to.global.u64 %rd8, %rd4;
    cvta.to.global.u64 %rd21, %rd9;

    mov.u32 %r7, %tid.x;
    mov.u32 %r8, %ctaid.x;
    mov.u32 %r9, %ntid.x;
    mad.lo.u32 %r10, %r8, %r9, %r7;
    setp.ge.u32 %p1, %r10, %r6;
    @%p1 bra SINGLE_ATTENTION_DONE;

    div.u32 %r11, %r10, %r3;
    mul.lo.u32 %r12, %r11, %r3;
    sub.u32 %r13, %r10, %r12;
    div.u32 %r14, %r1, %r2;
    div.u32 %r15, %r11, %r14;
    mov.f32 %f2, 0fFF800000;
    mov.u32 %r16, %r41;

SINGLE_ATTENTION_MAX_POS:
    setp.ge.u32 %p2, %r16, %r4;
    @%p2 bra SINGLE_ATTENTION_MAX_DONE;
    mov.f32 %f3, 0f00000000;
    mov.u32 %r17, 0;

SINGLE_ATTENTION_MAX_DOT:
    setp.ge.u32 %p3, %r17, %r3;
    @%p3 bra SINGLE_ATTENTION_MAX_DOT_DONE;
    mad.lo.u32 %r18, %r11, %r3, %r17;
    mul.wide.u32 %rd9, %r18, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f4, [%rd10];
    div.u32 %r35, %r16, %r34;
    mul.lo.u32 %r36, %r35, %r34;
    sub.u32 %r37, %r16, %r36;
    mul.wide.u32 %rd22, %r35, 4;
    add.s64 %rd23, %rd21, %rd22;
    ld.global.u32 %r38, [%rd23];
    mul.lo.u32 %r39, %r38, %r34;
    add.u32 %r40, %r39, %r37;
    mul.lo.u32 %r19, %r40, %r5;
    mul.lo.u32 %r20, %r15, %r3;
    add.u32 %r21, %r19, %r20;
    add.u32 %r22, %r21, %r17;
    mul.wide.u32 %rd11, %r22, 4;
    add.s64 %rd12, %rd6, %rd11;
    ld.global.f32 %f5, [%rd12];
    fma.rn.f32 %f3, %f4, %f5, %f3;
    add.u32 %r17, %r17, 1;
    bra SINGLE_ATTENTION_MAX_DOT;

SINGLE_ATTENTION_MAX_DOT_DONE:
    mul.f32 %f6, %f3, %f1;
    max.f32 %f2, %f2, %f6;
    add.u32 %r16, %r16, 1;
    bra SINGLE_ATTENTION_MAX_POS;

SINGLE_ATTENTION_MAX_DONE:
    mov.f32 %f7, 0f00000000;
    mov.f32 %f8, 0f00000000;
    mov.f32 %f9, 0f3FB8AA3B;
    mov.u32 %r23, %r41;

SINGLE_ATTENTION_SUM_POS:
    setp.ge.u32 %p4, %r23, %r4;
    @%p4 bra SINGLE_ATTENTION_SUM_DONE;
    mov.f32 %f10, 0f00000000;
    mov.u32 %r24, 0;

SINGLE_ATTENTION_SUM_DOT:
    setp.ge.u32 %p5, %r24, %r3;
    @%p5 bra SINGLE_ATTENTION_SUM_DOT_DONE;
    mad.lo.u32 %r25, %r11, %r3, %r24;
    mul.wide.u32 %rd13, %r25, 4;
    add.s64 %rd14, %rd5, %rd13;
    ld.global.f32 %f11, [%rd14];
    div.u32 %r35, %r23, %r34;
    mul.lo.u32 %r36, %r35, %r34;
    sub.u32 %r37, %r23, %r36;
    mul.wide.u32 %rd24, %r35, 4;
    add.s64 %rd25, %rd21, %rd24;
    ld.global.u32 %r38, [%rd25];
    mul.lo.u32 %r39, %r38, %r34;
    add.u32 %r40, %r39, %r37;
    mul.lo.u32 %r26, %r40, %r5;
    mul.lo.u32 %r27, %r15, %r3;
    add.u32 %r28, %r26, %r27;
    add.u32 %r29, %r28, %r24;
    mul.wide.u32 %rd15, %r29, 4;
    add.s64 %rd16, %rd6, %rd15;
    ld.global.f32 %f12, [%rd16];
    fma.rn.f32 %f10, %f11, %f12, %f10;
    add.u32 %r24, %r24, 1;
    bra SINGLE_ATTENTION_SUM_DOT;

SINGLE_ATTENTION_SUM_DOT_DONE:
    mul.f32 %f13, %f10, %f1;
    sub.f32 %f14, %f13, %f2;
    mul.f32 %f15, %f14, %f9;
    ex2.approx.f32 %f16, %f15;
    add.f32 %f7, %f7, %f16;
    mul.lo.u32 %r30, %r40, %r5;
    mul.lo.u32 %r31, %r15, %r3;
    add.u32 %r32, %r30, %r31;
    add.u32 %r33, %r32, %r13;
    mul.wide.u32 %rd17, %r33, 4;
    add.s64 %rd18, %rd7, %rd17;
    ld.global.f32 %f17, [%rd18];
    fma.rn.f32 %f8, %f16, %f17, %f8;
    add.u32 %r23, %r23, 1;
    bra SINGLE_ATTENTION_SUM_POS;

SINGLE_ATTENTION_SUM_DONE:
    div.rn.f32 %f18, %f8, %f7;
    mul.wide.u32 %rd19, %r10, 4;
    add.s64 %rd20, %rd8, %rd19;
    st.global.f32 [%rd20], %f18;

SINGLE_ATTENTION_DONE:
    ret;
}

.visible .entry single_query_attention_q8_online_kernel(
    .param .u64 single_query_attention_q8_online_kernel_param_0,
    .param .u64 single_query_attention_q8_online_kernel_param_1,
    .param .u64 single_query_attention_q8_online_kernel_param_2,
    .param .u64 single_query_attention_q8_online_kernel_param_3,
    .param .u64 single_query_attention_q8_online_kernel_param_4,
    .param .u64 single_query_attention_q8_online_kernel_param_5,
    .param .u32 single_query_attention_q8_online_kernel_param_6,
    .param .u32 single_query_attention_q8_online_kernel_param_7,
    .param .u32 single_query_attention_q8_online_kernel_param_8,
    .param .u32 single_query_attention_q8_online_kernel_param_9,
    .param .f32 single_query_attention_q8_online_kernel_param_10,
    .param .u64 single_query_attention_q8_online_kernel_param_11,
    .param .u32 single_query_attention_q8_online_kernel_param_12,
    .param .u32 single_query_attention_q8_online_kernel_param_13
)
{
    .shared .align 4 .b8 single_q8_attention_online_reduce[1024];
    .shared .align 4 .b8 single_q8_attention_online_state[16];
    .reg .pred %p<9>;
    .reg .f32 %f<26>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<36>;

    ld.param.u64 %rd1, [single_query_attention_q8_online_kernel_param_0];
    ld.param.u64 %rd2, [single_query_attention_q8_online_kernel_param_1];
    ld.param.u64 %rd3, [single_query_attention_q8_online_kernel_param_2];
    ld.param.u64 %rd4, [single_query_attention_q8_online_kernel_param_3];
    ld.param.u64 %rd5, [single_query_attention_q8_online_kernel_param_4];
    ld.param.u64 %rd6, [single_query_attention_q8_online_kernel_param_5];
    ld.param.u32 %r1, [single_query_attention_q8_online_kernel_param_6];
    ld.param.u32 %r2, [single_query_attention_q8_online_kernel_param_7];
    ld.param.u32 %r3, [single_query_attention_q8_online_kernel_param_8];
    ld.param.u32 %r4, [single_query_attention_q8_online_kernel_param_9];
    ld.param.f32 %f1, [single_query_attention_q8_online_kernel_param_10];
    ld.param.u64 %rd7, [single_query_attention_q8_online_kernel_param_11];
    ld.param.u32 %r5, [single_query_attention_q8_online_kernel_param_12];
    ld.param.u32 %r6, [single_query_attention_q8_online_kernel_param_13];

    cvta.to.global.u64 %rd8, %rd1;
    cvta.to.global.u64 %rd9, %rd2;
    cvta.to.global.u64 %rd10, %rd3;
    cvta.to.global.u64 %rd11, %rd4;
    cvta.to.global.u64 %rd12, %rd5;
    cvta.to.global.u64 %rd13, %rd6;
    cvta.to.global.u64 %rd14, %rd7;
    mov.u64 %rd15, single_q8_attention_online_reduce;
    mov.u64 %rd16, single_q8_attention_online_state;
    add.s64 %rd17, %rd16, 4;
    add.s64 %rd18, %rd16, 8;
    add.s64 %rd19, %rd16, 12;

    mov.u32 %r7, %tid.x;
    mov.u32 %r8, %ctaid.x;
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra SINGLE_Q8_ATTENTION_ONLINE_DONE;

    div.u32 %r9, %r1, %r2;
    div.u32 %r10, %r8, %r9;
    mul.lo.u32 %r16, %r2, %r3;
    mov.f32 %f17, 0f00000000;
    setp.ne.u32 %p2, %r7, 0;
    @%p2 bra SINGLE_Q8_ATTENTION_ONLINE_INIT_DONE;
    mov.f32 %f8, 0fFF800000;
    mov.f32 %f9, 0f00000000;
    st.shared.f32 [%rd16], %f8;
    st.shared.f32 [%rd17], %f9;

SINGLE_Q8_ATTENTION_ONLINE_INIT_DONE:
    bar.sync 0;
    mov.u32 %r20, %r6;

SINGLE_Q8_ATTENTION_ONLINE_POS:
    setp.ge.u32 %p3, %r20, %r4;
    @%p3 bra SINGLE_Q8_ATTENTION_ONLINE_WRITE;
    mov.f32 %f2, 0f00000000;
    setp.ge.u32 %p4, %r7, %r3;
    @%p4 bra SINGLE_Q8_ATTENTION_ONLINE_PARTIAL_DONE;

    mad.lo.u32 %r11, %r8, %r3, %r7;
    mul.wide.u32 %rd20, %r11, 4;
    add.s64 %rd20, %rd8, %rd20;
    ld.global.f32 %f3, [%rd20];
    div.u32 %r12, %r20, %r5;
    rem.u32 %r13, %r20, %r5;
    mul.wide.u32 %rd21, %r12, 4;
    add.s64 %rd21, %rd14, %rd21;
    ld.global.u32 %r14, [%rd21];
    mad.lo.u32 %r15, %r14, %r5, %r13;
    mad.lo.u32 %r17, %r10, %r3, %r7;
    mad.lo.u32 %r18, %r15, %r16, %r17;
    cvt.u64.u32 %rd22, %r18;
    add.s64 %rd22, %rd9, %rd22;
    ld.global.s8 %r19, [%rd22];
    cvt.rn.f32.s32 %f4, %r19;
    mul.wide.u32 %rd23, %r15, 4;
    add.s64 %rd23, %rd11, %rd23;
    ld.global.f32 %f5, [%rd23];
    mul.f32 %f6, %f4, %f5;
    mul.f32 %f2, %f3, %f6;

SINGLE_Q8_ATTENTION_ONLINE_PARTIAL_DONE:
    mul.wide.u32 %rd24, %r7, 4;
    add.s64 %rd24, %rd15, %rd24;
    st.shared.f32 [%rd24], %f2;
    bar.sync 0;
    mov.u32 %r21, 128;

SINGLE_Q8_ATTENTION_ONLINE_REDUCE:
    setp.eq.u32 %p5, %r21, 0;
    @%p5 bra SINGLE_Q8_ATTENTION_ONLINE_REDUCE_DONE;
    setp.ge.u32 %p6, %r7, %r21;
    @%p6 bra SINGLE_Q8_ATTENTION_ONLINE_REDUCE_SKIP;
    add.u32 %r22, %r7, %r21;
    mul.wide.u32 %rd25, %r22, 4;
    add.s64 %rd25, %rd15, %rd25;
    ld.shared.f32 %f18, [%rd24];
    ld.shared.f32 %f19, [%rd25];
    add.f32 %f20, %f18, %f19;
    st.shared.f32 [%rd24], %f20;

SINGLE_Q8_ATTENTION_ONLINE_REDUCE_SKIP:
    bar.sync 0;
    shr.u32 %r21, %r21, 1;
    bra SINGLE_Q8_ATTENTION_ONLINE_REDUCE;

SINGLE_Q8_ATTENTION_ONLINE_REDUCE_DONE:
    setp.ne.u32 %p7, %r7, 0;
    @%p7 bra SINGLE_Q8_ATTENTION_ONLINE_STATE_DONE;
    ld.shared.f32 %f7, [%rd15];
    mul.f32 %f7, %f7, %f1;
    ld.shared.f32 %f8, [%rd16];
    ld.shared.f32 %f9, [%rd17];
    max.f32 %f10, %f8, %f7;
    mov.f32 %f11, 0f3FB8AA3B;
    sub.f32 %f12, %f8, %f10;
    mul.f32 %f12, %f12, %f11;
    ex2.approx.f32 %f13, %f12;
    sub.f32 %f14, %f7, %f10;
    mul.f32 %f14, %f14, %f11;
    ex2.approx.f32 %f15, %f14;
    fma.rn.f32 %f16, %f9, %f13, %f15;
    st.shared.f32 [%rd16], %f10;
    st.shared.f32 [%rd17], %f16;
    st.shared.f32 [%rd18], %f13;
    st.shared.f32 [%rd19], %f15;

SINGLE_Q8_ATTENTION_ONLINE_STATE_DONE:
    bar.sync 0;
    setp.ge.u32 %p8, %r7, %r3;
    @%p8 bra SINGLE_Q8_ATTENTION_ONLINE_VALUE_DONE;
    ld.shared.f32 %f13, [%rd18];
    ld.shared.f32 %f15, [%rd19];
    cvt.u64.u32 %rd26, %r18;
    add.s64 %rd26, %rd10, %rd26;
    ld.global.s8 %r19, [%rd26];
    cvt.rn.f32.s32 %f21, %r19;
    mul.wide.u32 %rd27, %r15, 4;
    add.s64 %rd27, %rd12, %rd27;
    ld.global.f32 %f22, [%rd27];
    mul.f32 %f23, %f21, %f22;
    mul.f32 %f24, %f17, %f13;
    fma.rn.f32 %f17, %f15, %f23, %f24;

SINGLE_Q8_ATTENTION_ONLINE_VALUE_DONE:
    bar.sync 0;
    add.u32 %r20, %r20, 1;
    bra SINGLE_Q8_ATTENTION_ONLINE_POS;

SINGLE_Q8_ATTENTION_ONLINE_WRITE:
    setp.ge.u32 %p8, %r7, %r3;
    @%p8 bra SINGLE_Q8_ATTENTION_ONLINE_DONE;
    ld.shared.f32 %f9, [%rd17];
    div.rn.f32 %f25, %f17, %f9;
    mad.lo.u32 %r23, %r8, %r3, %r7;
    mul.wide.u32 %rd28, %r23, 4;
    add.s64 %rd28, %rd13, %rd28;
    st.global.f32 [%rd28], %f25;

SINGLE_Q8_ATTENTION_ONLINE_DONE:
    ret;
}

.visible .entry single_query_attention_q8_kernel(
    .param .u64 single_query_attention_q8_kernel_param_0,
    .param .u64 single_query_attention_q8_kernel_param_1,
    .param .u64 single_query_attention_q8_kernel_param_2,
    .param .u64 single_query_attention_q8_kernel_param_3,
    .param .u64 single_query_attention_q8_kernel_param_4,
    .param .u64 single_query_attention_q8_kernel_param_5,
    .param .u32 single_query_attention_q8_kernel_param_6,
    .param .u32 single_query_attention_q8_kernel_param_7,
    .param .u32 single_query_attention_q8_kernel_param_8,
    .param .u32 single_query_attention_q8_kernel_param_9,
    .param .f32 single_query_attention_q8_kernel_param_10,
    .param .u64 single_query_attention_q8_kernel_param_11,
    .param .u32 single_query_attention_q8_kernel_param_12,
    .param .u32 single_query_attention_q8_kernel_param_13
)
{
    .reg .pred %p<8>;
    .reg .f32 %f<34>;
    .reg .b32 %r<52>;
    .reg .b64 %rd<36>;

    ld.param.u64 %rd1, [single_query_attention_q8_kernel_param_0];
    ld.param.u64 %rd2, [single_query_attention_q8_kernel_param_1];
    ld.param.u64 %rd3, [single_query_attention_q8_kernel_param_2];
    ld.param.u64 %rd4, [single_query_attention_q8_kernel_param_3];
    ld.param.u64 %rd5, [single_query_attention_q8_kernel_param_4];
    ld.param.u64 %rd6, [single_query_attention_q8_kernel_param_5];
    ld.param.u32 %r1, [single_query_attention_q8_kernel_param_6];
    ld.param.u32 %r2, [single_query_attention_q8_kernel_param_7];
    ld.param.u32 %r3, [single_query_attention_q8_kernel_param_8];
    ld.param.u32 %r4, [single_query_attention_q8_kernel_param_9];
    ld.param.f32 %f1, [single_query_attention_q8_kernel_param_10];
    ld.param.u64 %rd30, [single_query_attention_q8_kernel_param_11];
    ld.param.u32 %r37, [single_query_attention_q8_kernel_param_12];
    ld.param.u32 %r50, [single_query_attention_q8_kernel_param_13];

    cvta.to.global.u64 %rd7, %rd1;
    cvta.to.global.u64 %rd8, %rd2;
    cvta.to.global.u64 %rd9, %rd3;
    cvta.to.global.u64 %rd10, %rd4;
    cvta.to.global.u64 %rd11, %rd5;
    cvta.to.global.u64 %rd12, %rd6;
    cvta.to.global.u64 %rd31, %rd30;

    mul.lo.u32 %r5, %r2, %r3;
    mul.lo.u32 %r6, %r1, %r3;

    mov.u32 %r7, %tid.x;
    mov.u32 %r8, %ctaid.x;
    mov.u32 %r9, %ntid.x;
    mad.lo.u32 %r10, %r8, %r9, %r7;
    setp.ge.u32 %p1, %r10, %r6;
    @%p1 bra SINGLE_Q8_ATTENTION_DONE;

    div.u32 %r11, %r10, %r3;
    mul.lo.u32 %r12, %r11, %r3;
    sub.u32 %r13, %r10, %r12;
    div.u32 %r14, %r1, %r2;
    div.u32 %r15, %r11, %r14;
    mov.f32 %f2, 0fFF800000;
    mov.u32 %r16, %r50;

SINGLE_Q8_ATTENTION_MAX_POS:
    setp.ge.u32 %p2, %r16, %r4;
    @%p2 bra SINGLE_Q8_ATTENTION_MAX_DONE;
    mov.f32 %f3, 0f00000000;
    div.u32 %r39, %r16, %r37;
    mul.lo.u32 %r40, %r39, %r37;
    sub.u32 %r41, %r16, %r40;
    mul.wide.u32 %rd32, %r39, 4;
    add.s64 %rd33, %rd31, %rd32;
    ld.global.u32 %r38, [%rd33];
    mul.lo.u32 %r38, %r38, %r37;
    add.u32 %r38, %r38, %r41;
    mul.wide.u32 %rd13, %r38, 4;
    add.s64 %rd14, %rd10, %rd13;
    ld.global.f32 %f4, [%rd14];
    mov.u32 %r17, 0;

SINGLE_Q8_ATTENTION_MAX_DOT:
    setp.ge.u32 %p3, %r17, %r3;
    @%p3 bra SINGLE_Q8_ATTENTION_MAX_DOT_DONE;
    mad.lo.u32 %r18, %r11, %r3, %r17;
    mul.wide.u32 %rd15, %r18, 4;
    add.s64 %rd16, %rd7, %rd15;
    ld.global.f32 %f5, [%rd16];
    mul.lo.u32 %r19, %r38, %r5;
    mul.lo.u32 %r20, %r15, %r3;
    add.u32 %r21, %r19, %r20;
    add.u32 %r22, %r21, %r17;
    cvt.u64.u32 %rd17, %r22;
    add.s64 %rd18, %rd8, %rd17;
    ld.global.s8 %r23, [%rd18];
    cvt.rn.f32.s32 %f6, %r23;
    mul.f32 %f7, %f6, %f4;
    fma.rn.f32 %f3, %f5, %f7, %f3;
    add.u32 %r17, %r17, 1;
    bra SINGLE_Q8_ATTENTION_MAX_DOT;

SINGLE_Q8_ATTENTION_MAX_DOT_DONE:
    mul.f32 %f8, %f3, %f1;
    max.f32 %f2, %f2, %f8;
    add.u32 %r16, %r16, 1;
    bra SINGLE_Q8_ATTENTION_MAX_POS;

SINGLE_Q8_ATTENTION_MAX_DONE:
    mov.f32 %f9, 0f00000000;
    mov.f32 %f10, 0f00000000;
    mov.f32 %f11, 0f3FB8AA3B;
    mov.u32 %r24, %r50;

SINGLE_Q8_ATTENTION_SUM_POS:
    setp.ge.u32 %p4, %r24, %r4;
    @%p4 bra SINGLE_Q8_ATTENTION_SUM_DONE;
    mov.f32 %f12, 0f00000000;
    div.u32 %r39, %r24, %r37;
    mul.lo.u32 %r40, %r39, %r37;
    sub.u32 %r41, %r24, %r40;
    mul.wide.u32 %rd32, %r39, 4;
    add.s64 %rd33, %rd31, %rd32;
    ld.global.u32 %r38, [%rd33];
    mul.lo.u32 %r38, %r38, %r37;
    add.u32 %r38, %r38, %r41;
    mul.wide.u32 %rd19, %r38, 4;
    add.s64 %rd20, %rd10, %rd19;
    ld.global.f32 %f13, [%rd20];
    add.s64 %rd21, %rd11, %rd19;
    ld.global.f32 %f14, [%rd21];
    mov.u32 %r25, 0;

SINGLE_Q8_ATTENTION_SUM_DOT:
    setp.ge.u32 %p5, %r25, %r3;
    @%p5 bra SINGLE_Q8_ATTENTION_SUM_DOT_DONE;
    mad.lo.u32 %r26, %r11, %r3, %r25;
    mul.wide.u32 %rd22, %r26, 4;
    add.s64 %rd23, %rd7, %rd22;
    ld.global.f32 %f15, [%rd23];
    mul.lo.u32 %r27, %r38, %r5;
    mul.lo.u32 %r28, %r15, %r3;
    add.u32 %r29, %r27, %r28;
    add.u32 %r30, %r29, %r25;
    cvt.u64.u32 %rd24, %r30;
    add.s64 %rd25, %rd8, %rd24;
    ld.global.s8 %r31, [%rd25];
    cvt.rn.f32.s32 %f16, %r31;
    mul.f32 %f17, %f16, %f13;
    fma.rn.f32 %f12, %f15, %f17, %f12;
    add.u32 %r25, %r25, 1;
    bra SINGLE_Q8_ATTENTION_SUM_DOT;

SINGLE_Q8_ATTENTION_SUM_DOT_DONE:
    mul.f32 %f18, %f12, %f1;
    sub.f32 %f19, %f18, %f2;
    mul.f32 %f20, %f19, %f11;
    ex2.approx.f32 %f21, %f20;
    add.f32 %f9, %f9, %f21;
    mul.lo.u32 %r32, %r38, %r5;
    mul.lo.u32 %r33, %r15, %r3;
    add.u32 %r34, %r32, %r33;
    add.u32 %r35, %r34, %r13;
    cvt.u64.u32 %rd26, %r35;
    add.s64 %rd27, %rd9, %rd26;
    ld.global.s8 %r36, [%rd27];
    cvt.rn.f32.s32 %f22, %r36;
    mul.f32 %f23, %f22, %f14;
    fma.rn.f32 %f10, %f21, %f23, %f10;
    add.u32 %r24, %r24, 1;
    bra SINGLE_Q8_ATTENTION_SUM_POS;

SINGLE_Q8_ATTENTION_SUM_DONE:
    div.rn.f32 %f24, %f10, %f9;
    mul.wide.u32 %rd28, %r10, 4;
    add.s64 %rd29, %rd12, %rd28;
    st.global.f32 [%rd29], %f24;

SINGLE_Q8_ATTENTION_DONE:
    ret;
}

.visible .entry single_query_attention_kq4_vq8_online_kernel(
    .param .u64 single_query_attention_kq4_vq8_online_kernel_param_0,
    .param .u64 single_query_attention_kq4_vq8_online_kernel_param_1,
    .param .u64 single_query_attention_kq4_vq8_online_kernel_param_2,
    .param .u64 single_query_attention_kq4_vq8_online_kernel_param_3,
    .param .u64 single_query_attention_kq4_vq8_online_kernel_param_4,
    .param .u64 single_query_attention_kq4_vq8_online_kernel_param_5,
    .param .u32 single_query_attention_kq4_vq8_online_kernel_param_6,
    .param .u32 single_query_attention_kq4_vq8_online_kernel_param_7,
    .param .u32 single_query_attention_kq4_vq8_online_kernel_param_8,
    .param .u32 single_query_attention_kq4_vq8_online_kernel_param_9,
    .param .f32 single_query_attention_kq4_vq8_online_kernel_param_10,
    .param .u64 single_query_attention_kq4_vq8_online_kernel_param_11,
    .param .u32 single_query_attention_kq4_vq8_online_kernel_param_12,
    .param .u32 single_query_attention_kq4_vq8_online_kernel_param_13
)
{
    .shared .align 4 .b8 single_kq4_vq8_attention_online_reduce[1024];
    .shared .align 4 .b8 single_kq4_vq8_attention_online_state[16];
    .reg .pred %p<12>;
    .reg .f32 %f<26>;
    .reg .b32 %r<56>;
    .reg .b64 %rd<38>;

    ld.param.u64 %rd1, [single_query_attention_kq4_vq8_online_kernel_param_0];
    ld.param.u64 %rd2, [single_query_attention_kq4_vq8_online_kernel_param_1];
    ld.param.u64 %rd3, [single_query_attention_kq4_vq8_online_kernel_param_2];
    ld.param.u64 %rd4, [single_query_attention_kq4_vq8_online_kernel_param_3];
    ld.param.u64 %rd5, [single_query_attention_kq4_vq8_online_kernel_param_4];
    ld.param.u64 %rd6, [single_query_attention_kq4_vq8_online_kernel_param_5];
    ld.param.u32 %r1, [single_query_attention_kq4_vq8_online_kernel_param_6];
    ld.param.u32 %r2, [single_query_attention_kq4_vq8_online_kernel_param_7];
    ld.param.u32 %r3, [single_query_attention_kq4_vq8_online_kernel_param_8];
    ld.param.u32 %r4, [single_query_attention_kq4_vq8_online_kernel_param_9];
    ld.param.f32 %f1, [single_query_attention_kq4_vq8_online_kernel_param_10];
    ld.param.u64 %rd7, [single_query_attention_kq4_vq8_online_kernel_param_11];
    ld.param.u32 %r5, [single_query_attention_kq4_vq8_online_kernel_param_12];
    ld.param.u32 %r6, [single_query_attention_kq4_vq8_online_kernel_param_13];

    cvta.to.global.u64 %rd8, %rd1;
    cvta.to.global.u64 %rd9, %rd2;
    cvta.to.global.u64 %rd10, %rd3;
    cvta.to.global.u64 %rd11, %rd4;
    cvta.to.global.u64 %rd12, %rd5;
    cvta.to.global.u64 %rd13, %rd6;
    cvta.to.global.u64 %rd14, %rd7;
    mov.u64 %rd15, single_kq4_vq8_attention_online_reduce;
    mov.u64 %rd16, single_kq4_vq8_attention_online_state;
    add.s64 %rd17, %rd16, 4;
    add.s64 %rd18, %rd16, 8;
    add.s64 %rd19, %rd16, 12;

    mov.u32 %r7, %tid.x;
    mov.u32 %r8, %ctaid.x;
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_DONE;

    div.u32 %r9, %r1, %r2;
    div.u32 %r10, %r8, %r9;
    mul.lo.u32 %r11, %r2, %r3;
    add.u32 %r12, %r11, 1;
    shr.u32 %r12, %r12, 1;
    add.u32 %r13, %r11, 63;
    shr.u32 %r13, %r13, 6;
    mov.f32 %f17, 0f00000000;
    setp.ne.u32 %p2, %r7, 0;
    @%p2 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_INIT_DONE;
    mov.f32 %f8, 0fFF800000;
    mov.f32 %f9, 0f00000000;
    st.shared.f32 [%rd16], %f8;
    st.shared.f32 [%rd17], %f9;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_INIT_DONE:
    bar.sync 0;
    mov.u32 %r23, %r6;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_POS:
    setp.ge.u32 %p3, %r23, %r4;
    @%p3 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_WRITE;
    mov.f32 %f2, 0f00000000;
    setp.ge.u32 %p4, %r7, %r3;
    @%p4 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_PARTIAL_DONE;

    mad.lo.u32 %r27, %r8, %r3, %r7;
    mul.wide.u32 %rd20, %r27, 4;
    add.s64 %rd20, %rd8, %rd20;
    ld.global.f32 %f3, [%rd20];
    div.u32 %r14, %r23, %r5;
    rem.u32 %r15, %r23, %r5;
    mul.wide.u32 %rd21, %r14, 4;
    add.s64 %rd21, %rd14, %rd21;
    ld.global.u32 %r16, [%rd21];
    mad.lo.u32 %r17, %r16, %r5, %r15;
    mad.lo.u32 %r18, %r10, %r3, %r7;
    shr.u32 %r19, %r18, 6;
    mad.lo.u32 %r19, %r17, %r13, %r19;
    mul.wide.u32 %rd22, %r19, 4;
    add.s64 %rd22, %rd11, %rd22;
    ld.global.f32 %f5, [%rd22];
    shr.u32 %r20, %r18, 1;
    mad.lo.u32 %r20, %r17, %r12, %r20;
    cvt.u64.u32 %rd23, %r20;
    add.s64 %rd23, %rd9, %rd23;
    ld.global.u8 %r21, [%rd23];
    and.b32 %r22, %r18, 1;
    setp.eq.u32 %p5, %r22, 0;
    @%p5 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_KEY_LOW;
    shr.u32 %r21, %r21, 4;
    bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_KEY_READY;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_KEY_LOW:
    and.b32 %r21, %r21, 15;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_KEY_READY:
    cvt.s32.u32 %r22, %r21;
    add.s32 %r22, %r22, -8;
    cvt.rn.f32.s32 %f4, %r22;
    mul.f32 %f6, %f4, %f5;
    mul.f32 %f2, %f3, %f6;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_PARTIAL_DONE:
    mul.wide.u32 %rd24, %r7, 4;
    add.s64 %rd24, %rd15, %rd24;
    st.shared.f32 [%rd24], %f2;
    bar.sync 0;
    mov.u32 %r24, 128;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_REDUCE:
    setp.eq.u32 %p6, %r24, 0;
    @%p6 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_REDUCE_DONE;
    setp.ge.u32 %p7, %r7, %r24;
    @%p7 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_REDUCE_SKIP;
    add.u32 %r25, %r7, %r24;
    mul.wide.u32 %rd25, %r25, 4;
    add.s64 %rd25, %rd15, %rd25;
    ld.shared.f32 %f18, [%rd24];
    ld.shared.f32 %f19, [%rd25];
    add.f32 %f20, %f18, %f19;
    st.shared.f32 [%rd24], %f20;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_REDUCE_SKIP:
    bar.sync 0;
    shr.u32 %r24, %r24, 1;
    bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_REDUCE;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_REDUCE_DONE:
    setp.ne.u32 %p8, %r7, 0;
    @%p8 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_STATE_DONE;
    ld.shared.f32 %f7, [%rd15];
    mul.f32 %f7, %f7, %f1;
    ld.shared.f32 %f8, [%rd16];
    ld.shared.f32 %f9, [%rd17];
    max.f32 %f10, %f8, %f7;
    mov.f32 %f11, 0f3FB8AA3B;
    sub.f32 %f12, %f8, %f10;
    mul.f32 %f12, %f12, %f11;
    ex2.approx.f32 %f13, %f12;
    sub.f32 %f14, %f7, %f10;
    mul.f32 %f14, %f14, %f11;
    ex2.approx.f32 %f15, %f14;
    fma.rn.f32 %f16, %f9, %f13, %f15;
    st.shared.f32 [%rd16], %f10;
    st.shared.f32 [%rd17], %f16;
    st.shared.f32 [%rd18], %f13;
    st.shared.f32 [%rd19], %f15;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_STATE_DONE:
    bar.sync 0;
    setp.ge.u32 %p9, %r7, %r3;
    @%p9 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_VALUE_DONE;
    ld.shared.f32 %f13, [%rd18];
    ld.shared.f32 %f15, [%rd19];
    mad.lo.u32 %r28, %r17, %r11, %r18;
    cvt.u64.u32 %rd26, %r28;
    add.s64 %rd26, %rd10, %rd26;
    ld.global.s8 %r29, [%rd26];
    cvt.rn.f32.s32 %f21, %r29;
    mul.wide.u32 %rd27, %r17, 4;
    add.s64 %rd27, %rd12, %rd27;
    ld.global.f32 %f22, [%rd27];
    mul.f32 %f23, %f21, %f22;
    mul.f32 %f24, %f17, %f13;
    fma.rn.f32 %f17, %f15, %f23, %f24;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_VALUE_DONE:
    bar.sync 0;
    add.u32 %r23, %r23, 1;
    bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_POS;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_WRITE:
    setp.ge.u32 %p10, %r7, %r3;
    @%p10 bra SINGLE_KQ4VQ8_ATTENTION_ONLINE_DONE;
    ld.shared.f32 %f9, [%rd17];
    div.rn.f32 %f25, %f17, %f9;
    mad.lo.u32 %r26, %r8, %r3, %r7;
    mul.wide.u32 %rd28, %r26, 4;
    add.s64 %rd28, %rd13, %rd28;
    st.global.f32 [%rd28], %f25;

SINGLE_KQ4VQ8_ATTENTION_ONLINE_DONE:
    ret;
}

.visible .entry single_query_attention_kq4_vq8_kernel(
    .param .u64 single_query_attention_kq4_vq8_kernel_param_0,
    .param .u64 single_query_attention_kq4_vq8_kernel_param_1,
    .param .u64 single_query_attention_kq4_vq8_kernel_param_2,
    .param .u64 single_query_attention_kq4_vq8_kernel_param_3,
    .param .u64 single_query_attention_kq4_vq8_kernel_param_4,
    .param .u64 single_query_attention_kq4_vq8_kernel_param_5,
    .param .u32 single_query_attention_kq4_vq8_kernel_param_6,
    .param .u32 single_query_attention_kq4_vq8_kernel_param_7,
    .param .u32 single_query_attention_kq4_vq8_kernel_param_8,
    .param .u32 single_query_attention_kq4_vq8_kernel_param_9,
    .param .f32 single_query_attention_kq4_vq8_kernel_param_10,
    .param .u64 single_query_attention_kq4_vq8_kernel_param_11,
    .param .u32 single_query_attention_kq4_vq8_kernel_param_12,
    .param .u32 single_query_attention_kq4_vq8_kernel_param_13
)
{
    .reg .pred %p<12>;
    .reg .f32 %f<34>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<42>;

    ld.param.u64 %rd1, [single_query_attention_kq4_vq8_kernel_param_0];
    ld.param.u64 %rd2, [single_query_attention_kq4_vq8_kernel_param_1];
    ld.param.u64 %rd3, [single_query_attention_kq4_vq8_kernel_param_2];
    ld.param.u64 %rd4, [single_query_attention_kq4_vq8_kernel_param_3];
    ld.param.u64 %rd5, [single_query_attention_kq4_vq8_kernel_param_4];
    ld.param.u64 %rd6, [single_query_attention_kq4_vq8_kernel_param_5];
    ld.param.u32 %r1, [single_query_attention_kq4_vq8_kernel_param_6];
    ld.param.u32 %r2, [single_query_attention_kq4_vq8_kernel_param_7];
    ld.param.u32 %r3, [single_query_attention_kq4_vq8_kernel_param_8];
    ld.param.u32 %r4, [single_query_attention_kq4_vq8_kernel_param_9];
    ld.param.f32 %f1, [single_query_attention_kq4_vq8_kernel_param_10];
    ld.param.u64 %rd31, [single_query_attention_kq4_vq8_kernel_param_11];
    ld.param.u32 %r53, [single_query_attention_kq4_vq8_kernel_param_12];
    ld.param.u32 %r63, [single_query_attention_kq4_vq8_kernel_param_13];

    cvta.to.global.u64 %rd7, %rd1;
    cvta.to.global.u64 %rd8, %rd2;
    cvta.to.global.u64 %rd9, %rd3;
    cvta.to.global.u64 %rd10, %rd4;
    cvta.to.global.u64 %rd11, %rd5;
    cvta.to.global.u64 %rd12, %rd6;
    cvta.to.global.u64 %rd32, %rd31;

    mul.lo.u32 %r5, %r2, %r3;
    add.u32 %r6, %r5, 1;
    shr.u32 %r7, %r6, 1;
    add.u32 %r8, %r5, 63;
    shr.u32 %r9, %r8, 6;
    mul.lo.u32 %r10, %r1, %r3;

    mov.u32 %r11, %tid.x;
    mov.u32 %r12, %ctaid.x;
    mov.u32 %r13, %ntid.x;
    mad.lo.u32 %r14, %r12, %r13, %r11;
    setp.ge.u32 %p1, %r14, %r10;
    @%p1 bra SINGLE_KQ4VQ8_ATTENTION_DONE;

    div.u32 %r15, %r14, %r3;
    mul.lo.u32 %r16, %r15, %r3;
    sub.u32 %r17, %r14, %r16;
    div.u32 %r18, %r1, %r2;
    div.u32 %r19, %r15, %r18;
    mov.f32 %f2, 0fFF800000;
    mov.u32 %r20, %r63;

SINGLE_KQ4VQ8_ATTENTION_MAX_POS:
    setp.ge.u32 %p2, %r20, %r4;
    @%p2 bra SINGLE_KQ4VQ8_ATTENTION_MAX_DONE;
    mov.f32 %f3, 0f00000000;
    div.u32 %r54, %r20, %r53;
    mul.lo.u32 %r55, %r54, %r53;
    sub.u32 %r56, %r20, %r55;
    mul.wide.u32 %rd33, %r54, 4;
    add.s64 %rd34, %rd32, %rd33;
    ld.global.u32 %r57, [%rd34];
    mul.lo.u32 %r57, %r57, %r53;
    add.u32 %r57, %r57, %r56;
    mul.lo.u32 %r21, %r57, %r9;
    mov.u32 %r23, 0;

SINGLE_KQ4VQ8_ATTENTION_MAX_DOT:
    setp.ge.u32 %p3, %r23, %r3;
    @%p3 bra SINGLE_KQ4VQ8_ATTENTION_MAX_DOT_DONE;
    mad.lo.u32 %r24, %r15, %r3, %r23;
    mul.wide.u32 %rd15, %r24, 4;
    add.s64 %rd16, %rd7, %rd15;
    ld.global.f32 %f5, [%rd16];
    mul.lo.u32 %r25, %r57, %r7;
    mul.lo.u32 %r26, %r19, %r3;
    add.u32 %r27, %r26, %r23;
    shr.u32 %r28, %r27, 6;
    add.u32 %r29, %r21, %r28;
    mul.wide.u32 %rd13, %r29, 4;
    add.s64 %rd14, %rd10, %rd13;
    ld.global.f32 %f4, [%rd14];
    shr.u32 %r28, %r27, 1;
    add.u32 %r29, %r25, %r28;
    cvt.u64.u32 %rd17, %r29;
    add.s64 %rd18, %rd8, %rd17;
    ld.global.u8 %r30, [%rd18];
    and.b32 %r31, %r27, 1;
    setp.eq.u32 %p4, %r31, 0;
    @%p4 bra SINGLE_KQ4VQ8_MAX_KEY_LOW;
    shr.u32 %r32, %r30, 4;
    bra SINGLE_KQ4VQ8_MAX_KEY_READY;

SINGLE_KQ4VQ8_MAX_KEY_LOW:
    and.b32 %r32, %r30, 15;

SINGLE_KQ4VQ8_MAX_KEY_READY:
    cvt.s32.u32 %r33, %r32;
    add.s32 %r33, %r33, -8;
    cvt.rn.f32.s32 %f6, %r33;
    mul.f32 %f7, %f6, %f4;
    fma.rn.f32 %f3, %f5, %f7, %f3;
    add.u32 %r23, %r23, 1;
    bra SINGLE_KQ4VQ8_ATTENTION_MAX_DOT;

SINGLE_KQ4VQ8_ATTENTION_MAX_DOT_DONE:
    mul.f32 %f8, %f3, %f1;
    max.f32 %f2, %f2, %f8;
    add.u32 %r20, %r20, 1;
    bra SINGLE_KQ4VQ8_ATTENTION_MAX_POS;

SINGLE_KQ4VQ8_ATTENTION_MAX_DONE:
    mov.f32 %f9, 0f00000000;
    mov.f32 %f10, 0f00000000;
    mov.f32 %f11, 0f3FB8AA3B;
    mov.u32 %r34, %r63;

SINGLE_KQ4VQ8_ATTENTION_SUM_POS:
    setp.ge.u32 %p5, %r34, %r4;
    @%p5 bra SINGLE_KQ4VQ8_ATTENTION_SUM_DONE;
    mov.f32 %f12, 0f00000000;
    div.u32 %r54, %r34, %r53;
    mul.lo.u32 %r55, %r54, %r53;
    sub.u32 %r56, %r34, %r55;
    mul.wide.u32 %rd33, %r54, 4;
    add.s64 %rd34, %rd32, %rd33;
    ld.global.u32 %r57, [%rd34];
    mul.lo.u32 %r57, %r57, %r53;
    add.u32 %r57, %r57, %r56;
    mul.lo.u32 %r35, %r57, %r9;
    mul.wide.u32 %rd21, %r57, 4;
    add.s64 %rd22, %rd11, %rd21;
    ld.global.f32 %f14, [%rd22];
    mov.u32 %r37, 0;

SINGLE_KQ4VQ8_ATTENTION_SUM_DOT:
    setp.ge.u32 %p6, %r37, %r3;
    @%p6 bra SINGLE_KQ4VQ8_ATTENTION_SUM_DOT_DONE;
    mad.lo.u32 %r38, %r15, %r3, %r37;
    mul.wide.u32 %rd23, %r38, 4;
    add.s64 %rd24, %rd7, %rd23;
    ld.global.f32 %f15, [%rd24];
    mul.lo.u32 %r39, %r57, %r7;
    mul.lo.u32 %r40, %r19, %r3;
    add.u32 %r41, %r40, %r37;
    shr.u32 %r42, %r41, 6;
    add.u32 %r43, %r35, %r42;
    mul.wide.u32 %rd19, %r43, 4;
    add.s64 %rd20, %rd10, %rd19;
    ld.global.f32 %f13, [%rd20];
    shr.u32 %r42, %r41, 1;
    add.u32 %r43, %r39, %r42;
    cvt.u64.u32 %rd25, %r43;
    add.s64 %rd26, %rd8, %rd25;
    ld.global.u8 %r44, [%rd26];
    and.b32 %r45, %r41, 1;
    setp.eq.u32 %p7, %r45, 0;
    @%p7 bra SINGLE_KQ4VQ8_SUM_KEY_LOW;
    shr.u32 %r46, %r44, 4;
    bra SINGLE_KQ4VQ8_SUM_KEY_READY;

SINGLE_KQ4VQ8_SUM_KEY_LOW:
    and.b32 %r46, %r44, 15;

SINGLE_KQ4VQ8_SUM_KEY_READY:
    cvt.s32.u32 %r47, %r46;
    add.s32 %r47, %r47, -8;
    cvt.rn.f32.s32 %f16, %r47;
    mul.f32 %f17, %f16, %f13;
    fma.rn.f32 %f12, %f15, %f17, %f12;
    add.u32 %r37, %r37, 1;
    bra SINGLE_KQ4VQ8_ATTENTION_SUM_DOT;

SINGLE_KQ4VQ8_ATTENTION_SUM_DOT_DONE:
    mul.f32 %f18, %f12, %f1;
    sub.f32 %f19, %f18, %f2;
    mul.f32 %f20, %f19, %f11;
    ex2.approx.f32 %f21, %f20;
    add.f32 %f9, %f9, %f21;
    mul.lo.u32 %r48, %r57, %r5;
    mul.lo.u32 %r49, %r19, %r3;
    add.u32 %r50, %r48, %r49;
    add.u32 %r51, %r50, %r17;
    cvt.u64.u32 %rd27, %r51;
    add.s64 %rd28, %rd9, %rd27;
    ld.global.s8 %r52, [%rd28];
    cvt.rn.f32.s32 %f22, %r52;
    mul.f32 %f23, %f22, %f14;
    fma.rn.f32 %f10, %f21, %f23, %f10;
    add.u32 %r34, %r34, 1;
    bra SINGLE_KQ4VQ8_ATTENTION_SUM_POS;

SINGLE_KQ4VQ8_ATTENTION_SUM_DONE:
    div.rn.f32 %f24, %f10, %f9;
    mul.wide.u32 %rd29, %r14, 4;
    add.s64 %rd30, %rd12, %rd29;
    st.global.f32 [%rd30], %f24;

SINGLE_KQ4VQ8_ATTENTION_DONE:
    ret;
}

.visible .entry single_query_attention_mixed_kq4_vq8_kernel(
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_0,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_1,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_2,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_3,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_4,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_5,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_6,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_7,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_8,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_9,
    .param .u64 single_query_attention_mixed_kq4_vq8_kernel_param_10,
    .param .u32 single_query_attention_mixed_kq4_vq8_kernel_param_11,
    .param .u32 single_query_attention_mixed_kq4_vq8_kernel_param_12,
    .param .u32 single_query_attention_mixed_kq4_vq8_kernel_param_13,
    .param .u32 single_query_attention_mixed_kq4_vq8_kernel_param_14,
    .param .f32 single_query_attention_mixed_kq4_vq8_kernel_param_15,
    .param .u32 single_query_attention_mixed_kq4_vq8_kernel_param_16,
    .param .u32 single_query_attention_mixed_kq4_vq8_kernel_param_17,
    .param .u32 single_query_attention_mixed_kq4_vq8_kernel_param_18
)
{
    .reg .pred %p<16>;
    .reg .f32 %f<32>;
    .reg .b32 %r<72>;
    .reg .b64 %rd<64>;

    ld.param.u64 %rd1, [single_query_attention_mixed_kq4_vq8_kernel_param_0];
    ld.param.u64 %rd2, [single_query_attention_mixed_kq4_vq8_kernel_param_1];
    ld.param.u64 %rd3, [single_query_attention_mixed_kq4_vq8_kernel_param_2];
    ld.param.u64 %rd4, [single_query_attention_mixed_kq4_vq8_kernel_param_3];
    ld.param.u64 %rd5, [single_query_attention_mixed_kq4_vq8_kernel_param_4];
    ld.param.u64 %rd6, [single_query_attention_mixed_kq4_vq8_kernel_param_5];
    ld.param.u64 %rd7, [single_query_attention_mixed_kq4_vq8_kernel_param_6];
    ld.param.u64 %rd8, [single_query_attention_mixed_kq4_vq8_kernel_param_7];
    ld.param.u64 %rd9, [single_query_attention_mixed_kq4_vq8_kernel_param_8];
    ld.param.u64 %rd10, [single_query_attention_mixed_kq4_vq8_kernel_param_9];
    ld.param.u64 %rd11, [single_query_attention_mixed_kq4_vq8_kernel_param_10];
    ld.param.u32 %r1, [single_query_attention_mixed_kq4_vq8_kernel_param_11];
    ld.param.u32 %r2, [single_query_attention_mixed_kq4_vq8_kernel_param_12];
    ld.param.u32 %r3, [single_query_attention_mixed_kq4_vq8_kernel_param_13];
    ld.param.u32 %r4, [single_query_attention_mixed_kq4_vq8_kernel_param_14];
    ld.param.f32 %f1, [single_query_attention_mixed_kq4_vq8_kernel_param_15];
    ld.param.u32 %r5, [single_query_attention_mixed_kq4_vq8_kernel_param_16];
    ld.param.u32 %r6, [single_query_attention_mixed_kq4_vq8_kernel_param_17];
    ld.param.u32 %r7, [single_query_attention_mixed_kq4_vq8_kernel_param_18];

    cvta.to.global.u64 %rd12, %rd1;
    cvta.to.global.u64 %rd13, %rd2;
    cvta.to.global.u64 %rd14, %rd3;
    cvta.to.global.u64 %rd15, %rd4;
    cvta.to.global.u64 %rd16, %rd5;
    cvta.to.global.u64 %rd17, %rd6;
    cvta.to.global.u64 %rd18, %rd7;
    cvta.to.global.u64 %rd19, %rd8;
    cvta.to.global.u64 %rd20, %rd9;
    cvta.to.global.u64 %rd21, %rd10;
    cvta.to.global.u64 %rd22, %rd11;

    mul.lo.u32 %r8, %r2, %r3;
    add.u32 %r9, %r8, 1;
    shr.u32 %r9, %r9, 1;
    add.u32 %r10, %r8, 63;
    shr.u32 %r10, %r10, 6;
    mul.lo.u32 %r11, %r1, %r3;

    mov.u32 %r12, %tid.x;
    mov.u32 %r13, %ctaid.x;
    mov.u32 %r14, %ntid.x;
    mad.lo.u32 %r15, %r13, %r14, %r12;
    setp.ge.u32 %p1, %r15, %r11;
    @%p1 bra SINGLE_MIXED_ATTENTION_DONE;

    div.u32 %r16, %r15, %r3;
    mul.lo.u32 %r17, %r16, %r3;
    sub.u32 %r18, %r15, %r17;
    div.u32 %r19, %r1, %r2;
    div.u32 %r20, %r16, %r19;
    mov.f32 %f2, 0fFF800000;
    mov.u32 %r21, %r7;

SINGLE_MIXED_MAX_POS:
    setp.ge.u32 %p2, %r21, %r4;
    @%p2 bra SINGLE_MIXED_MAX_DONE;
    mul.wide.u32 %rd23, %r21, 4;
    add.s64 %rd24, %rd20, %rd23;
    ld.global.u32 %r22, [%rd24];
    and.b32 %r23, %r22, 2147483647;
    setp.ne.u32 %p3, %r22, %r23;
    @%p3 bra SINGLE_MIXED_MAX_HOT_PAGE;

    div.u32 %r24, %r23, %r6;
    rem.u32 %r25, %r23, %r6;
    mul.wide.u32 %rd25, %r24, 4;
    add.s64 %rd26, %rd22, %rd25;
    ld.global.u32 %r26, [%rd26];
    mad.lo.u32 %r27, %r26, %r6, %r25;
    bra SINGLE_MIXED_MAX_PAGE_READY;

SINGLE_MIXED_MAX_HOT_PAGE:
    div.u32 %r24, %r23, %r5;
    rem.u32 %r25, %r23, %r5;
    mul.wide.u32 %rd25, %r24, 4;
    add.s64 %rd26, %rd21, %rd25;
    ld.global.u32 %r26, [%rd26];
    mad.lo.u32 %r27, %r26, %r5, %r25;

SINGLE_MIXED_MAX_PAGE_READY:
    mov.f32 %f3, 0f00000000;
    mov.u32 %r28, 0;

SINGLE_MIXED_MAX_DOT:
    setp.ge.u32 %p4, %r28, %r3;
    @%p4 bra SINGLE_MIXED_MAX_DOT_DONE;
    add.u32 %r29, %r17, %r28;
    mul.wide.u32 %rd27, %r29, 4;
    add.s64 %rd28, %rd12, %rd27;
    ld.global.f32 %f4, [%rd28];
    mad.lo.u32 %r30, %r20, %r3, %r28;
    @%p3 bra SINGLE_MIXED_MAX_HOT_KEY;

    shr.u32 %r31, %r30, 6;
    mad.lo.u32 %r32, %r27, %r10, %r31;
    mul.wide.u32 %rd29, %r32, 4;
    add.s64 %rd30, %rd17, %rd29;
    ld.global.f32 %f5, [%rd30];
    shr.u32 %r33, %r30, 1;
    mad.lo.u32 %r34, %r27, %r9, %r33;
    cvt.u64.u32 %rd31, %r34;
    add.s64 %rd32, %rd15, %rd31;
    ld.global.u8 %r35, [%rd32];
    and.b32 %r36, %r30, 1;
    setp.eq.u32 %p5, %r36, 0;
    @%p5 bra SINGLE_MIXED_MAX_COLD_KEY_LOW;
    shr.u32 %r37, %r35, 4;
    bra SINGLE_MIXED_MAX_COLD_KEY_READY;

SINGLE_MIXED_MAX_COLD_KEY_LOW:
    and.b32 %r37, %r35, 15;

SINGLE_MIXED_MAX_COLD_KEY_READY:
    cvt.s32.u32 %r38, %r37;
    add.s32 %r38, %r38, -8;
    cvt.rn.f32.s32 %f6, %r38;
    mul.f32 %f7, %f6, %f5;
    bra SINGLE_MIXED_MAX_KEY_READY;

SINGLE_MIXED_MAX_HOT_KEY:
    mad.lo.u32 %r39, %r27, %r8, %r30;
    mul.wide.u32 %rd33, %r39, 4;
    add.s64 %rd34, %rd13, %rd33;
    ld.global.f32 %f7, [%rd34];

SINGLE_MIXED_MAX_KEY_READY:
    fma.rn.f32 %f3, %f4, %f7, %f3;
    add.u32 %r28, %r28, 1;
    bra SINGLE_MIXED_MAX_DOT;

SINGLE_MIXED_MAX_DOT_DONE:
    mul.f32 %f8, %f3, %f1;
    max.f32 %f2, %f2, %f8;
    add.u32 %r21, %r21, 1;
    bra SINGLE_MIXED_MAX_POS;

SINGLE_MIXED_MAX_DONE:
    mov.f32 %f9, 0f00000000;
    mov.f32 %f10, 0f00000000;
    mov.f32 %f11, 0f3FB8AA3B;
    mov.u32 %r40, %r7;

SINGLE_MIXED_SUM_POS:
    setp.ge.u32 %p6, %r40, %r4;
    @%p6 bra SINGLE_MIXED_SUM_DONE;
    mul.wide.u32 %rd35, %r40, 4;
    add.s64 %rd36, %rd20, %rd35;
    ld.global.u32 %r41, [%rd36];
    and.b32 %r42, %r41, 2147483647;
    setp.ne.u32 %p7, %r41, %r42;
    @%p7 bra SINGLE_MIXED_SUM_HOT_PAGE;

    div.u32 %r43, %r42, %r6;
    rem.u32 %r44, %r42, %r6;
    mul.wide.u32 %rd37, %r43, 4;
    add.s64 %rd38, %rd22, %rd37;
    ld.global.u32 %r45, [%rd38];
    mad.lo.u32 %r46, %r45, %r6, %r44;
    bra SINGLE_MIXED_SUM_PAGE_READY;

SINGLE_MIXED_SUM_HOT_PAGE:
    div.u32 %r43, %r42, %r5;
    rem.u32 %r44, %r42, %r5;
    mul.wide.u32 %rd37, %r43, 4;
    add.s64 %rd38, %rd21, %rd37;
    ld.global.u32 %r45, [%rd38];
    mad.lo.u32 %r46, %r45, %r5, %r44;

SINGLE_MIXED_SUM_PAGE_READY:
    mov.f32 %f12, 0f00000000;
    mov.u32 %r47, 0;

SINGLE_MIXED_SUM_DOT:
    setp.ge.u32 %p8, %r47, %r3;
    @%p8 bra SINGLE_MIXED_SUM_DOT_DONE;
    add.u32 %r48, %r17, %r47;
    mul.wide.u32 %rd39, %r48, 4;
    add.s64 %rd40, %rd12, %rd39;
    ld.global.f32 %f13, [%rd40];
    mad.lo.u32 %r49, %r20, %r3, %r47;
    @%p7 bra SINGLE_MIXED_SUM_HOT_KEY;

    shr.u32 %r50, %r49, 6;
    mad.lo.u32 %r51, %r46, %r10, %r50;
    mul.wide.u32 %rd41, %r51, 4;
    add.s64 %rd42, %rd17, %rd41;
    ld.global.f32 %f14, [%rd42];
    shr.u32 %r52, %r49, 1;
    mad.lo.u32 %r53, %r46, %r9, %r52;
    cvt.u64.u32 %rd43, %r53;
    add.s64 %rd44, %rd15, %rd43;
    ld.global.u8 %r54, [%rd44];
    and.b32 %r55, %r49, 1;
    setp.eq.u32 %p9, %r55, 0;
    @%p9 bra SINGLE_MIXED_SUM_COLD_KEY_LOW;
    shr.u32 %r56, %r54, 4;
    bra SINGLE_MIXED_SUM_COLD_KEY_READY;

SINGLE_MIXED_SUM_COLD_KEY_LOW:
    and.b32 %r56, %r54, 15;

SINGLE_MIXED_SUM_COLD_KEY_READY:
    cvt.s32.u32 %r57, %r56;
    add.s32 %r57, %r57, -8;
    cvt.rn.f32.s32 %f15, %r57;
    mul.f32 %f16, %f15, %f14;
    bra SINGLE_MIXED_SUM_KEY_READY;

SINGLE_MIXED_SUM_HOT_KEY:
    mad.lo.u32 %r58, %r46, %r8, %r49;
    mul.wide.u32 %rd45, %r58, 4;
    add.s64 %rd46, %rd13, %rd45;
    ld.global.f32 %f16, [%rd46];

SINGLE_MIXED_SUM_KEY_READY:
    fma.rn.f32 %f12, %f13, %f16, %f12;
    add.u32 %r47, %r47, 1;
    bra SINGLE_MIXED_SUM_DOT;

SINGLE_MIXED_SUM_DOT_DONE:
    mul.f32 %f17, %f12, %f1;
    sub.f32 %f18, %f17, %f2;
    mul.f32 %f19, %f18, %f11;
    ex2.approx.f32 %f20, %f19;
    add.f32 %f9, %f9, %f20;
    mad.lo.u32 %r59, %r20, %r3, %r18;
    @%p7 bra SINGLE_MIXED_SUM_HOT_VALUE;

    mul.wide.u32 %rd47, %r46, 4;
    add.s64 %rd48, %rd18, %rd47;
    ld.global.f32 %f21, [%rd48];
    mad.lo.u32 %r60, %r46, %r8, %r59;
    cvt.u64.u32 %rd49, %r60;
    add.s64 %rd50, %rd16, %rd49;
    ld.global.s8 %r61, [%rd50];
    cvt.rn.f32.s32 %f22, %r61;
    mul.f32 %f23, %f22, %f21;
    bra SINGLE_MIXED_SUM_VALUE_READY;

SINGLE_MIXED_SUM_HOT_VALUE:
    mad.lo.u32 %r62, %r46, %r8, %r59;
    mul.wide.u32 %rd51, %r62, 4;
    add.s64 %rd52, %rd14, %rd51;
    ld.global.f32 %f23, [%rd52];

SINGLE_MIXED_SUM_VALUE_READY:
    fma.rn.f32 %f10, %f20, %f23, %f10;
    add.u32 %r40, %r40, 1;
    bra SINGLE_MIXED_SUM_POS;

SINGLE_MIXED_SUM_DONE:
    div.rn.f32 %f24, %f10, %f9;
    mul.wide.u32 %rd53, %r15, 4;
    add.s64 %rd54, %rd19, %rd53;
    st.global.f32 [%rd54], %f24;

SINGLE_MIXED_ATTENTION_DONE:
    ret;
}

"#;
    // ponytail: one block per row; enough until profiling proves we need warp intrinsics.
    const Q8_0_MATVEC_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry q8_0_matvec_kernel(
    .param .u64 q8_0_matvec_kernel_param_0,
    .param .u64 q8_0_matvec_kernel_param_1,
    .param .u64 q8_0_matvec_kernel_param_2,
    .param .u64 q8_0_matvec_kernel_param_3,
    .param .u32 q8_0_matvec_kernel_param_4,
    .param .u32 q8_0_matvec_kernel_param_5
)
{
    .shared .align 4 .b8 q8_reduce[1024];
    .reg .pred %p<5>;
    .reg .f32 %f<8>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;

    ld.param.u64 %rd1, [q8_0_matvec_kernel_param_0];
    ld.param.u64 %rd2, [q8_0_matvec_kernel_param_1];
    ld.param.u64 %rd3, [q8_0_matvec_kernel_param_2];
    ld.param.u64 %rd4, [q8_0_matvec_kernel_param_3];
    ld.param.u32 %r1, [q8_0_matvec_kernel_param_4];
    ld.param.u32 %r2, [q8_0_matvec_kernel_param_5];

    cvta.to.global.u64 %rd5, %rd1;
    cvta.to.global.u64 %rd6, %rd2;
    cvta.to.global.u64 %rd7, %rd3;
    cvta.to.global.u64 %rd8, %rd4;
    mov.u64 %rd17, q8_reduce;

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra Q8_DONE;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;
    mul.wide.u32 %rd15, %r4, 4;
    add.s64 %rd16, %rd17, %rd15;

    shr.u32 %r6, %r2, 5;
    mul.lo.u32 %r7, %r3, %r6;
    mov.u32 %r8, %r4;
    mov.f32 %f1, 0f00000000;

Q8_LOOP:
    setp.ge.u32 %p3, %r8, %r2;
    @%p3 bra Q8_REDUCE;

    shr.u32 %r9, %r8, 5;
    and.b32 %r10, %r8, 31;
    add.u32 %r11, %r7, %r9;

    mul.wide.u32 %rd9, %r11, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f2, [%rd10];

    mul.lo.u32 %r12, %r11, 32;
    add.u32 %r13, %r12, %r10;
    cvt.u64.u32 %rd11, %r13;
    add.s64 %rd12, %rd6, %rd11;
    ld.global.s8 %r14, [%rd12];
    cvt.rn.f32.s32 %f3, %r14;

    mul.wide.u32 %rd13, %r8, 4;
    add.s64 %rd14, %rd7, %rd13;
    ld.global.f32 %f4, [%rd14];

    mul.f32 %f5, %f2, %f3;
    fma.rn.f32 %f1, %f5, %f4, %f1;

    add.u32 %r8, %r8, %r5;
    bra Q8_LOOP;

Q8_REDUCE:
    st.shared.f32 [%rd16], %f1;
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 128;
    @%p2 bra Q8_REDUCE_128_DONE;
    add.u32 %r15, %r4, 128;
    mul.wide.u32 %rd18, %r15, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.shared.f32 %f2, [%rd16];
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f4, %f2, %f3;
    st.shared.f32 [%rd16], %f4;
Q8_REDUCE_128_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 64;
    @%p2 bra Q8_REDUCE_64_DONE;
    add.u32 %r15, %r4, 64;
    mul.wide.u32 %rd18, %r15, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.shared.f32 %f2, [%rd16];
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f4, %f2, %f3;
    st.shared.f32 [%rd16], %f4;
Q8_REDUCE_64_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 32;
    @%p2 bra Q8_REDUCE_32_DONE;
    add.u32 %r15, %r4, 32;
    mul.wide.u32 %rd18, %r15, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.shared.f32 %f2, [%rd16];
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f4, %f2, %f3;
    st.shared.f32 [%rd16], %f4;
Q8_REDUCE_32_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 16;
    @%p2 bra Q8_REDUCE_16_DONE;
    add.u32 %r15, %r4, 16;
    mul.wide.u32 %rd18, %r15, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.shared.f32 %f2, [%rd16];
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f4, %f2, %f3;
    st.shared.f32 [%rd16], %f4;
Q8_REDUCE_16_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 8;
    @%p2 bra Q8_REDUCE_8_DONE;
    add.u32 %r15, %r4, 8;
    mul.wide.u32 %rd18, %r15, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.shared.f32 %f2, [%rd16];
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f4, %f2, %f3;
    st.shared.f32 [%rd16], %f4;
Q8_REDUCE_8_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 4;
    @%p2 bra Q8_REDUCE_4_DONE;
    add.u32 %r15, %r4, 4;
    mul.wide.u32 %rd18, %r15, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.shared.f32 %f2, [%rd16];
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f4, %f2, %f3;
    st.shared.f32 [%rd16], %f4;
Q8_REDUCE_4_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 2;
    @%p2 bra Q8_REDUCE_2_DONE;
    add.u32 %r15, %r4, 2;
    mul.wide.u32 %rd18, %r15, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.shared.f32 %f2, [%rd16];
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f4, %f2, %f3;
    st.shared.f32 [%rd16], %f4;
Q8_REDUCE_2_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 1;
    @%p2 bra Q8_STORE;
    add.u32 %r15, %r4, 1;
    mul.wide.u32 %rd18, %r15, 4;
    add.s64 %rd19, %rd17, %rd18;
    ld.shared.f32 %f2, [%rd16];
    ld.shared.f32 %f3, [%rd19];
    add.f32 %f4, %f2, %f3;
    st.shared.f32 [%rd16], %f4;

Q8_STORE:
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra Q8_DONE;
    ld.shared.f32 %f6, [%rd17];
    mul.wide.u32 %rd18, %r3, 4;
    add.s64 %rd19, %rd8, %rd18;
    st.global.f32 [%rd19], %f6;

Q8_DONE:
    ret;
}
"#;
    // ponytail: block-parallel packed Q4_K is correct but still too slow for runtime default.
    const Q4_K_MATVEC_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry q4_k_matvec_kernel(
    .param .u64 q4_k_matvec_kernel_param_0,
    .param .u64 q4_k_matvec_kernel_param_1,
    .param .u64 q4_k_matvec_kernel_param_2,
    .param .u64 q4_k_matvec_kernel_param_3,
    .param .u64 q4_k_matvec_kernel_param_4,
    .param .u64 q4_k_matvec_kernel_param_5,
    .param .u32 q4_k_matvec_kernel_param_6,
    .param .u32 q4_k_matvec_kernel_param_7
)
{
    .shared .align 4 .b8 q4k_reduce[1024];
    .reg .pred %p<10>;
    .reg .f32 %f<16>;
    .reg .b32 %r<60>;
    .reg .b64 %rd<40>;

    ld.param.u64 %rd1, [q4_k_matvec_kernel_param_0];
    ld.param.u64 %rd2, [q4_k_matvec_kernel_param_1];
    ld.param.u64 %rd3, [q4_k_matvec_kernel_param_2];
    ld.param.u64 %rd4, [q4_k_matvec_kernel_param_3];
    ld.param.u64 %rd5, [q4_k_matvec_kernel_param_4];
    ld.param.u64 %rd6, [q4_k_matvec_kernel_param_5];
    ld.param.u32 %r1, [q4_k_matvec_kernel_param_6];
    ld.param.u32 %r2, [q4_k_matvec_kernel_param_7];

    cvta.to.global.u64 %rd7, %rd1;
    cvta.to.global.u64 %rd8, %rd2;
    cvta.to.global.u64 %rd9, %rd3;
    cvta.to.global.u64 %rd10, %rd4;
    cvta.to.global.u64 %rd11, %rd5;
    cvta.to.global.u64 %rd12, %rd6;
    mov.u64 %rd28, q4k_reduce;

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra Q4K_DONE;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;

    mul.wide.u32 %rd13, %r3, 4;
    add.s64 %rd14, %rd12, %rd13;
    mul.wide.u32 %rd29, %r4, 4;
    add.s64 %rd30, %rd28, %rd29;

    shr.u32 %r6, %r2, 8;
    mul.lo.u32 %r7, %r3, %r6;
    mov.u32 %r8, %r4;
    mov.f32 %f1, 0f00000000;

Q4K_LOOP:
    setp.ge.u32 %p3, %r8, %r2;
    @%p3 bra Q4K_STORE;

    shr.u32 %r9, %r8, 8;
    and.b32 %r10, %r8, 255;
    shr.u32 %r11, %r10, 6;
    and.b32 %r12, %r10, 63;
    setp.lt.u32 %p4, %r12, 32;
    and.b32 %r13, %r12, 31;
    shl.b32 %r14, %r11, 1;
    @%p4 bra Q4K_SCALE_LOW_INDEX;
    add.u32 %r14, %r14, 1;

Q4K_SCALE_LOW_INDEX:
    add.u32 %r15, %r7, %r9;

    mul.wide.u32 %rd15, %r15, 4;
    add.s64 %rd16, %rd7, %rd15;
    ld.global.f32 %f2, [%rd16];
    add.s64 %rd17, %rd8, %rd15;
    ld.global.f32 %f3, [%rd17];

    mul.lo.u32 %r16, %r15, 12;
    setp.lt.u32 %p5, %r14, 4;
    @%p5 bra Q4K_SCALE_DIRECT;

    add.u32 %r17, %r16, %r14;
    add.u32 %r18, %r17, 4;
    cvt.u64.u32 %rd18, %r18;
    add.s64 %rd19, %rd9, %rd18;
    ld.global.u8 %r19, [%rd19];
    and.b32 %r20, %r19, 15;

    sub.u32 %r21, %r17, 4;
    cvt.u64.u32 %rd20, %r21;
    add.s64 %rd21, %rd9, %rd20;
    ld.global.u8 %r22, [%rd21];
    shr.u32 %r23, %r22, 6;
    shl.b32 %r24, %r23, 4;
    or.b32 %r25, %r20, %r24;

    shr.u32 %r26, %r19, 4;
    cvt.u64.u32 %rd22, %r17;
    add.s64 %rd23, %rd9, %rd22;
    ld.global.u8 %r27, [%rd23];
    shr.u32 %r28, %r27, 6;
    shl.b32 %r29, %r28, 4;
    or.b32 %r30, %r26, %r29;
    bra Q4K_SCALE_READY;

Q4K_SCALE_DIRECT:
    add.u32 %r17, %r16, %r14;
    cvt.u64.u32 %rd18, %r17;
    add.s64 %rd19, %rd9, %rd18;
    ld.global.u8 %r25, [%rd19];
    and.b32 %r25, %r25, 63;
    add.u32 %r18, %r17, 4;
    cvt.u64.u32 %rd20, %r18;
    add.s64 %rd21, %rd9, %rd20;
    ld.global.u8 %r30, [%rd21];
    and.b32 %r30, %r30, 63;

Q4K_SCALE_READY:
    mul.lo.u32 %r31, %r15, 128;
    mul.lo.u32 %r32, %r11, 32;
    add.u32 %r33, %r31, %r32;
    add.u32 %r34, %r33, %r13;
    cvt.u64.u32 %rd24, %r34;
    add.s64 %rd25, %rd10, %rd24;
    ld.global.u8 %r35, [%rd25];
    @%p4 bra Q4K_QUANT_LOW;
    shr.u32 %r36, %r35, 4;
    bra Q4K_QUANT_READY;

Q4K_QUANT_LOW:
    and.b32 %r36, %r35, 15;

Q4K_QUANT_READY:
    cvt.rn.f32.u32 %f4, %r25;
    cvt.rn.f32.u32 %f5, %r30;
    cvt.rn.f32.u32 %f6, %r36;
    mul.f32 %f7, %f2, %f4;
    mul.f32 %f8, %f7, %f6;
    mul.f32 %f9, %f3, %f5;
    sub.f32 %f10, %f8, %f9;

    mul.wide.u32 %rd26, %r8, 4;
    add.s64 %rd27, %rd11, %rd26;
    ld.global.f32 %f11, [%rd27];
    fma.rn.f32 %f1, %f10, %f11, %f1;

    add.u32 %r8, %r8, %r5;
    bra Q4K_LOOP;

Q4K_STORE:
    st.shared.f32 [%rd30], %f1;
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 128;
    @%p2 bra Q4K_REDUCE_128_DONE;
    add.u32 %r37, %r4, 128;
    mul.wide.u32 %rd31, %r37, 4;
    add.s64 %rd32, %rd28, %rd31;
    ld.shared.f32 %f12, [%rd30];
    ld.shared.f32 %f13, [%rd32];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd30], %f14;
Q4K_REDUCE_128_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 64;
    @%p2 bra Q4K_REDUCE_64_DONE;
    add.u32 %r37, %r4, 64;
    mul.wide.u32 %rd31, %r37, 4;
    add.s64 %rd32, %rd28, %rd31;
    ld.shared.f32 %f12, [%rd30];
    ld.shared.f32 %f13, [%rd32];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd30], %f14;
Q4K_REDUCE_64_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 32;
    @%p2 bra Q4K_REDUCE_32_DONE;
    add.u32 %r37, %r4, 32;
    mul.wide.u32 %rd31, %r37, 4;
    add.s64 %rd32, %rd28, %rd31;
    ld.shared.f32 %f12, [%rd30];
    ld.shared.f32 %f13, [%rd32];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd30], %f14;
Q4K_REDUCE_32_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 16;
    @%p2 bra Q4K_REDUCE_16_DONE;
    add.u32 %r37, %r4, 16;
    mul.wide.u32 %rd31, %r37, 4;
    add.s64 %rd32, %rd28, %rd31;
    ld.shared.f32 %f12, [%rd30];
    ld.shared.f32 %f13, [%rd32];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd30], %f14;
Q4K_REDUCE_16_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 8;
    @%p2 bra Q4K_REDUCE_8_DONE;
    add.u32 %r37, %r4, 8;
    mul.wide.u32 %rd31, %r37, 4;
    add.s64 %rd32, %rd28, %rd31;
    ld.shared.f32 %f12, [%rd30];
    ld.shared.f32 %f13, [%rd32];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd30], %f14;
Q4K_REDUCE_8_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 4;
    @%p2 bra Q4K_REDUCE_4_DONE;
    add.u32 %r37, %r4, 4;
    mul.wide.u32 %rd31, %r37, 4;
    add.s64 %rd32, %rd28, %rd31;
    ld.shared.f32 %f12, [%rd30];
    ld.shared.f32 %f13, [%rd32];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd30], %f14;
Q4K_REDUCE_4_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 2;
    @%p2 bra Q4K_REDUCE_2_DONE;
    add.u32 %r37, %r4, 2;
    mul.wide.u32 %rd31, %r37, 4;
    add.s64 %rd32, %rd28, %rd31;
    ld.shared.f32 %f12, [%rd30];
    ld.shared.f32 %f13, [%rd32];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd30], %f14;
Q4K_REDUCE_2_DONE:
    bar.sync 0;

    setp.ge.u32 %p2, %r4, 1;
    @%p2 bra Q4K_WRITE;
    add.u32 %r37, %r4, 1;
    mul.wide.u32 %rd31, %r37, 4;
    add.s64 %rd32, %rd28, %rd31;
    ld.shared.f32 %f12, [%rd30];
    ld.shared.f32 %f13, [%rd32];
    add.f32 %f14, %f12, %f13;
    st.shared.f32 [%rd30], %f14;

Q4K_WRITE:
    setp.ne.u32 %p2, %r4, 0;
    @%p2 bra Q4K_DONE;
    ld.shared.f32 %f12, [%rd28];
    st.global.f32 [%rd14], %f12;

Q4K_DONE:
    ret;
}
"#;
    const Q6_K_MATVEC_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry q6_k_matvec_kernel(
    .param .u64 q6_k_matvec_kernel_param_0,
    .param .u64 q6_k_matvec_kernel_param_1,
    .param .u64 q6_k_matvec_kernel_param_2,
    .param .u64 q6_k_matvec_kernel_param_3,
    .param .u32 q6_k_matvec_kernel_param_4,
    .param .u32 q6_k_matvec_kernel_param_5
)
{
    .shared .align 4 .b8 q6k_reduce[1024];
    .reg .pred %p<8>;
    .reg .f32 %f<12>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<32>;

    ld.param.u64 %rd1, [q6_k_matvec_kernel_param_0];
    ld.param.u64 %rd2, [q6_k_matvec_kernel_param_1];
    ld.param.u64 %rd3, [q6_k_matvec_kernel_param_2];
    ld.param.u64 %rd4, [q6_k_matvec_kernel_param_3];
    ld.param.u32 %r1, [q6_k_matvec_kernel_param_4];
    ld.param.u32 %r2, [q6_k_matvec_kernel_param_5];

    cvta.to.global.u64 %rd5, %rd1;
    cvta.to.global.u64 %rd6, %rd2;
    cvta.to.global.u64 %rd7, %rd3;
    cvta.to.global.u64 %rd8, %rd4;
    mov.u64 %rd20, q6k_reduce;

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra Q6K_DONE;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;
    shr.u32 %r6, %r2, 8;
    mul.lo.u32 %r7, %r3, %r6;
    mov.u32 %r8, %r4;
    mov.f32 %f1, 0f00000000;

Q6K_LOOP:
    setp.ge.u32 %p2, %r8, %r2;
    @%p2 bra Q6K_STORE;

    shr.u32 %r9, %r8, 8;
    and.b32 %r10, %r8, 255;
    shr.u32 %r11, %r10, 7;
    and.b32 %r12, %r10, 127;
    shr.u32 %r13, %r12, 5;
    and.b32 %r14, %r12, 31;
    add.u32 %r15, %r7, %r9;

    mul.wide.u32 %rd9, %r15, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f2, [%rd10];

    mul.lo.u32 %r16, %r15, 210;
    shl.b32 %r17, %r11, 6;
    and.b32 %r18, %r13, 1;
    shl.b32 %r19, %r18, 5;
    add.u32 %r20, %r16, %r17;
    add.u32 %r20, %r20, %r19;
    add.u32 %r20, %r20, %r14;
    cvt.u64.u32 %rd13, %r20;
    add.s64 %rd14, %rd6, %rd13;
    ld.global.u8 %r21, [%rd14];
    setp.ge.u32 %p3, %r13, 2;
    @%p3 bra Q6K_QUANT_HIGH;
    and.b32 %r22, %r21, 15;
    bra Q6K_QUANT_LOW_READY;

Q6K_QUANT_HIGH:
    shr.u32 %r22, %r21, 4;

Q6K_QUANT_LOW_READY:
    shl.b32 %r23, %r11, 5;
    add.u32 %r24, %r16, 128;
    add.u32 %r24, %r24, %r23;
    add.u32 %r24, %r24, %r14;
    cvt.u64.u32 %rd15, %r24;
    add.s64 %rd16, %rd6, %rd15;
    ld.global.u8 %r25, [%rd16];
    shl.b32 %r26, %r13, 1;
    shr.u32 %r27, %r25, %r26;
    and.b32 %r27, %r27, 3;
    shl.b32 %r28, %r27, 4;
    or.b32 %r29, %r22, %r28;
    sub.s32 %r30, %r29, 32;

    shl.b32 %r31, %r11, 3;
    shr.u32 %r32, %r14, 4;
    shl.b32 %r33, %r13, 1;
    add.u32 %r34, %r16, 192;
    add.u32 %r34, %r34, %r31;
    add.u32 %r34, %r34, %r32;
    add.u32 %r34, %r34, %r33;
    cvt.u64.u32 %rd17, %r34;
    add.s64 %rd18, %rd6, %rd17;
    ld.global.s8 %r35, [%rd18];

    cvt.rn.f32.s32 %f3, %r35;
    cvt.rn.f32.s32 %f4, %r30;
    mul.f32 %f5, %f2, %f3;
    mul.f32 %f6, %f5, %f4;
    mul.wide.u32 %rd19, %r8, 4;
    add.s64 %rd23, %rd7, %rd19;
    ld.global.f32 %f7, [%rd23];
    fma.rn.f32 %f1, %f6, %f7, %f1;

    add.u32 %r8, %r8, %r5;
    bra Q6K_LOOP;

Q6K_STORE:
    mul.wide.u32 %rd21, %r4, 4;
    add.s64 %rd22, %rd20, %rd21;
    st.shared.f32 [%rd22], %f1;
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 128;
    @%p4 bra Q6K_REDUCE_128_DONE;
    add.u32 %r36, %r4, 128;
    mul.wide.u32 %rd24, %r36, 4;
    add.s64 %rd25, %rd20, %rd24;
    ld.shared.f32 %f8, [%rd22];
    ld.shared.f32 %f9, [%rd25];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd22], %f10;
Q6K_REDUCE_128_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 64;
    @%p4 bra Q6K_REDUCE_64_DONE;
    add.u32 %r36, %r4, 64;
    mul.wide.u32 %rd24, %r36, 4;
    add.s64 %rd25, %rd20, %rd24;
    ld.shared.f32 %f8, [%rd22];
    ld.shared.f32 %f9, [%rd25];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd22], %f10;
Q6K_REDUCE_64_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 32;
    @%p4 bra Q6K_REDUCE_32_DONE;
    add.u32 %r36, %r4, 32;
    mul.wide.u32 %rd24, %r36, 4;
    add.s64 %rd25, %rd20, %rd24;
    ld.shared.f32 %f8, [%rd22];
    ld.shared.f32 %f9, [%rd25];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd22], %f10;
Q6K_REDUCE_32_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 16;
    @%p4 bra Q6K_REDUCE_16_DONE;
    add.u32 %r36, %r4, 16;
    mul.wide.u32 %rd24, %r36, 4;
    add.s64 %rd25, %rd20, %rd24;
    ld.shared.f32 %f8, [%rd22];
    ld.shared.f32 %f9, [%rd25];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd22], %f10;
Q6K_REDUCE_16_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 8;
    @%p4 bra Q6K_REDUCE_8_DONE;
    add.u32 %r36, %r4, 8;
    mul.wide.u32 %rd24, %r36, 4;
    add.s64 %rd25, %rd20, %rd24;
    ld.shared.f32 %f8, [%rd22];
    ld.shared.f32 %f9, [%rd25];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd22], %f10;
Q6K_REDUCE_8_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 4;
    @%p4 bra Q6K_REDUCE_4_DONE;
    add.u32 %r36, %r4, 4;
    mul.wide.u32 %rd24, %r36, 4;
    add.s64 %rd25, %rd20, %rd24;
    ld.shared.f32 %f8, [%rd22];
    ld.shared.f32 %f9, [%rd25];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd22], %f10;
Q6K_REDUCE_4_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 2;
    @%p4 bra Q6K_REDUCE_2_DONE;
    add.u32 %r36, %r4, 2;
    mul.wide.u32 %rd24, %r36, 4;
    add.s64 %rd25, %rd20, %rd24;
    ld.shared.f32 %f8, [%rd22];
    ld.shared.f32 %f9, [%rd25];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd22], %f10;
Q6K_REDUCE_2_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 1;
    @%p4 bra Q6K_WRITE;
    add.u32 %r36, %r4, 1;
    mul.wide.u32 %rd24, %r36, 4;
    add.s64 %rd25, %rd20, %rd24;
    ld.shared.f32 %f8, [%rd22];
    ld.shared.f32 %f9, [%rd25];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd22], %f10;

Q6K_WRITE:
    setp.ne.u32 %p4, %r4, 0;
    @%p4 bra Q6K_DONE;
    ld.shared.f32 %f8, [%rd20];
    mul.wide.u32 %rd26, %r3, 4;
    add.s64 %rd27, %rd8, %rd26;
    st.global.f32 [%rd27], %f8;

Q6K_DONE:
    ret;
}
"#;
    const EMBEDDING_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry embedding_kernel(
    .param .u64 embedding_kernel_param_0,
    .param .u64 embedding_kernel_param_1,
    .param .u64 embedding_kernel_param_2,
    .param .u32 embedding_kernel_param_3,
    .param .u32 embedding_kernel_param_4,
    .param .u32 embedding_kernel_param_5
)
{
    .reg .pred %p<4>;
    .reg .f32 %f<4>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [embedding_kernel_param_0];
    ld.param.u64 %rd2, [embedding_kernel_param_1];
    ld.param.u64 %rd3, [embedding_kernel_param_2];
    ld.param.u32 %r1, [embedding_kernel_param_3];
    ld.param.u32 %r2, [embedding_kernel_param_4];
    ld.param.u32 %r3, [embedding_kernel_param_5];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;

    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r5, %r4, %r6;

    mul.lo.u32 %r8, %r1, %r2;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra EMBED_DONE;

    div.u32 %r9, %r7, %r2;
    mul.lo.u32 %r10, %r9, %r2;
    sub.u32 %r11, %r7, %r10;

    mul.wide.u32 %rd7, %r9, 4;
    add.s64 %rd8, %rd5, %rd7;
    ld.global.u32 %r12, [%rd8];
    setp.ge.u32 %p2, %r12, %r3;
    @%p2 bra EMBED_ZERO;

    mul.lo.u32 %r13, %r12, %r2;
    add.u32 %r14, %r13, %r11;
    mul.wide.u32 %rd9, %r14, 4;
    add.s64 %rd10, %rd4, %rd9;
    ld.global.f32 %f1, [%rd10];
    bra EMBED_STORE;

EMBED_ZERO:
    mov.f32 %f1, 0f00000000;

EMBED_STORE:
    mul.wide.u32 %rd11, %r7, 4;
    add.s64 %rd12, %rd6, %rd11;
    st.global.f32 [%rd12], %f1;

EMBED_DONE:
    ret;
}

.visible .entry q8_0_embedding_kernel(
    .param .u64 q8_0_embedding_kernel_param_0,
    .param .u64 q8_0_embedding_kernel_param_1,
    .param .u64 q8_0_embedding_kernel_param_2,
    .param .u64 q8_0_embedding_kernel_param_3,
    .param .u32 q8_0_embedding_kernel_param_4,
    .param .u32 q8_0_embedding_kernel_param_5,
    .param .u32 q8_0_embedding_kernel_param_6
)
{
    .reg .pred %p<4>;
    .reg .f32 %f<4>;
    .reg .b32 %r<28>;
    .reg .b64 %rd<18>;

    ld.param.u64 %rd1, [q8_0_embedding_kernel_param_0];
    ld.param.u64 %rd2, [q8_0_embedding_kernel_param_1];
    ld.param.u64 %rd3, [q8_0_embedding_kernel_param_2];
    ld.param.u64 %rd4, [q8_0_embedding_kernel_param_3];
    ld.param.u32 %r1, [q8_0_embedding_kernel_param_4];
    ld.param.u32 %r2, [q8_0_embedding_kernel_param_5];
    ld.param.u32 %r3, [q8_0_embedding_kernel_param_6];

    cvta.to.global.u64 %rd5, %rd1;
    cvta.to.global.u64 %rd6, %rd2;
    cvta.to.global.u64 %rd7, %rd3;
    cvta.to.global.u64 %rd8, %rd4;

    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r5, %r4, %r6;

    mul.lo.u32 %r8, %r1, %r2;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra Q8_EMBED_DONE;

    div.u32 %r9, %r7, %r2;
    mul.lo.u32 %r10, %r9, %r2;
    sub.u32 %r11, %r7, %r10;

    mul.wide.u32 %rd9, %r9, 4;
    add.s64 %rd10, %rd7, %rd9;
    ld.global.u32 %r12, [%rd10];
    setp.ge.u32 %p2, %r12, %r3;
    @%p2 bra Q8_EMBED_ZERO;

    shr.u32 %r13, %r2, 5;
    shr.u32 %r14, %r11, 5;
    and.b32 %r15, %r11, 31;
    mul.lo.u32 %r16, %r12, %r13;
    add.u32 %r17, %r16, %r14;

    mul.wide.u32 %rd11, %r17, 4;
    add.s64 %rd12, %rd5, %rd11;
    ld.global.f32 %f1, [%rd12];

    mul.lo.u32 %r18, %r17, 32;
    add.u32 %r19, %r18, %r15;
    mul.wide.u32 %rd13, %r19, 1;
    add.s64 %rd14, %rd6, %rd13;
    ld.global.s8 %r20, [%rd14];
    cvt.rn.f32.s32 %f2, %r20;
    mul.f32 %f3, %f1, %f2;
    bra Q8_EMBED_STORE;

Q8_EMBED_ZERO:
    mov.f32 %f3, 0f00000000;

Q8_EMBED_STORE:
    mul.wide.u32 %rd15, %r7, 4;
    add.s64 %rd16, %rd8, %rd15;
    st.global.f32 [%rd16], %f3;

Q8_EMBED_DONE:
    ret;
}

.visible .entry q4_k_embedding_kernel(
    .param .u64 q4_k_embedding_kernel_param_0,
    .param .u64 q4_k_embedding_kernel_param_1,
    .param .u64 q4_k_embedding_kernel_param_2,
    .param .u32 q4_k_embedding_kernel_param_3,
    .param .u32 q4_k_embedding_kernel_param_4,
    .param .u32 q4_k_embedding_kernel_param_5
)
{
    .reg .pred %p<4>;
    .reg .f32 %f<4>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [q4_k_embedding_kernel_param_0];
    ld.param.u64 %rd2, [q4_k_embedding_kernel_param_1];
    ld.param.u64 %rd3, [q4_k_embedding_kernel_param_2];
    ld.param.u32 %r1, [q4_k_embedding_kernel_param_3];
    ld.param.u32 %r2, [q4_k_embedding_kernel_param_4];
    ld.param.u32 %r3, [q4_k_embedding_kernel_param_5];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;

    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r5, %r4, %r6;

    mul.lo.u32 %r8, %r1, %r2;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra Q4K_EMBED_DONE;

    div.u32 %r9, %r7, %r2;
    mul.lo.u32 %r10, %r9, %r2;
    sub.u32 %r11, %r7, %r10;

    mul.wide.u32 %rd7, %r9, 4;
    add.s64 %rd8, %rd5, %rd7;
    ld.global.u32 %r12, [%rd8];
    setp.ge.u32 %p2, %r12, %r3;
    @%p2 bra Q4K_EMBED_ZERO;

    mul.lo.u32 %r13, %r12, %r2;
    add.u32 %r14, %r13, %r11;
    mul.wide.u32 %rd9, %r14, 4;
    add.s64 %rd10, %rd4, %rd9;
    ld.global.f32 %f1, [%rd10];
    bra Q4K_EMBED_STORE;

Q4K_EMBED_ZERO:
    mov.f32 %f1, 0f00000000;

Q4K_EMBED_STORE:
    mul.wide.u32 %rd11, %r7, 4;
    add.s64 %rd12, %rd6, %rd11;
    st.global.f32 [%rd12], %f1;

Q4K_EMBED_DONE:
    ret;
}

.visible .entry q4_k_packed_embedding_kernel(
    .param .u64 q4_k_packed_embedding_kernel_param_0,
    .param .u64 q4_k_packed_embedding_kernel_param_1,
    .param .u64 q4_k_packed_embedding_kernel_param_2,
    .param .u64 q4_k_packed_embedding_kernel_param_3,
    .param .u64 q4_k_packed_embedding_kernel_param_4,
    .param .u64 q4_k_packed_embedding_kernel_param_5,
    .param .u32 q4_k_packed_embedding_kernel_param_6,
    .param .u32 q4_k_packed_embedding_kernel_param_7,
    .param .u32 q4_k_packed_embedding_kernel_param_8
)
{
    .reg .pred %p<10>;
    .reg .f32 %f<12>;
    .reg .b32 %r<56>;
    .reg .b64 %rd<34>;

    ld.param.u64 %rd1, [q4_k_packed_embedding_kernel_param_0];
    ld.param.u64 %rd2, [q4_k_packed_embedding_kernel_param_1];
    ld.param.u64 %rd3, [q4_k_packed_embedding_kernel_param_2];
    ld.param.u64 %rd4, [q4_k_packed_embedding_kernel_param_3];
    ld.param.u64 %rd5, [q4_k_packed_embedding_kernel_param_4];
    ld.param.u64 %rd6, [q4_k_packed_embedding_kernel_param_5];
    ld.param.u32 %r1, [q4_k_packed_embedding_kernel_param_6];
    ld.param.u32 %r2, [q4_k_packed_embedding_kernel_param_7];
    ld.param.u32 %r3, [q4_k_packed_embedding_kernel_param_8];

    cvta.to.global.u64 %rd7, %rd1;
    cvta.to.global.u64 %rd8, %rd2;
    cvta.to.global.u64 %rd9, %rd3;
    cvta.to.global.u64 %rd10, %rd4;
    cvta.to.global.u64 %rd11, %rd5;
    cvta.to.global.u64 %rd12, %rd6;

    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r5, %r4, %r6;

    mul.lo.u32 %r8, %r1, %r2;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra Q4KP_EMBED_DONE;

    div.u32 %r9, %r7, %r2;
    mul.lo.u32 %r10, %r9, %r2;
    sub.u32 %r11, %r7, %r10;

    mul.wide.u32 %rd13, %r9, 4;
    add.s64 %rd14, %rd11, %rd13;
    ld.global.u32 %r12, [%rd14];
    setp.ge.u32 %p2, %r12, %r3;
    @%p2 bra Q4KP_EMBED_ZERO;

    shr.u32 %r13, %r2, 8;
    shr.u32 %r14, %r11, 8;
    and.b32 %r15, %r11, 255;
    shr.u32 %r16, %r15, 6;
    and.b32 %r17, %r15, 63;
    setp.lt.u32 %p3, %r17, 32;
    and.b32 %r18, %r17, 31;
    shl.b32 %r19, %r16, 1;
    @%p3 bra Q4KP_SCALE_LOW_INDEX;
    add.u32 %r19, %r19, 1;

Q4KP_SCALE_LOW_INDEX:
    mul.lo.u32 %r20, %r12, %r13;
    add.u32 %r21, %r20, %r14;

    mul.wide.u32 %rd15, %r21, 4;
    add.s64 %rd16, %rd7, %rd15;
    ld.global.f32 %f1, [%rd16];
    add.s64 %rd17, %rd8, %rd15;
    ld.global.f32 %f2, [%rd17];

    mul.lo.u32 %r22, %r21, 12;
    setp.lt.u32 %p4, %r19, 4;
    @%p4 bra Q4KP_SCALE_DIRECT;

    add.u32 %r23, %r22, %r19;
    add.u32 %r24, %r23, 4;
    cvt.u64.u32 %rd18, %r24;
    add.s64 %rd19, %rd9, %rd18;
    ld.global.u8 %r25, [%rd19];
    and.b32 %r26, %r25, 15;

    sub.u32 %r27, %r23, 4;
    cvt.u64.u32 %rd20, %r27;
    add.s64 %rd21, %rd9, %rd20;
    ld.global.u8 %r28, [%rd21];
    shr.u32 %r29, %r28, 6;
    shl.b32 %r30, %r29, 4;
    or.b32 %r31, %r26, %r30;

    shr.u32 %r32, %r25, 4;
    cvt.u64.u32 %rd22, %r23;
    add.s64 %rd23, %rd9, %rd22;
    ld.global.u8 %r33, [%rd23];
    shr.u32 %r34, %r33, 6;
    shl.b32 %r35, %r34, 4;
    or.b32 %r36, %r32, %r35;
    bra Q4KP_SCALE_READY;

Q4KP_SCALE_DIRECT:
    add.u32 %r23, %r22, %r19;
    cvt.u64.u32 %rd18, %r23;
    add.s64 %rd19, %rd9, %rd18;
    ld.global.u8 %r31, [%rd19];
    and.b32 %r31, %r31, 63;
    add.u32 %r24, %r23, 4;
    cvt.u64.u32 %rd20, %r24;
    add.s64 %rd21, %rd9, %rd20;
    ld.global.u8 %r36, [%rd21];
    and.b32 %r36, %r36, 63;

Q4KP_SCALE_READY:
    mul.lo.u32 %r37, %r21, 128;
    mul.lo.u32 %r38, %r16, 32;
    add.u32 %r39, %r37, %r38;
    add.u32 %r40, %r39, %r18;
    cvt.u64.u32 %rd24, %r40;
    add.s64 %rd25, %rd10, %rd24;
    ld.global.u8 %r41, [%rd25];
    @%p3 bra Q4KP_QUANT_LOW;
    shr.u32 %r42, %r41, 4;
    bra Q4KP_QUANT_READY;

Q4KP_QUANT_LOW:
    and.b32 %r42, %r41, 15;

Q4KP_QUANT_READY:
    cvt.rn.f32.u32 %f3, %r31;
    cvt.rn.f32.u32 %f4, %r36;
    cvt.rn.f32.u32 %f5, %r42;
    mul.f32 %f6, %f1, %f3;
    mul.f32 %f7, %f6, %f5;
    mul.f32 %f8, %f2, %f4;
    sub.f32 %f9, %f7, %f8;
    bra Q4KP_EMBED_STORE;

Q4KP_EMBED_ZERO:
    mov.f32 %f9, 0f00000000;

Q4KP_EMBED_STORE:
    mul.wide.u32 %rd26, %r7, 4;
    add.s64 %rd27, %rd12, %rd26;
    st.global.f32 [%rd27], %f9;

Q4KP_EMBED_DONE:
    ret;
}

.visible .entry q6_k_packed_embedding_kernel(
    .param .u64 q6_k_packed_embedding_kernel_param_0,
    .param .u64 q6_k_packed_embedding_kernel_param_1,
    .param .u64 q6_k_packed_embedding_kernel_param_2,
    .param .u64 q6_k_packed_embedding_kernel_param_3,
    .param .u32 q6_k_packed_embedding_kernel_param_4,
    .param .u32 q6_k_packed_embedding_kernel_param_5,
    .param .u32 q6_k_packed_embedding_kernel_param_6
)
{
    .reg .pred %p<6>;
    .reg .f32 %f<8>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<28>;

    ld.param.u64 %rd1, [q6_k_packed_embedding_kernel_param_0];
    ld.param.u64 %rd2, [q6_k_packed_embedding_kernel_param_1];
    ld.param.u64 %rd3, [q6_k_packed_embedding_kernel_param_2];
    ld.param.u64 %rd4, [q6_k_packed_embedding_kernel_param_3];
    ld.param.u32 %r1, [q6_k_packed_embedding_kernel_param_4];
    ld.param.u32 %r2, [q6_k_packed_embedding_kernel_param_5];
    ld.param.u32 %r3, [q6_k_packed_embedding_kernel_param_6];

    cvta.to.global.u64 %rd5, %rd1;
    cvta.to.global.u64 %rd6, %rd2;
    cvta.to.global.u64 %rd7, %rd3;
    cvta.to.global.u64 %rd8, %rd4;

    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r5, %r4, %r6;
    mul.lo.u32 %r8, %r1, %r2;
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra Q6KP_EMBED_DONE;

    div.u32 %r9, %r7, %r2;
    mul.lo.u32 %r10, %r9, %r2;
    sub.u32 %r11, %r7, %r10;
    mul.wide.u32 %rd9, %r9, 4;
    add.s64 %rd10, %rd7, %rd9;
    ld.global.u32 %r12, [%rd10];
    setp.ge.u32 %p2, %r12, %r3;
    @%p2 bra Q6KP_EMBED_ZERO;

    shr.u32 %r13, %r2, 8;
    shr.u32 %r14, %r11, 8;
    and.b32 %r15, %r11, 255;
    shr.u32 %r16, %r15, 7;
    and.b32 %r17, %r15, 127;
    shr.u32 %r18, %r17, 5;
    and.b32 %r19, %r17, 31;
    mul.lo.u32 %r20, %r12, %r13;
    add.u32 %r21, %r20, %r14;

    mul.wide.u32 %rd11, %r21, 4;
    add.s64 %rd12, %rd5, %rd11;
    ld.global.f32 %f1, [%rd12];

    mul.lo.u32 %r22, %r21, 210;
    shl.b32 %r23, %r16, 6;
    and.b32 %r24, %r18, 1;
    shl.b32 %r25, %r24, 5;
    add.u32 %r26, %r22, %r23;
    add.u32 %r26, %r26, %r25;
    add.u32 %r26, %r26, %r19;
    cvt.u64.u32 %rd13, %r26;
    add.s64 %rd14, %rd6, %rd13;
    ld.global.u8 %r27, [%rd14];
    setp.ge.u32 %p3, %r18, 2;
    @%p3 bra Q6KP_QUANT_HIGH;
    and.b32 %r28, %r27, 15;
    bra Q6KP_QUANT_LOW_READY;

Q6KP_QUANT_HIGH:
    shr.u32 %r28, %r27, 4;

Q6KP_QUANT_LOW_READY:
    shl.b32 %r29, %r16, 5;
    add.u32 %r30, %r22, 128;
    add.u32 %r30, %r30, %r29;
    add.u32 %r30, %r30, %r19;
    cvt.u64.u32 %rd15, %r30;
    add.s64 %rd16, %rd6, %rd15;
    ld.global.u8 %r31, [%rd16];
    shl.b32 %r32, %r18, 1;
    shr.u32 %r33, %r31, %r32;
    and.b32 %r33, %r33, 3;
    shl.b32 %r34, %r33, 4;
    or.b32 %r35, %r28, %r34;
    sub.s32 %r36, %r35, 32;

    shl.b32 %r37, %r16, 3;
    shr.u32 %r38, %r19, 4;
    shl.b32 %r39, %r18, 1;
    add.u32 %r40, %r22, 192;
    add.u32 %r40, %r40, %r37;
    add.u32 %r40, %r40, %r38;
    add.u32 %r40, %r40, %r39;
    cvt.u64.u32 %rd17, %r40;
    add.s64 %rd18, %rd6, %rd17;
    ld.global.s8 %r41, [%rd18];

    cvt.rn.f32.s32 %f2, %r41;
    cvt.rn.f32.s32 %f3, %r36;
    mul.f32 %f4, %f1, %f2;
    mul.f32 %f5, %f4, %f3;
    bra Q6KP_EMBED_STORE;

Q6KP_EMBED_ZERO:
    mov.f32 %f5, 0f00000000;

Q6KP_EMBED_STORE:
    mul.wide.u32 %rd19, %r7, 4;
    add.s64 %rd20, %rd8, %rd19;
    st.global.f32 [%rd20], %f5;

Q6KP_EMBED_DONE:
    ret;
}

"#;

    fn cuda_error(context: &str, err: impl Display) -> XrtError {
        XrtError::Cuda(format!("{context}: {err}"))
    }

    fn to_u32(value: usize, what: &str) -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| XrtError::Shape(format!("{what} {value} exceeds CUDA u32 limits")))
    }

    fn encode_adaptive_kv_route(is_hot: bool, local_position: usize) -> Result<u32> {
        const HOT_BIT: u32 = 1 << 31;
        let position = to_u32(local_position, "CUDA adaptive KV local position")?;
        if position >= HOT_BIT {
            return Err(XrtError::Shape(format!(
                "CUDA adaptive KV local position {local_position} exceeds 31-bit route limits"
            )));
        }
        Ok(if is_hot { position | HOT_BIT } else { position })
    }

    fn expect_len(actual: usize, expected: usize, what: &str) -> Result<()> {
        if actual == expected {
            Ok(())
        } else {
            Err(XrtError::Shape(format!(
                "{what} length mismatch: expected {expected}, found {actual}"
            )))
        }
    }

    pub(super) fn decode_float_tensor_bytes(
        bytes: &[u8],
        tensor_name: &str,
        dtype: DType,
        element_count: usize,
    ) -> Result<Vec<f32>> {
        let element_bytes = match dtype {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F16 | DType::BF16 => std::mem::size_of::<u16>(),
            dtype => {
                return Err(XrtError::Unsupported(format!(
                    "resident float tensor upload requires F32, F16, or BF16 dtype, tensor `{tensor_name}` is {dtype:?}"
                )));
            }
        };
        let expected_bytes = checked_mul(element_count, element_bytes, "float tensor byte length")?;
        expect_len(bytes.len(), expected_bytes, tensor_name)?;
        match dtype {
            DType::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            DType::F16 => bytes.chunks_exact(2).map(decode_f16).collect(),
            DType::BF16 => bytes.chunks_exact(2).map(decode_bf16).collect(),
            _ => unreachable!("dtype was checked above"),
        }
    }

    fn decode_float_tensor_values(
        gguf: &GgufFile,
        tensor_name: &str,
        dtype: DType,
        element_count: usize,
    ) -> Result<Vec<f32>> {
        decode_float_tensor_bytes(
            gguf.tensor_data(tensor_name)?,
            tensor_name,
            dtype,
            element_count,
        )
    }

    fn split_q8_0_matrix(matrix: &[u8], rows: usize, cols: usize) -> Result<(Vec<f32>, Vec<u8>)> {
        if cols % DType::Q8_0.block_size() != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "Q8_0 matrix column count {cols} is not divisible by {}",
                DType::Q8_0.block_size()
            )));
        }

        let blocks_per_row = cols / DType::Q8_0.block_size();
        let total_blocks = checked_mul(rows, blocks_per_row, "Q8_0 matrix block count")?;
        let expected_bytes = checked_mul(
            total_blocks,
            DType::Q8_0.block_bytes(),
            "Q8_0 matrix byte length",
        )?;
        expect_len(matrix.len(), expected_bytes, "Q8_0 matrix")?;

        let mut scales = Vec::with_capacity(total_blocks);
        let mut quants = Vec::with_capacity(checked_mul(
            total_blocks,
            DType::Q8_0.block_size(),
            "Q8_0 quant byte length",
        )?);
        for block in matrix.chunks_exact(DType::Q8_0.block_bytes()) {
            scales.push(decode_f16(&block[..2])?);
            quants.extend_from_slice(&block[2..]);
        }
        Ok((scales, quants))
    }

    pub(super) fn q8_layer_kv_allocated_bytes(capacity: usize, width: usize) -> Result<u64> {
        let elements = checked_mul(capacity, width, "CUDA Q8 KV cache elements")?;
        let scale_bytes = checked_mul(
            capacity,
            std::mem::size_of::<f32>(),
            "CUDA Q8 KV cache scale bytes",
        )?;
        elements
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(scale_bytes.checked_mul(2)?))
            .map(|bytes| bytes as u64)
            .ok_or_else(|| XrtError::Runtime("CUDA Q8 KV cache byte count overflow".to_string()))
    }

    fn kq4_key_row_bytes(width: usize) -> usize {
        width.div_ceil(2)
    }

    fn kq4_key_groups(width: usize) -> usize {
        width.div_ceil(64)
    }

    pub(super) fn kq4_vq8_layer_kv_allocated_bytes(capacity: usize, width: usize) -> Result<u64> {
        let key_bytes = checked_mul(capacity, kq4_key_row_bytes(width), "CUDA KQ4/VQ8 key bytes")?;
        let value_bytes = checked_mul(capacity, width, "CUDA KQ4/VQ8 value bytes")?;
        let key_scales = checked_mul(
            capacity,
            kq4_key_groups(width),
            "CUDA KQ4/VQ8 key scale count",
        )?;
        let key_scale_bytes = checked_mul(
            key_scales,
            std::mem::size_of::<f32>(),
            "CUDA KQ4/VQ8 key scale bytes",
        )?;
        let value_scale_bytes = checked_mul(
            capacity,
            std::mem::size_of::<f32>(),
            "CUDA KQ4/VQ8 value scale bytes",
        )?;
        key_bytes
            .checked_add(value_bytes)
            .and_then(|bytes| bytes.checked_add(key_scale_bytes))
            .and_then(|bytes| bytes.checked_add(value_scale_bytes))
            .map(|bytes| bytes as u64)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA KQ4/VQ8 KV cache byte count overflow".to_string())
            })
    }

    fn split_q4_0_matrix(matrix: &[u8], rows: usize, cols: usize) -> Result<(Vec<f32>, Vec<u8>)> {
        if cols % DType::Q4_0.block_size() != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "Q4_0 matrix column count {cols} is not divisible by {}",
                DType::Q4_0.block_size()
            )));
        }

        let blocks_per_row = cols / DType::Q4_0.block_size();
        let total_blocks = checked_mul(rows, blocks_per_row, "Q4_0 matrix block count")?;
        let expected_bytes = checked_mul(
            total_blocks,
            DType::Q4_0.block_bytes(),
            "Q4_0 matrix byte length",
        )?;
        expect_len(matrix.len(), expected_bytes, "Q4_0 matrix")?;

        let mut scales = Vec::with_capacity(total_blocks);
        let mut quants = Vec::with_capacity(checked_mul(
            total_blocks,
            DType::Q4_0.block_size(),
            "Q4_0 expanded quant byte length",
        )?);
        for block in matrix.chunks_exact(DType::Q4_0.block_bytes()) {
            scales.push(decode_f16(&block[..2])?);
            let base = quants.len();
            quants.resize(base + DType::Q4_0.block_size(), 0);
            for (idx, packed) in block[2..].iter().copied().enumerate() {
                quants[base + idx] = ((packed & 0x0f) as i8 - 8) as u8;
                quants[base + 16 + idx] = (((packed >> 4) & 0x0f) as i8 - 8) as u8;
            }
        }
        Ok((scales, quants))
    }

    fn split_q4_k_matrix(
        matrix: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<u8>, Vec<u8>)> {
        if cols % DType::Q4_K.block_size() != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "Q4_K matrix column count {cols} is not divisible by {}",
                DType::Q4_K.block_size()
            )));
        }

        let blocks_per_row = cols / DType::Q4_K.block_size();
        let total_blocks = checked_mul(rows, blocks_per_row, "Q4_K matrix block count")?;
        let expected_bytes = checked_mul(
            total_blocks,
            DType::Q4_K.block_bytes(),
            "Q4_K matrix byte length",
        )?;
        expect_len(matrix.len(), expected_bytes, "Q4_K matrix")?;

        let mut d = Vec::with_capacity(total_blocks);
        let mut dmin = Vec::with_capacity(total_blocks);
        let mut scales = Vec::with_capacity(checked_mul(total_blocks, 12, "Q4_K scale bytes")?);
        let mut quants = Vec::with_capacity(checked_mul(total_blocks, 128, "Q4_K quant bytes")?);
        for block in matrix.chunks_exact(DType::Q4_K.block_bytes()) {
            d.push(decode_f16(&block[0..2])?);
            dmin.push(decode_f16(&block[2..4])?);
            scales.extend_from_slice(&block[4..16]);
            quants.extend_from_slice(&block[16..144]);
        }
        Ok((d, dmin, scales, quants))
    }

    pub(super) fn q6_k_block_scales(matrix: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
        if cols % DType::Q6_K.block_size() != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "Q6_K matrix column count {cols} is not divisible by {}",
                DType::Q6_K.block_size()
            )));
        }

        let blocks_per_row = cols / DType::Q6_K.block_size();
        let total_blocks = checked_mul(rows, blocks_per_row, "Q6_K matrix block count")?;
        let expected_bytes = checked_mul(
            total_blocks,
            DType::Q6_K.block_bytes(),
            "Q6_K matrix byte length",
        )?;
        expect_len(matrix.len(), expected_bytes, "Q6_K matrix")?;

        matrix
            .chunks_exact(DType::Q6_K.block_bytes())
            .map(|block| decode_f16(&block[208..210]))
            .collect()
    }

    pub(super) fn dequantize_q6_k_matrix_transposed(
        matrix: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        if cols % DType::Q6_K.block_size() != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "Q6_K matrix column count {cols} is not divisible by {}",
                DType::Q6_K.block_size()
            )));
        }

        let blocks_per_row = cols / DType::Q6_K.block_size();
        let total_blocks = checked_mul(rows, blocks_per_row, "Q6_K matrix block count")?;
        let expected_bytes = checked_mul(
            total_blocks,
            DType::Q6_K.block_bytes(),
            "Q6_K matrix byte length",
        )?;
        expect_len(matrix.len(), expected_bytes, "Q6_K matrix")?;

        let mut transposed = vec![0.0f32; checked_mul(rows, cols, "Q6_K matrix elements")?];
        for row in 0..rows {
            for block_index in 0..blocks_per_row {
                let block_offset = (row * blocks_per_row + block_index) * DType::Q6_K.block_bytes();
                let block = &matrix[block_offset..block_offset + DType::Q6_K.block_bytes()];
                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = decode_f16(&block[208..210])?;
                for group in 0..2 {
                    let ql_group = &ql[group * 64..(group + 1) * 64];
                    let qh_group = &qh[group * 32..(group + 1) * 32];
                    let scale_group = &scales[group * 8..(group + 1) * 8];
                    let base_col = block_index * 256 + group * 128;
                    for lane in 0..32 {
                        let scale_index = lane / 16;
                        let q1 =
                            ((ql_group[lane] & 0x0f) | ((qh_group[lane] & 0x03) << 4)) as i32 - 32;
                        let q2 = ((ql_group[lane + 32] & 0x0f)
                            | (((qh_group[lane] >> 2) & 0x03) << 4))
                            as i32
                            - 32;
                        let q3 = ((ql_group[lane] >> 4) | (((qh_group[lane] >> 4) & 0x03) << 4))
                            as i32
                            - 32;
                        let q4 = ((ql_group[lane + 32] >> 4)
                            | (((qh_group[lane] >> 6) & 0x03) << 4))
                            as i32
                            - 32;
                        transposed[(base_col + lane) * rows + row] =
                            d * scales_i8(scale_group[scale_index]) * q1 as f32;
                        transposed[(base_col + 32 + lane) * rows + row] =
                            d * scales_i8(scale_group[scale_index + 2]) * q2 as f32;
                        transposed[(base_col + 64 + lane) * rows + row] =
                            d * scales_i8(scale_group[scale_index + 4]) * q3 as f32;
                        transposed[(base_col + 96 + lane) * rows + row] =
                            d * scales_i8(scale_group[scale_index + 6]) * q4 as f32;
                    }
                }
            }
        }
        Ok(transposed)
    }

    pub(super) fn dequantize_q5_k_matrix_transposed(
        matrix: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        if cols % DType::Q5_K.block_size() != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "Q5_K matrix column count {cols} is not divisible by {}",
                DType::Q5_K.block_size()
            )));
        }

        let blocks_per_row = cols / DType::Q5_K.block_size();
        let row_bytes = checked_mul(blocks_per_row, DType::Q5_K.block_bytes(), "Q5_K row bytes")?;
        let expected_bytes = checked_mul(rows, row_bytes, "Q5_K matrix byte length")?;
        expect_len(matrix.len(), expected_bytes, "Q5_K matrix")?;

        let mut transposed = vec![0.0f32; checked_mul(rows, cols, "Q5_K matrix elements")?];
        let mut row_values = vec![0.0f32; cols];
        for row in 0..rows {
            let row_start = row * row_bytes;
            dequantize_q5_k_row(&matrix[row_start..row_start + row_bytes], &mut row_values)?;
            for col in 0..cols {
                transposed[col * rows + row] = row_values[col];
            }
        }
        Ok(transposed)
    }

    pub(super) fn dequantize_q4_k_matrix_transposed(
        matrix: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        if cols % DType::Q4_K.block_size() != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "Q4_K matrix column count {cols} is not divisible by {}",
                DType::Q4_K.block_size()
            )));
        }

        let blocks_per_row = cols / DType::Q4_K.block_size();
        let row_bytes = checked_mul(blocks_per_row, DType::Q4_K.block_bytes(), "Q4_K row bytes")?;
        let expected_bytes = checked_mul(rows, row_bytes, "Q4_K matrix byte length")?;
        expect_len(matrix.len(), expected_bytes, "Q4_K matrix")?;

        let mut transposed = vec![0.0f32; checked_mul(rows, cols, "Q4_K matrix elements")?];
        let mut row_values = vec![0.0f32; cols];
        for row in 0..rows {
            let row_start = row * row_bytes;
            dequantize_q4_k_row(&matrix[row_start..row_start + row_bytes], &mut row_values)?;
            for col in 0..cols {
                transposed[col * rows + row] = row_values[col];
            }
        }
        Ok(transposed)
    }

    pub(super) fn transpose_row_major(
        values_transposed: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        expect_len(
            values_transposed.len(),
            checked_mul(rows, cols, "expanded K-quant matrix elements")?,
            "expanded K-quant transposed matrix",
        )?;
        let mut row_major = vec![0.0f32; values_transposed.len()];
        for row in 0..rows {
            for col in 0..cols {
                row_major[row * cols + col] = values_transposed[col * rows + row];
            }
        }
        Ok(row_major)
    }

    fn scales_i8(value: u8) -> f32 {
        (value as i8) as f32
    }

    fn load_module(
        device: &Arc<DriverCudaDevice>,
        module_name: &'static str,
        ptx: &'static str,
        functions: &[&'static str],
    ) -> Result<()> {
        device
            .load_ptx(Ptx::from_src(ptx), module_name, functions)
            .map_err(|err| {
                let mut context = format!("failed to load PTX module `{module_name}`");
                if let Some(log) = ptx_jit_error_log(ptx) {
                    context.push_str("; CUDA JIT log: ");
                    context.push_str(&log.replace(['\r', '\n'], " "));
                }
                cuda_error(&context, err)
            })
    }

    pub(crate) fn ptx_jit_error_log(ptx: &str) -> Option<String> {
        let ptx = CString::new(ptx).ok()?;
        let mut module = MaybeUninit::uninit();
        let mut log = vec![0u8; 4096];
        let mut options = [
            sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
            sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut option_values = [log.as_mut_ptr().cast::<c_void>(), log.len() as *mut c_void];

        let result = unsafe {
            sys::lib().cuModuleLoadDataEx(
                module.as_mut_ptr(),
                ptx.as_ptr().cast::<c_void>(),
                options.len() as u32,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            )
        };
        if result == sys::CUresult::CUDA_SUCCESS {
            let _ = unsafe { sys::lib().cuModuleUnload(module.assume_init()) };
            return None;
        }

        let end = log.iter().position(|byte| *byte == 0).unwrap_or(log.len());
        let message = String::from_utf8_lossy(&log[..end]).trim().to_string();
        (!message.is_empty()).then_some(message)
    }

    fn one_dim_launch(num_elems: u32) -> LaunchConfig {
        let grid_x = (num_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
        LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn row_launch(rows: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn matmul_launch(m: u32, n: u32) -> LaunchConfig {
        let grid_x = (n + MATMUL_TILE - 1) / MATMUL_TILE;
        let grid_y = (m + MATMUL_TILE - 1) / MATMUL_TILE;
        LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (MATMUL_TILE, MATMUL_TILE, 1),
            shared_mem_bytes: 0,
        }
    }

    #[derive(Debug, Clone)]
    pub struct CudaDevice {
        device: Arc<DriverCudaDevice>,
        modules: LoadedModules,
    }

    pub type CudaBackend = CudaDevice;

    pub struct CudaBytes {
        #[allow(dead_code)]
        data: CudaSlice<u8>,
        len: usize,
    }

    impl std::fmt::Debug for CudaBytes {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaBytes")
                .field("len", &self.len)
                .finish_non_exhaustive()
        }
    }

    impl CudaBytes {
        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn byte_len(&self) -> usize {
            self.len
        }
    }

    pub struct CudaF32Buffer {
        #[allow(dead_code)]
        data: CudaSlice<f32>,
        len: usize,
    }

    impl std::fmt::Debug for CudaF32Buffer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaF32Buffer")
                .field("len", &self.len)
                .field("byte_len", &self.byte_len())
                .finish_non_exhaustive()
        }
    }

    impl CudaF32Buffer {
        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn byte_len(&self) -> usize {
            self.len * std::mem::size_of::<f32>()
        }
    }

    pub struct GpuTensor {
        pub name: String,
        pub dimensions: Vec<usize>,
        pub dtype: DType,
        pub byte_len: usize,
        buffer: CudaBytes,
    }

    impl std::fmt::Debug for GpuTensor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("GpuTensor")
                .field("name", &self.name)
                .field("dimensions", &self.dimensions)
                .field("dtype", &self.dtype)
                .field("byte_len", &self.byte_len)
                .finish_non_exhaustive()
        }
    }

    impl GpuTensor {
        pub fn buffer(&self) -> &CudaBytes {
            &self.buffer
        }
    }

    pub struct GpuF32Tensor {
        pub name: String,
        pub dimensions: Vec<usize>,
        buffer: CudaF32Buffer,
    }

    impl std::fmt::Debug for GpuF32Tensor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("GpuF32Tensor")
                .field("name", &self.name)
                .field("dimensions", &self.dimensions)
                .field("len", &self.buffer.len())
                .field("byte_len", &self.buffer.byte_len())
                .finish_non_exhaustive()
        }
    }

    impl GpuF32Tensor {
        pub fn buffer(&self) -> &CudaF32Buffer {
            &self.buffer
        }

        pub fn len(&self) -> usize {
            self.buffer.len()
        }

        pub fn is_empty(&self) -> bool {
            self.buffer.is_empty()
        }

        pub fn byte_len(&self) -> usize {
            self.buffer.byte_len()
        }
    }

    pub struct CudaQ8_0Matrix {
        scales: CudaF32Buffer,
        quants: CudaBytes,
        rows: usize,
        cols: usize,
    }

    impl std::fmt::Debug for CudaQ8_0Matrix {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaQ8_0Matrix")
                .field("rows", &self.rows)
                .field("cols", &self.cols)
                .field("scale_count", &self.scales.len())
                .field("quant_bytes", &self.quants.byte_len())
                .finish_non_exhaustive()
        }
    }

    impl CudaQ8_0Matrix {
        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn cols(&self) -> usize {
            self.cols
        }

        pub fn scale_count(&self) -> usize {
            self.scales.len()
        }

        pub fn quant_byte_len(&self) -> usize {
            self.quants.byte_len()
        }
    }

    pub type CudaQ4_0Matrix = CudaQ8_0Matrix;
    pub type CudaQ5KMatrix = CudaQ4KMatrix;
    pub type CudaQ6KMatrix = CudaQ4KMatrix;

    enum CudaKQuantMatrixStorage {
        Q4K {
            d: CudaF32Buffer,
            dmin: CudaF32Buffer,
            scales: CudaBytes,
            quants: CudaBytes,
        },
        Q6K {
            d: CudaF32Buffer,
            blocks: CudaBytes,
        },
        ExpandedF32 {
            values_transposed: CudaF32Buffer,
            values_row_major: Option<CudaF32Buffer>,
        },
    }

    pub struct CudaQ4KMatrix {
        storage: CudaKQuantMatrixStorage,
        rows: usize,
        cols: usize,
    }

    impl std::fmt::Debug for CudaQ4KMatrix {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaQ4KMatrix")
                .field("rows", &self.rows)
                .field("cols", &self.cols)
                .field("byte_len", &self.byte_len())
                .finish_non_exhaustive()
        }
    }

    impl CudaQ4KMatrix {
        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn cols(&self) -> usize {
            self.cols
        }

        pub fn byte_len(&self) -> usize {
            match &self.storage {
                CudaKQuantMatrixStorage::Q4K {
                    d,
                    dmin,
                    scales,
                    quants,
                } => d
                    .byte_len()
                    .saturating_add(dmin.byte_len())
                    .saturating_add(scales.byte_len())
                    .saturating_add(quants.byte_len()),
                CudaKQuantMatrixStorage::Q6K { d, blocks } => {
                    d.byte_len().saturating_add(blocks.byte_len())
                }
                CudaKQuantMatrixStorage::ExpandedF32 {
                    values_transposed,
                    values_row_major,
                } => values_transposed
                    .byte_len()
                    .saturating_add(values_row_major.as_ref().map_or(0, CudaF32Buffer::byte_len)),
            }
        }
    }

    pub struct CudaAdaptiveKvRoutes {
        data: CudaSlice<u32>,
        capacity: usize,
        len: usize,
    }

    impl std::fmt::Debug for CudaAdaptiveKvRoutes {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaAdaptiveKvRoutes")
                .field("capacity", &self.capacity)
                .field("len", &self.len)
                .finish_non_exhaustive()
        }
    }

    impl CudaAdaptiveKvRoutes {
        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn allocated_bytes(&self) -> u64 {
            self.capacity
                .saturating_mul(std::mem::size_of::<u32>())
                .try_into()
                .unwrap_or(u64::MAX)
        }

        pub fn clear(&mut self) {
            self.len = 0;
        }

        pub fn truncate(&mut self, new_len: usize) {
            self.len = self.len.min(new_len);
        }
    }

    pub struct CudaLayerKvCache {
        keys: CudaF32Buffer,
        values: CudaF32Buffer,
        page_table: CudaSlice<u32>,
        capacity: usize,
        len: usize,
        width: usize,
        page_tokens: usize,
        page_count: usize,
    }

    impl std::fmt::Debug for CudaLayerKvCache {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaLayerKvCache")
                .field("capacity", &self.capacity)
                .field("len", &self.len)
                .field("width", &self.width)
                .field("page_tokens", &self.page_tokens)
                .field("page_count", &self.page_count)
                .finish_non_exhaustive()
        }
    }

    impl CudaLayerKvCache {
        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn width(&self) -> usize {
            self.width
        }

        pub fn page_tokens(&self) -> usize {
            self.page_tokens
        }

        pub fn page_count(&self) -> usize {
            self.page_count
        }

        pub fn allocated_bytes(&self) -> u64 {
            self.keys
                .byte_len()
                .saturating_add(self.values.byte_len())
                .saturating_add(self.page_count.saturating_mul(std::mem::size_of::<u32>()))
                .try_into()
                .unwrap_or(u64::MAX)
        }

        pub fn clear(&mut self) {
            self.len = 0;
        }

        pub fn truncate(&mut self, new_len: usize) {
            self.len = self.len.min(new_len);
        }
    }

    pub struct CudaQ8LayerKvCache {
        keys: CudaBytes,
        values: CudaBytes,
        key_scales: CudaF32Buffer,
        value_scales: CudaF32Buffer,
        page_table: CudaSlice<u32>,
        capacity: usize,
        len: usize,
        width: usize,
        page_tokens: usize,
        page_count: usize,
    }

    impl std::fmt::Debug for CudaQ8LayerKvCache {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaQ8LayerKvCache")
                .field("capacity", &self.capacity)
                .field("len", &self.len)
                .field("width", &self.width)
                .field("allocated_bytes", &self.allocated_bytes())
                .finish_non_exhaustive()
        }
    }

    impl CudaQ8LayerKvCache {
        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn width(&self) -> usize {
            self.width
        }

        pub fn page_tokens(&self) -> usize {
            self.page_tokens
        }

        pub fn page_count(&self) -> usize {
            self.page_count
        }

        pub fn allocated_bytes(&self) -> u64 {
            self.keys
                .byte_len()
                .saturating_add(self.values.byte_len())
                .saturating_add(self.key_scales.byte_len())
                .saturating_add(self.value_scales.byte_len())
                .saturating_add(self.page_count.saturating_mul(std::mem::size_of::<u32>()))
                .try_into()
                .unwrap_or(u64::MAX)
        }

        pub fn clear(&mut self) {
            self.len = 0;
        }

        pub fn truncate(&mut self, new_len: usize) {
            self.len = self.len.min(new_len);
        }
    }

    pub struct CudaKeyQ4ValueQ8LayerKvCache {
        keys: CudaBytes,
        values: CudaBytes,
        key_scales: CudaF32Buffer,
        value_scales: CudaF32Buffer,
        page_table: CudaSlice<u32>,
        capacity: usize,
        len: usize,
        width: usize,
        page_tokens: usize,
        page_count: usize,
    }

    impl std::fmt::Debug for CudaKeyQ4ValueQ8LayerKvCache {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaKeyQ4ValueQ8LayerKvCache")
                .field("capacity", &self.capacity)
                .field("len", &self.len)
                .field("width", &self.width)
                .field("allocated_bytes", &self.allocated_bytes())
                .finish_non_exhaustive()
        }
    }

    impl CudaKeyQ4ValueQ8LayerKvCache {
        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        pub fn width(&self) -> usize {
            self.width
        }

        pub fn page_tokens(&self) -> usize {
            self.page_tokens
        }

        pub fn page_count(&self) -> usize {
            self.page_count
        }

        pub fn allocated_bytes(&self) -> u64 {
            self.keys
                .byte_len()
                .saturating_add(self.values.byte_len())
                .saturating_add(self.key_scales.byte_len())
                .saturating_add(self.value_scales.byte_len())
                .saturating_add(self.page_count.saturating_mul(std::mem::size_of::<u32>()))
                .try_into()
                .unwrap_or(u64::MAX)
        }

        pub fn clear(&mut self) {
            self.len = 0;
        }

        pub fn truncate(&mut self, new_len: usize) {
            self.len = self.len.min(new_len);
        }
    }

    #[derive(Debug)]
    pub struct GpuModelWeights {
        tensors: Vec<GpuTensor>,
        total_bytes: u64,
    }

    impl GpuModelWeights {
        pub fn from_gguf(device: &CudaDevice, gguf: &GgufFile) -> Result<Self> {
            let mut tensors = Vec::with_capacity(gguf.tensor_infos().len());
            let mut total_bytes = 0u64;
            for info in gguf.tensor_infos() {
                let bytes = gguf.tensor_data(&info.name)?;
                let buffer = device.upload_bytes(bytes)?;
                total_bytes = total_bytes.saturating_add(bytes.len() as u64);
                tensors.push(GpuTensor {
                    name: info.name.clone(),
                    dimensions: info.dimensions.clone(),
                    dtype: info.dtype,
                    byte_len: bytes.len(),
                    buffer,
                });
            }
            Ok(Self {
                tensors,
                total_bytes,
            })
        }

        pub fn tensors(&self) -> &[GpuTensor] {
            &self.tensors
        }

        pub fn tensor_count(&self) -> usize {
            self.tensors.len()
        }

        pub fn total_bytes(&self) -> u64 {
            self.total_bytes
        }
    }

    impl CudaDevice {
        pub fn new(ordinal: usize) -> Result<Self> {
            let device = DriverCudaDevice::new(ordinal).map_err(|err| {
                XrtError::Cuda(format!("failed to open CUDA device {ordinal}: {err}"))
            })?;

            info!("initialized CUDA backend on device {}", ordinal);
            Ok(Self {
                device,
                modules: MODULES,
            })
        }

        pub fn inner(&self) -> &Arc<DriverCudaDevice> {
            &self.device
        }

        pub fn name(&self) -> Result<String> {
            self.device
                .name()
                .map_err(|err| cuda_error("failed to query CUDA device name", err))
        }

        pub fn memory_info(&self) -> Result<(u64, u64)> {
            let (free, total) = driver_result::mem_get_info()
                .map_err(|err| cuda_error("failed to query CUDA memory info", err))?;
            Ok((free as u64, total as u64))
        }

        pub fn upload_bytes(&self, bytes: &[u8]) -> Result<CudaBytes> {
            let data = self
                .device
                .htod_copy(bytes.to_vec())
                .map_err(|err| cuda_error("failed to copy bytes to device", err))?;
            Ok(CudaBytes {
                data,
                len: bytes.len(),
            })
        }

        pub fn upload_f32(&self, values: &[f32]) -> Result<CudaF32Buffer> {
            let data = self
                .device
                .htod_copy(values.to_vec())
                .map_err(|err| cuda_error("failed to copy f32 buffer to device", err))?;
            Ok(CudaF32Buffer {
                data,
                len: values.len(),
            })
        }

        pub fn upload_f32_into(
            &self,
            values: &[f32],
            destination: &mut CudaF32Buffer,
        ) -> Result<()> {
            expect_len(destination.len(), values.len(), "f32 upload destination")?;
            if values.is_empty() {
                return Ok(());
            }
            self.device
                .htod_sync_copy_into(values, &mut destination.data)
                .map_err(|err| cuda_error("failed to copy f32 values into device buffer", err))
        }

        pub fn copy_f32_device(
            &self,
            source: &CudaF32Buffer,
            destination: &mut CudaF32Buffer,
        ) -> Result<()> {
            expect_len(
                destination.len(),
                source.len(),
                "f32 device copy destination",
            )?;
            if source.is_empty() {
                return Ok(());
            }
            self.device
                .dtod_copy(&source.data, &mut destination.data)
                .map_err(|err| cuda_error("failed to copy f32 device buffer", err))
        }

        pub fn download_f32(&self, buffer: &CudaF32Buffer) -> Result<Vec<f32>> {
            self.device
                .dtoh_sync_copy(&buffer.data)
                .map_err(|err| cuda_error("failed to copy f32 buffer to host", err))
        }

        pub fn zeros_f32(&self, len: usize) -> Result<CudaF32Buffer> {
            let data = self
                .device
                .alloc_zeros::<f32>(len)
                .map_err(|err| cuda_error("failed to allocate f32 buffer on device", err))?;
            Ok(CudaF32Buffer { data, len })
        }

        pub fn zeros_bytes(&self, len: usize) -> Result<CudaBytes> {
            let data = self
                .device
                .alloc_zeros::<u8>(len)
                .map_err(|err| cuda_error("failed to allocate byte buffer on device", err))?;
            Ok(CudaBytes { data, len })
        }

        pub fn alloc_adaptive_kv_routes(&self, capacity: usize) -> Result<CudaAdaptiveKvRoutes> {
            let data = self
                .device
                .alloc_zeros::<u32>(capacity)
                .map_err(|err| cuda_error("failed to allocate CUDA adaptive KV routes", err))?;
            Ok(CudaAdaptiveKvRoutes {
                data,
                capacity,
                len: 0,
            })
        }

        pub fn grow_adaptive_kv_routes(
            &self,
            routes: &mut CudaAdaptiveKvRoutes,
            new_capacity: usize,
        ) -> Result<()> {
            if new_capacity < routes.capacity {
                return Err(XrtError::Runtime(format!(
                    "cannot shrink CUDA adaptive KV route capacity from {} to {new_capacity}",
                    routes.capacity
                )));
            }
            if new_capacity == routes.capacity {
                return Ok(());
            }
            let mut grown = self.alloc_adaptive_kv_routes(new_capacity)?;
            self.copy_page_table_prefix(
                &routes.data,
                &mut grown.data,
                routes.len,
                "CUDA adaptive KV routes",
            )?;
            self.device.synchronize().map_err(|err| {
                cuda_error("failed to synchronize CUDA adaptive KV route growth", err)
            })?;
            grown.len = routes.len;
            *routes = grown;
            Ok(())
        }

        pub fn replace_adaptive_kv_routes(
            &self,
            routes: &mut CudaAdaptiveKvRoutes,
            hot_mask: &[u8],
        ) -> Result<()> {
            if hot_mask.len() > routes.capacity {
                return Err(XrtError::Runtime(format!(
                    "CUDA adaptive KV route length {} exceeds capacity {}",
                    hot_mask.len(),
                    routes.capacity
                )));
            }
            let mut hot_position = 0usize;
            let mut cold_position = 0usize;
            let mut encoded = Vec::with_capacity(hot_mask.len());
            for &is_hot in hot_mask {
                let local_position = if is_hot == 0 {
                    let position = cold_position;
                    cold_position += 1;
                    position
                } else {
                    let position = hot_position;
                    hot_position += 1;
                    position
                };
                encoded.push(encode_adaptive_kv_route(is_hot != 0, local_position)?);
            }
            if !encoded.is_empty() {
                let mut destination =
                    routes.data.try_slice_mut(..encoded.len()).ok_or_else(|| {
                        XrtError::Runtime(
                            "failed to create CUDA adaptive KV route destination view".to_string(),
                        )
                    })?;
                self.device
                    .htod_sync_copy_into(&encoded, &mut destination)
                    .map_err(|err| cuda_error("failed to replace CUDA adaptive KV routes", err))?;
            }
            routes.len = encoded.len();
            Ok(())
        }

        pub fn append_adaptive_kv_route(
            &self,
            routes: &mut CudaAdaptiveKvRoutes,
            is_hot: bool,
            local_position: usize,
        ) -> Result<()> {
            if routes.len >= routes.capacity {
                return Err(XrtError::Runtime(format!(
                    "CUDA adaptive KV routes are full: len={}, capacity={}",
                    routes.len, routes.capacity
                )));
            }
            let index = to_u32(routes.len, "CUDA adaptive KV route index")?;
            let encoded = encode_adaptive_kv_route(is_hot, local_position)?;
            let func = self.function(self.modules.attention, "adaptive_kv_route_write_kernel")?;
            unsafe { func.launch(one_dim_launch(1), (&mut routes.data, index, encoded)) }
                .map_err(|err| cuda_error("failed to append CUDA adaptive KV route", err))?;
            routes.len += 1;
            Ok(())
        }

        pub fn alloc_layer_kv_cache(
            &self,
            capacity: usize,
            width: usize,
        ) -> Result<CudaLayerKvCache> {
            self.alloc_paged_layer_kv_cache(capacity, width, capacity.max(1))
        }

        pub fn alloc_paged_layer_kv_cache(
            &self,
            capacity: usize,
            width: usize,
            page_tokens: usize,
        ) -> Result<CudaLayerKvCache> {
            let elements = checked_mul(capacity, width, "CUDA paged KV cache elements")?;
            let (page_table, page_tokens, page_count) =
                self.alloc_identity_page_table(capacity, page_tokens, "CUDA F32 KV")?;
            Ok(CudaLayerKvCache {
                keys: self.zeros_f32(elements)?,
                values: self.zeros_f32(elements)?,
                page_table,
                capacity,
                len: 0,
                width,
                page_tokens,
                page_count,
            })
        }

        pub fn remap_paged_layer_kv_pages(
            &self,
            cache: &mut CudaLayerKvCache,
            page_map: &[u32],
        ) -> Result<()> {
            self.remap_page_table(
                cache.capacity,
                cache.page_tokens,
                cache.page_count,
                &mut cache.page_table,
                page_map,
                "CUDA F32 KV",
            )
        }

        pub fn alloc_q8_layer_kv_cache(
            &self,
            capacity: usize,
            width: usize,
        ) -> Result<CudaQ8LayerKvCache> {
            self.alloc_paged_q8_layer_kv_cache(capacity, width, capacity.max(1))
        }

        pub fn alloc_paged_q8_layer_kv_cache(
            &self,
            capacity: usize,
            width: usize,
            page_tokens: usize,
        ) -> Result<CudaQ8LayerKvCache> {
            let _ = q8_layer_kv_allocated_bytes(capacity, width)?;
            let elements = checked_mul(capacity, width, "CUDA Q8 KV cache elements")?;
            let (page_table, page_tokens, page_count) =
                self.alloc_identity_page_table(capacity, page_tokens, "CUDA Q8 KV")?;
            Ok(CudaQ8LayerKvCache {
                keys: self.zeros_bytes(elements)?,
                values: self.zeros_bytes(elements)?,
                key_scales: self.zeros_f32(capacity)?,
                value_scales: self.zeros_f32(capacity)?,
                page_table,
                capacity,
                len: 0,
                width,
                page_tokens,
                page_count,
            })
        }

        pub fn remap_paged_q8_layer_kv_pages(
            &self,
            cache: &mut CudaQ8LayerKvCache,
            page_map: &[u32],
        ) -> Result<()> {
            self.remap_page_table(
                cache.capacity,
                cache.page_tokens,
                cache.page_count,
                &mut cache.page_table,
                page_map,
                "CUDA Q8 KV",
            )
        }

        pub fn alloc_key_q4_value_q8_layer_kv_cache(
            &self,
            capacity: usize,
            width: usize,
        ) -> Result<CudaKeyQ4ValueQ8LayerKvCache> {
            self.alloc_paged_key_q4_value_q8_layer_kv_cache(capacity, width, capacity.max(1))
        }

        pub fn alloc_paged_key_q4_value_q8_layer_kv_cache(
            &self,
            capacity: usize,
            width: usize,
            page_tokens: usize,
        ) -> Result<CudaKeyQ4ValueQ8LayerKvCache> {
            let _ = kq4_vq8_layer_kv_allocated_bytes(capacity, width)?;
            let key_bytes =
                checked_mul(capacity, kq4_key_row_bytes(width), "CUDA KQ4/VQ8 key bytes")?;
            let value_bytes = checked_mul(capacity, width, "CUDA KQ4/VQ8 value bytes")?;
            let key_scales = checked_mul(
                capacity,
                kq4_key_groups(width),
                "CUDA KQ4/VQ8 key scale count",
            )?;
            let (page_table, page_tokens, page_count) =
                self.alloc_identity_page_table(capacity, page_tokens, "CUDA KQ4/VQ8 KV")?;
            Ok(CudaKeyQ4ValueQ8LayerKvCache {
                keys: self.zeros_bytes(key_bytes)?,
                values: self.zeros_bytes(value_bytes)?,
                key_scales: self.zeros_f32(key_scales)?,
                value_scales: self.zeros_f32(capacity)?,
                page_table,
                capacity,
                len: 0,
                width,
                page_tokens,
                page_count,
            })
        }

        pub fn remap_paged_key_q4_value_q8_layer_kv_pages(
            &self,
            cache: &mut CudaKeyQ4ValueQ8LayerKvCache,
            page_map: &[u32],
        ) -> Result<()> {
            self.remap_page_table(
                cache.capacity,
                cache.page_tokens,
                cache.page_count,
                &mut cache.page_table,
                page_map,
                "CUDA KQ4/VQ8 KV",
            )
        }

        fn alloc_identity_page_table(
            &self,
            capacity: usize,
            page_tokens: usize,
            what: &str,
        ) -> Result<(CudaSlice<u32>, usize, usize)> {
            let page_tokens = page_tokens.max(1);
            let page_count = capacity.div_ceil(page_tokens);
            let page_table = (0..page_count)
                .map(|page| to_u32(page, &format!("{what} page index")))
                .collect::<Result<Vec<_>>>()?;
            let page_table = self
                .device
                .htod_copy(page_table)
                .map_err(|err| cuda_error(&format!("failed to upload {what} page table"), err))?;
            Ok((page_table, page_tokens, page_count))
        }

        fn remap_page_table(
            &self,
            capacity: usize,
            page_tokens: usize,
            page_count: usize,
            page_table: &mut CudaSlice<u32>,
            page_map: &[u32],
            what: &str,
        ) -> Result<()> {
            if capacity % page_tokens != 0 {
                return Err(XrtError::Unsupported(format!(
                    "{what} page remapping requires a full final page"
                )));
            }
            expect_len(page_map.len(), page_count, &format!("{what} page map"))?;
            let mut seen = vec![false; page_count];
            for &page in page_map {
                let page = usize::try_from(page).map_err(|_| {
                    XrtError::Shape(format!("{what} page index does not fit usize"))
                })?;
                if page >= page_count || std::mem::replace(&mut seen[page], true) {
                    return Err(XrtError::Shape(format!(
                        "{what} page map must be a permutation of physical pages"
                    )));
                }
            }
            self.device
                .htod_sync_copy_into(page_map, page_table)
                .map_err(|err| cuda_error(&format!("failed to update {what} page table"), err))
        }

        fn copy_f32_prefix(
            &self,
            source: &CudaF32Buffer,
            destination: &mut CudaF32Buffer,
            len: usize,
            what: &str,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            if len > source.len() || len > destination.len() {
                return Err(XrtError::Runtime(format!(
                    "{what} copy length {len} exceeds source {} or destination {}",
                    source.len(),
                    destination.len()
                )));
            }
            let source_view = source.data.slice(..len);
            let mut destination_view = destination.data.try_slice_mut(..len).ok_or_else(|| {
                XrtError::Runtime(format!("failed to create {what} destination view"))
            })?;
            self.device
                .dtod_copy(&source_view, &mut destination_view)
                .map_err(|err| cuda_error(&format!("failed to copy {what}"), err))
        }

        fn copy_byte_prefix(
            &self,
            source: &CudaBytes,
            destination: &mut CudaBytes,
            len: usize,
            what: &str,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            if len > source.len() || len > destination.len() {
                return Err(XrtError::Runtime(format!(
                    "{what} copy length {len} exceeds source {} or destination {}",
                    source.len(),
                    destination.len()
                )));
            }
            let source_view = source.data.slice(..len);
            let mut destination_view = destination.data.try_slice_mut(..len).ok_or_else(|| {
                XrtError::Runtime(format!("failed to create {what} destination view"))
            })?;
            self.device
                .dtod_copy(&source_view, &mut destination_view)
                .map_err(|err| cuda_error(&format!("failed to copy {what}"), err))
        }

        fn copy_page_table_prefix(
            &self,
            source: &CudaSlice<u32>,
            destination: &mut CudaSlice<u32>,
            len: usize,
            what: &str,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            if len > source.len() || len > destination.len() {
                return Err(XrtError::Runtime(format!(
                    "{what} copy length {len} exceeds source {} or destination {}",
                    source.len(),
                    destination.len()
                )));
            }
            let source_view = source.slice(..len);
            let mut destination_view = destination.try_slice_mut(..len).ok_or_else(|| {
                XrtError::Runtime(format!("failed to create {what} destination view"))
            })?;
            self.device
                .dtod_copy(&source_view, &mut destination_view)
                .map_err(|err| cuda_error(&format!("failed to copy {what}"), err))
        }

        pub fn grow_layer_kv_cache(
            &self,
            cache: &mut CudaLayerKvCache,
            new_capacity: usize,
        ) -> Result<()> {
            if new_capacity < cache.capacity {
                return Err(XrtError::Runtime(format!(
                    "cannot shrink CUDA F32 KV capacity from {} to {new_capacity}",
                    cache.capacity
                )));
            }
            if new_capacity == cache.capacity {
                return Ok(());
            }
            let allocated_elements = checked_mul(
                cache.capacity,
                cache.width,
                "CUDA F32 KV allocated elements",
            )?;
            let mut grown =
                self.alloc_paged_layer_kv_cache(new_capacity, cache.width, cache.page_tokens)?;
            self.copy_f32_prefix(
                &cache.keys,
                &mut grown.keys,
                allocated_elements,
                "CUDA F32 KV keys",
            )?;
            self.copy_f32_prefix(
                &cache.values,
                &mut grown.values,
                allocated_elements,
                "CUDA F32 KV values",
            )?;
            self.copy_page_table_prefix(
                &cache.page_table,
                &mut grown.page_table,
                cache.page_count,
                "CUDA F32 KV page table",
            )?;
            self.device
                .synchronize()
                .map_err(|err| cuda_error("failed to synchronize CUDA F32 KV growth", err))?;
            grown.len = cache.len;
            *cache = grown;
            Ok(())
        }

        pub fn grow_q8_layer_kv_cache(
            &self,
            cache: &mut CudaQ8LayerKvCache,
            new_capacity: usize,
        ) -> Result<()> {
            if new_capacity < cache.capacity {
                return Err(XrtError::Runtime(format!(
                    "cannot shrink CUDA Q8 KV capacity from {} to {new_capacity}",
                    cache.capacity
                )));
            }
            if new_capacity == cache.capacity {
                return Ok(());
            }
            let allocated_elements =
                checked_mul(cache.capacity, cache.width, "CUDA Q8 KV allocated elements")?;
            let mut grown =
                self.alloc_paged_q8_layer_kv_cache(new_capacity, cache.width, cache.page_tokens)?;
            self.copy_byte_prefix(
                &cache.keys,
                &mut grown.keys,
                allocated_elements,
                "CUDA Q8 KV keys",
            )?;
            self.copy_byte_prefix(
                &cache.values,
                &mut grown.values,
                allocated_elements,
                "CUDA Q8 KV values",
            )?;
            self.copy_f32_prefix(
                &cache.key_scales,
                &mut grown.key_scales,
                cache.capacity,
                "CUDA Q8 KV key scales",
            )?;
            self.copy_f32_prefix(
                &cache.value_scales,
                &mut grown.value_scales,
                cache.capacity,
                "CUDA Q8 KV value scales",
            )?;
            self.copy_page_table_prefix(
                &cache.page_table,
                &mut grown.page_table,
                cache.page_count,
                "CUDA Q8 KV page table",
            )?;
            self.device
                .synchronize()
                .map_err(|err| cuda_error("failed to synchronize CUDA Q8 KV growth", err))?;
            grown.len = cache.len;
            *cache = grown;
            Ok(())
        }

        pub fn grow_key_q4_value_q8_layer_kv_cache(
            &self,
            cache: &mut CudaKeyQ4ValueQ8LayerKvCache,
            new_capacity: usize,
        ) -> Result<()> {
            if new_capacity < cache.capacity {
                return Err(XrtError::Runtime(format!(
                    "cannot shrink CUDA KQ4/VQ8 KV capacity from {} to {new_capacity}",
                    cache.capacity
                )));
            }
            if new_capacity == cache.capacity {
                return Ok(());
            }
            let key_bytes = checked_mul(
                cache.capacity,
                kq4_key_row_bytes(cache.width),
                "CUDA KQ4/VQ8 allocated key bytes",
            )?;
            let value_bytes = checked_mul(
                cache.capacity,
                cache.width,
                "CUDA KQ4/VQ8 allocated value bytes",
            )?;
            let key_scales = checked_mul(
                cache.capacity,
                kq4_key_groups(cache.width),
                "CUDA KQ4/VQ8 allocated key scales",
            )?;
            let mut grown = self.alloc_paged_key_q4_value_q8_layer_kv_cache(
                new_capacity,
                cache.width,
                cache.page_tokens,
            )?;
            self.copy_byte_prefix(
                &cache.keys,
                &mut grown.keys,
                key_bytes,
                "CUDA KQ4/VQ8 KV keys",
            )?;
            self.copy_byte_prefix(
                &cache.values,
                &mut grown.values,
                value_bytes,
                "CUDA KQ4/VQ8 KV values",
            )?;
            self.copy_f32_prefix(
                &cache.key_scales,
                &mut grown.key_scales,
                key_scales,
                "CUDA KQ4/VQ8 KV key scales",
            )?;
            self.copy_f32_prefix(
                &cache.value_scales,
                &mut grown.value_scales,
                cache.capacity,
                "CUDA KQ4/VQ8 KV value scales",
            )?;
            self.copy_page_table_prefix(
                &cache.page_table,
                &mut grown.page_table,
                cache.page_count,
                "CUDA KQ4/VQ8 KV page table",
            )?;
            self.device
                .synchronize()
                .map_err(|err| cuda_error("failed to synchronize CUDA KQ4/VQ8 KV growth", err))?;
            grown.len = cache.len;
            *cache = grown;
            Ok(())
        }

        pub fn append_layer_kv(
            &self,
            cache: &mut CudaLayerKvCache,
            key: &CudaF32Buffer,
            value: &CudaF32Buffer,
        ) -> Result<()> {
            expect_len(key.len(), cache.width, "CUDA KV key")?;
            expect_len(value.len(), cache.width, "CUDA KV value")?;
            if cache.len >= cache.capacity {
                return Err(XrtError::Runtime(format!(
                    "CUDA KV cache is full: len={}, capacity={}",
                    cache.len, cache.capacity
                )));
            }
            if cache.width == 0 {
                cache.len += 1;
                return Ok(());
            }

            let slot_u32 = to_u32(cache.len, "CUDA KV slot")?;
            let width_u32 = to_u32(cache.width, "CUDA KV width")?;
            let page_tokens_u32 = to_u32(cache.page_tokens, "CUDA KV page tokens")?;
            let func = self.function(self.modules.attention, "paged_kv_cache_append_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(width_u32),
                    (
                        &mut cache.keys.data,
                        &mut cache.values.data,
                        &cache.page_table,
                        &key.data,
                        &value.data,
                        slot_u32,
                        width_u32,
                        page_tokens_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch paged KV cache append kernel", err))?;
            cache.len += 1;
            Ok(())
        }

        pub fn gather_paged_layer_kv(
            &self,
            cache: &CudaLayerKvCache,
            start_position: usize,
            count: usize,
        ) -> Result<(CudaF32Buffer, CudaF32Buffer)> {
            let end = start_position.checked_add(count).ok_or_else(|| {
                XrtError::Runtime("CUDA paged KV gather range overflow".to_string())
            })?;
            if end > cache.len {
                return Err(XrtError::Runtime(format!(
                    "CUDA paged KV gather range {start_position}..{end} exceeds cache length {}",
                    cache.len
                )));
            }
            let elements = checked_mul(count, cache.width, "CUDA paged KV gather elements")?;
            let mut keys = self.zeros_f32(elements)?;
            let mut values = self.zeros_f32(elements)?;
            if elements == 0 {
                return Ok((keys, values));
            }

            let func = self.function(self.modules.attention, "paged_kv_cache_gather_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(to_u32(elements, "CUDA paged KV gather elements")?),
                    (
                        &cache.keys.data,
                        &cache.values.data,
                        &cache.page_table,
                        &mut keys.data,
                        &mut values.data,
                        to_u32(count, "CUDA paged KV gather count")?,
                        to_u32(cache.width, "CUDA paged KV gather width")?,
                        to_u32(cache.page_tokens, "CUDA paged KV gather page tokens")?,
                        to_u32(start_position, "CUDA paged KV gather start position")?,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch paged KV gather kernel", err))?;
            Ok((keys, values))
        }

        pub fn copy_layer_kv(
            &self,
            cache: &CudaLayerKvCache,
            position: usize,
        ) -> Result<(CudaF32Buffer, CudaF32Buffer)> {
            self.gather_paged_layer_kv(cache, position, 1)
        }

        pub fn append_q8_layer_kv(
            &self,
            cache: &mut CudaQ8LayerKvCache,
            key: &CudaF32Buffer,
            value: &CudaF32Buffer,
        ) -> Result<()> {
            expect_len(key.len(), cache.width, "CUDA Q8 KV key")?;
            expect_len(value.len(), cache.width, "CUDA Q8 KV value")?;
            if cache.len >= cache.capacity {
                return Err(XrtError::Runtime(format!(
                    "CUDA Q8 KV cache is full: len={}, capacity={}",
                    cache.len, cache.capacity
                )));
            }
            if cache.width == 0 {
                cache.len += 1;
                return Ok(());
            }

            let slot_u32 = to_u32(cache.len, "CUDA Q8 KV slot")?;
            let width_u32 = to_u32(cache.width, "CUDA Q8 KV width")?;
            let page_tokens_u32 = to_u32(cache.page_tokens, "CUDA Q8 KV page tokens")?;
            let func = self.function(self.modules.attention, "q8_kv_cache_append_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(1),
                    (
                        &mut cache.keys.data,
                        &mut cache.values.data,
                        &mut cache.key_scales.data,
                        &mut cache.value_scales.data,
                        &key.data,
                        &value.data,
                        slot_u32,
                        width_u32,
                        &cache.page_table,
                        page_tokens_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch Q8 KV cache append kernel", err))?;
            cache.len += 1;
            Ok(())
        }

        pub fn append_key_q4_value_q8_layer_kv(
            &self,
            cache: &mut CudaKeyQ4ValueQ8LayerKvCache,
            key: &CudaF32Buffer,
            value: &CudaF32Buffer,
        ) -> Result<()> {
            expect_len(key.len(), cache.width, "CUDA KQ4/VQ8 KV key")?;
            expect_len(value.len(), cache.width, "CUDA KQ4/VQ8 KV value")?;
            if cache.len >= cache.capacity {
                return Err(XrtError::Runtime(format!(
                    "CUDA KQ4/VQ8 KV cache is full: len={}, capacity={}",
                    cache.len, cache.capacity
                )));
            }
            if cache.width == 0 {
                cache.len += 1;
                return Ok(());
            }

            let slot_u32 = to_u32(cache.len, "CUDA KQ4/VQ8 KV slot")?;
            let width_u32 = to_u32(cache.width, "CUDA KQ4/VQ8 KV width")?;
            let page_tokens_u32 = to_u32(cache.page_tokens, "CUDA KQ4/VQ8 KV page tokens")?;
            let func = self.function(self.modules.attention, "kq4_vq8_kv_cache_append_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(1),
                    (
                        &mut cache.keys.data,
                        &mut cache.values.data,
                        &mut cache.key_scales.data,
                        &mut cache.value_scales.data,
                        &key.data,
                        &value.data,
                        slot_u32,
                        width_u32,
                        &cache.page_table,
                        page_tokens_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch KQ4/VQ8 KV cache append kernel", err))?;
            cache.len += 1;
            Ok(())
        }

        pub fn copy_key_q4_value_q8_layer_kv_row(
            &self,
            source: &CudaKeyQ4ValueQ8LayerKvCache,
            source_position: usize,
            destination: &mut CudaKeyQ4ValueQ8LayerKvCache,
        ) -> Result<()> {
            if source_position >= source.len {
                return Err(XrtError::Runtime(format!(
                    "CUDA KQ4/VQ8 source position {source_position} is out of range for len {}",
                    source.len
                )));
            }
            expect_len(
                destination.width,
                source.width,
                "CUDA KQ4/VQ8 row-copy destination width",
            )?;
            if destination.len >= destination.capacity {
                return Err(XrtError::Runtime(format!(
                    "CUDA KQ4/VQ8 row-copy destination is full: len={}, capacity={}",
                    destination.len, destination.capacity
                )));
            }
            if source.width == 0 {
                destination.len += 1;
                return Ok(());
            }

            let source_position_u32 =
                to_u32(source_position, "CUDA KQ4/VQ8 row-copy source position")?;
            let destination_position_u32 = to_u32(
                destination.len,
                "CUDA KQ4/VQ8 row-copy destination position",
            )?;
            let width_u32 = to_u32(source.width, "CUDA KQ4/VQ8 row-copy width")?;
            let source_page_tokens_u32 = to_u32(
                source.page_tokens,
                "CUDA KQ4/VQ8 row-copy source page tokens",
            )?;
            let destination_page_tokens_u32 = to_u32(
                destination.page_tokens,
                "CUDA KQ4/VQ8 row-copy destination page tokens",
            )?;
            let func = self.function(self.modules.attention, "kq4_vq8_kv_cache_copy_row_kernel")?;
            let mut params = vec![
                (&source.keys.data).as_kernel_param(),
                (&source.values.data).as_kernel_param(),
                (&source.key_scales.data).as_kernel_param(),
                (&source.value_scales.data).as_kernel_param(),
                (&source.page_table).as_kernel_param(),
                (&mut destination.keys.data).as_kernel_param(),
                (&mut destination.values.data).as_kernel_param(),
                (&mut destination.key_scales.data).as_kernel_param(),
                (&mut destination.value_scales.data).as_kernel_param(),
                (&destination.page_table).as_kernel_param(),
                source_position_u32.as_kernel_param(),
                destination_position_u32.as_kernel_param(),
                width_u32.as_kernel_param(),
                source_page_tokens_u32.as_kernel_param(),
                destination_page_tokens_u32.as_kernel_param(),
            ];
            unsafe { func.launch(one_dim_launch(width_u32), &mut params) }
                .map_err(|err| cuda_error("failed to launch KQ4/VQ8 KV row-copy kernel", err))?;
            destination.len += 1;
            Ok(())
        }

        pub fn dequantize_q8_layer_kv(
            &self,
            cache: &CudaQ8LayerKvCache,
            position: usize,
        ) -> Result<(CudaF32Buffer, CudaF32Buffer)> {
            if position >= cache.len {
                return Err(XrtError::Runtime(format!(
                    "CUDA Q8 KV position {position} is out of range for len {}",
                    cache.len
                )));
            }
            if cache.width == 0 {
                return Ok((self.zeros_f32(0)?, self.zeros_f32(0)?));
            }

            let position_u32 = to_u32(position, "CUDA Q8 KV position")?;
            let width_u32 = to_u32(cache.width, "CUDA Q8 KV width")?;
            let page_tokens_u32 = to_u32(cache.page_tokens, "CUDA Q8 KV page tokens")?;
            let mut key_dev = self
                .device
                .alloc_zeros::<f32>(cache.width)
                .map_err(|err| cuda_error("failed to allocate Q8 KV key output", err))?;
            let mut value_dev = self
                .device
                .alloc_zeros::<f32>(cache.width)
                .map_err(|err| cuda_error("failed to allocate Q8 KV value output", err))?;
            let func = self.function(self.modules.attention, "q8_kv_cache_dequantize_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(width_u32),
                    (
                        &cache.keys.data,
                        &cache.values.data,
                        &cache.key_scales.data,
                        &cache.value_scales.data,
                        &mut key_dev,
                        &mut value_dev,
                        position_u32,
                        width_u32,
                        &cache.page_table,
                        page_tokens_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch Q8 KV cache dequantize kernel", err))?;

            Ok((
                CudaF32Buffer {
                    data: key_dev,
                    len: cache.width,
                },
                CudaF32Buffer {
                    data: value_dev,
                    len: cache.width,
                },
            ))
        }

        pub fn dequantize_key_q4_value_q8_layer_kv(
            &self,
            cache: &CudaKeyQ4ValueQ8LayerKvCache,
            position: usize,
        ) -> Result<(CudaF32Buffer, CudaF32Buffer)> {
            if position >= cache.len {
                return Err(XrtError::Runtime(format!(
                    "CUDA KQ4/VQ8 KV position {position} is out of range for len {}",
                    cache.len
                )));
            }
            if cache.width == 0 {
                return Ok((self.zeros_f32(0)?, self.zeros_f32(0)?));
            }

            let position_u32 = to_u32(position, "CUDA KQ4/VQ8 KV position")?;
            let width_u32 = to_u32(cache.width, "CUDA KQ4/VQ8 KV width")?;
            let page_tokens_u32 = to_u32(cache.page_tokens, "CUDA KQ4/VQ8 KV page tokens")?;
            let mut key_dev = self
                .device
                .alloc_zeros::<f32>(cache.width)
                .map_err(|err| cuda_error("failed to allocate KQ4/VQ8 KV key output", err))?;
            let mut value_dev = self
                .device
                .alloc_zeros::<f32>(cache.width)
                .map_err(|err| cuda_error("failed to allocate KQ4/VQ8 KV value output", err))?;
            let func =
                self.function(self.modules.attention, "kq4_vq8_kv_cache_dequantize_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(width_u32),
                    (
                        &cache.keys.data,
                        &cache.values.data,
                        &cache.key_scales.data,
                        &cache.value_scales.data,
                        &mut key_dev,
                        &mut value_dev,
                        position_u32,
                        width_u32,
                        &cache.page_table,
                        page_tokens_u32,
                    ),
                )
            }
            .map_err(|err| {
                cuda_error("failed to launch KQ4/VQ8 KV cache dequantize kernel", err)
            })?;

            Ok((
                CudaF32Buffer {
                    data: key_dev,
                    len: cache.width,
                },
                CudaF32Buffer {
                    data: value_dev,
                    len: cache.width,
                },
            ))
        }

        pub fn single_query_attention_q8_device(
            &self,
            query: &CudaF32Buffer,
            cache: &CudaQ8LayerKvCache,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
        ) -> Result<CudaF32Buffer> {
            self.single_query_attention_q8_windowed_device(
                query,
                cache,
                n_heads,
                n_kv_heads,
                head_dim,
                0,
                1.0f32 / (head_dim as f32).sqrt(),
            )
        }

        pub fn single_query_attention_q8_windowed_device(
            &self,
            query: &CudaF32Buffer,
            cache: &CudaQ8LayerKvCache,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            attend_start: usize,
            scale: f32,
        ) -> Result<CudaF32Buffer> {
            if cache.is_empty() {
                return Err(XrtError::Runtime(
                    "CUDA Q8 attention requires at least one KV cache entry".to_string(),
                ));
            }
            if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
                return Err(XrtError::Shape(format!(
                    "invalid attention head counts: heads={n_heads}, kv_heads={n_kv_heads}"
                )));
            }
            if attend_start >= cache.len {
                return Err(XrtError::Shape(format!(
                    "Q8 attention start {attend_start} must be less than cache length {}",
                    cache.len
                )));
            }
            if !scale.is_finite() || scale <= 0.0 {
                return Err(XrtError::Shape(format!(
                    "Q8 attention scale must be finite and positive, found {scale}"
                )));
            }

            let q_len = checked_mul(n_heads, head_dim, "Q8 attention query elements")?;
            let kv_width = checked_mul(n_kv_heads, head_dim, "Q8 attention KV width")?;
            expect_len(query.len(), q_len, "Q8 attention query")?;
            expect_len(cache.width, kv_width, "Q8 attention KV width")?;

            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(q_len)
                .map_err(|err| cuda_error("failed to allocate Q8 attention output", err))?;

            let n_heads_u32 = to_u32(n_heads, "Q8 attention head count")?;
            let n_kv_heads_u32 = to_u32(n_kv_heads, "Q8 attention KV head count")?;
            let head_dim_u32 = to_u32(head_dim, "Q8 attention head dimension")?;
            let cache_len_u32 = to_u32(cache.len, "Q8 attention cache length")?;
            let page_tokens_u32 = to_u32(cache.page_tokens, "CUDA Q8 KV page tokens")?;
            let attend_start_u32 = to_u32(attend_start, "Q8 attention start position")?;
            let output_len_u32 = to_u32(q_len, "Q8 attention output elements")?;
            let use_online_kernel = head_dim <= BLOCK_SIZE as usize;
            let kernel_name = if use_online_kernel {
                "single_query_attention_q8_online_kernel"
            } else {
                "single_query_attention_q8_kernel"
            };
            let launch = if use_online_kernel {
                row_launch(n_heads_u32)
            } else {
                one_dim_launch(output_len_u32)
            };
            let func = self.function(self.modules.attention, kernel_name)?;
            let mut params = vec![
                (&query.data).as_kernel_param(),
                (&cache.keys.data).as_kernel_param(),
                (&cache.values.data).as_kernel_param(),
                (&cache.key_scales.data).as_kernel_param(),
                (&cache.value_scales.data).as_kernel_param(),
                (&mut output_dev).as_kernel_param(),
                n_heads_u32.as_kernel_param(),
                n_kv_heads_u32.as_kernel_param(),
                head_dim_u32.as_kernel_param(),
                cache_len_u32.as_kernel_param(),
                scale.as_kernel_param(),
                (&cache.page_table).as_kernel_param(),
                page_tokens_u32.as_kernel_param(),
                attend_start_u32.as_kernel_param(),
            ];
            unsafe { func.launch(launch, &mut params) }.map_err(|err| {
                cuda_error(
                    &format!("failed to launch Q8 single-query attention kernel `{kernel_name}`"),
                    err,
                )
            })?;

            Ok(CudaF32Buffer {
                data: output_dev,
                len: q_len,
            })
        }

        pub fn single_query_attention_key_q4_value_q8_device(
            &self,
            query: &CudaF32Buffer,
            cache: &CudaKeyQ4ValueQ8LayerKvCache,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
        ) -> Result<CudaF32Buffer> {
            self.single_query_attention_key_q4_value_q8_windowed_device(
                query,
                cache,
                n_heads,
                n_kv_heads,
                head_dim,
                0,
                1.0f32 / (head_dim as f32).sqrt(),
            )
        }

        pub fn single_query_attention_key_q4_value_q8_windowed_device(
            &self,
            query: &CudaF32Buffer,
            cache: &CudaKeyQ4ValueQ8LayerKvCache,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            attend_start: usize,
            scale: f32,
        ) -> Result<CudaF32Buffer> {
            if cache.is_empty() {
                return Err(XrtError::Runtime(
                    "CUDA KQ4/VQ8 attention requires at least one KV cache entry".to_string(),
                ));
            }
            if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
                return Err(XrtError::Shape(format!(
                    "invalid attention head counts: heads={n_heads}, kv_heads={n_kv_heads}"
                )));
            }
            if attend_start >= cache.len {
                return Err(XrtError::Shape(format!(
                    "KQ4/VQ8 attention start {attend_start} must be less than cache length {}",
                    cache.len
                )));
            }
            if !scale.is_finite() || scale <= 0.0 {
                return Err(XrtError::Shape(format!(
                    "KQ4/VQ8 attention scale must be finite and positive, found {scale}"
                )));
            }
            let q_len = checked_mul(n_heads, head_dim, "KQ4/VQ8 attention query elements")?;
            let kv_width = checked_mul(n_kv_heads, head_dim, "KQ4/VQ8 attention KV width")?;
            expect_len(query.len(), q_len, "KQ4/VQ8 attention query")?;
            expect_len(cache.width, kv_width, "KQ4/VQ8 attention KV width")?;

            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(q_len)
                .map_err(|err| cuda_error("failed to allocate KQ4/VQ8 attention output", err))?;

            let n_heads_u32 = to_u32(n_heads, "KQ4/VQ8 attention head count")?;
            let n_kv_heads_u32 = to_u32(n_kv_heads, "KQ4/VQ8 attention KV head count")?;
            let head_dim_u32 = to_u32(head_dim, "KQ4/VQ8 attention head dimension")?;
            let cache_len_u32 = to_u32(cache.len, "KQ4/VQ8 attention cache length")?;
            let page_tokens_u32 = to_u32(cache.page_tokens, "CUDA KQ4/VQ8 KV page tokens")?;
            let attend_start_u32 = to_u32(attend_start, "KQ4/VQ8 attention start position")?;
            let output_len_u32 = to_u32(q_len, "KQ4/VQ8 attention output elements")?;
            let use_online_kernel = head_dim <= BLOCK_SIZE as usize;
            let kernel_name = if use_online_kernel {
                "single_query_attention_kq4_vq8_online_kernel"
            } else {
                "single_query_attention_kq4_vq8_kernel"
            };
            let launch = if use_online_kernel {
                row_launch(n_heads_u32)
            } else {
                one_dim_launch(output_len_u32)
            };
            let func = self.function(self.modules.attention, kernel_name)?;
            let mut params = vec![
                (&query.data).as_kernel_param(),
                (&cache.keys.data).as_kernel_param(),
                (&cache.values.data).as_kernel_param(),
                (&cache.key_scales.data).as_kernel_param(),
                (&cache.value_scales.data).as_kernel_param(),
                (&mut output_dev).as_kernel_param(),
                n_heads_u32.as_kernel_param(),
                n_kv_heads_u32.as_kernel_param(),
                head_dim_u32.as_kernel_param(),
                cache_len_u32.as_kernel_param(),
                scale.as_kernel_param(),
                (&cache.page_table).as_kernel_param(),
                page_tokens_u32.as_kernel_param(),
                attend_start_u32.as_kernel_param(),
            ];
            unsafe { func.launch(launch, &mut params) }.map_err(|err| {
                cuda_error(
                    &format!(
                        "failed to launch KQ4/VQ8 single-query attention kernel `{kernel_name}`"
                    ),
                    err,
                )
            })?;

            Ok(CudaF32Buffer {
                data: output_dev,
                len: q_len,
            })
        }

        pub fn single_query_attention_mixed_key_q4_value_q8_device(
            &self,
            query: &CudaF32Buffer,
            hot_cache: &CudaLayerKvCache,
            cold_cache: &CudaKeyQ4ValueQ8LayerKvCache,
            routes: &CudaAdaptiveKvRoutes,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
        ) -> Result<CudaF32Buffer> {
            self.single_query_attention_mixed_key_q4_value_q8_windowed_device(
                query,
                hot_cache,
                cold_cache,
                routes,
                n_heads,
                n_kv_heads,
                head_dim,
                0,
                1.0f32 / (head_dim as f32).sqrt(),
            )
        }

        pub fn single_query_attention_mixed_key_q4_value_q8_windowed_device(
            &self,
            query: &CudaF32Buffer,
            hot_cache: &CudaLayerKvCache,
            cold_cache: &CudaKeyQ4ValueQ8LayerKvCache,
            routes: &CudaAdaptiveKvRoutes,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            attend_start: usize,
            scale: f32,
        ) -> Result<CudaF32Buffer> {
            if routes.is_empty() {
                return Err(XrtError::Runtime(
                    "mixed CUDA attention requires at least one KV cache entry".to_string(),
                ));
            }
            let cache_len = hot_cache
                .len()
                .checked_add(cold_cache.len())
                .ok_or_else(|| {
                    XrtError::Runtime("mixed CUDA KV entry count overflow".to_string())
                })?;
            expect_len(routes.len(), cache_len, "mixed CUDA adaptive KV routes")?;
            if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
                return Err(XrtError::Shape(format!(
                    "invalid attention head counts: heads={n_heads}, kv_heads={n_kv_heads}"
                )));
            }
            if attend_start >= cache_len {
                return Err(XrtError::Shape(format!(
                    "mixed attention start {attend_start} must be less than cache length {cache_len}"
                )));
            }
            if !scale.is_finite() || scale <= 0.0 {
                return Err(XrtError::Shape(format!(
                    "mixed attention scale must be finite and positive, found {scale}"
                )));
            }

            let kv_width = checked_mul(n_kv_heads, head_dim, "mixed CUDA attention KV width")?;
            let q_len = checked_mul(n_heads, head_dim, "mixed CUDA attention query elements")?;
            expect_len(query.len(), q_len, "mixed CUDA attention query")?;
            expect_len(hot_cache.width(), kv_width, "mixed CUDA hot KV width")?;
            expect_len(cold_cache.width(), kv_width, "mixed CUDA cold KV width")?;

            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(q_len)
                .map_err(|err| cuda_error("failed to allocate mixed attention output", err))?;
            let n_heads_u32 = to_u32(n_heads, "mixed attention head count")?;
            let n_kv_heads_u32 = to_u32(n_kv_heads, "mixed attention KV head count")?;
            let head_dim_u32 = to_u32(head_dim, "mixed attention head dimension")?;
            let cache_len_u32 = to_u32(cache_len, "mixed attention cache length")?;
            let hot_page_tokens_u32 = to_u32(hot_cache.page_tokens, "mixed hot KV page tokens")?;
            let cold_page_tokens_u32 = to_u32(cold_cache.page_tokens, "mixed cold KV page tokens")?;
            let attend_start_u32 = to_u32(attend_start, "mixed attention start position")?;
            let output_len_u32 = to_u32(q_len, "mixed attention output elements")?;
            let func = self.function(
                self.modules.attention,
                "single_query_attention_mixed_kq4_vq8_kernel",
            )?;
            let mut params = vec![
                (&query.data).as_kernel_param(),
                (&hot_cache.keys.data).as_kernel_param(),
                (&hot_cache.values.data).as_kernel_param(),
                (&cold_cache.keys.data).as_kernel_param(),
                (&cold_cache.values.data).as_kernel_param(),
                (&cold_cache.key_scales.data).as_kernel_param(),
                (&cold_cache.value_scales.data).as_kernel_param(),
                (&mut output_dev).as_kernel_param(),
                (&routes.data).as_kernel_param(),
                (&hot_cache.page_table).as_kernel_param(),
                (&cold_cache.page_table).as_kernel_param(),
                n_heads_u32.as_kernel_param(),
                n_kv_heads_u32.as_kernel_param(),
                head_dim_u32.as_kernel_param(),
                cache_len_u32.as_kernel_param(),
                scale.as_kernel_param(),
                hot_page_tokens_u32.as_kernel_param(),
                cold_page_tokens_u32.as_kernel_param(),
                attend_start_u32.as_kernel_param(),
            ];
            unsafe { func.launch(one_dim_launch(output_len_u32), &mut params) }.map_err(|err| {
                cuda_error("failed to launch mixed single-query attention kernel", err)
            })?;

            Ok(CudaF32Buffer {
                data: output_dev,
                len: q_len,
            })
        }

        pub fn upload_f32_tensor(&self, gguf: &GgufFile, name: &str) -> Result<GpuF32Tensor> {
            let info = gguf.require_tensor(name)?;
            if !matches!(info.dtype, DType::F32 | DType::F16 | DType::BF16) {
                return Err(XrtError::Unsupported(format!(
                    "resident F32 tensor upload requires F32, F16, or BF16 dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            let values = decode_float_tensor_values(gguf, name, info.dtype, info.numel())?;
            Ok(GpuF32Tensor {
                name: name.to_string(),
                dimensions: info.dimensions.clone(),
                buffer: self.upload_f32(&values)?,
            })
        }

        pub fn upload_f32_tensor_transposed_2d(
            &self,
            gguf: &GgufFile,
            name: &str,
        ) -> Result<GpuF32Tensor> {
            let info = gguf.require_tensor(name)?;
            if !matches!(info.dtype, DType::F32 | DType::F16 | DType::BF16) {
                return Err(XrtError::Unsupported(format!(
                    "resident transposed F32 tensor upload requires F32, F16, or BF16 dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            if info.dimensions.len() != 2 {
                return Err(XrtError::Unsupported(format!(
                    "resident transposed F32 tensor upload requires a 2D tensor, tensor `{name}` has dimensions {:?}",
                    info.dimensions
                )));
            }

            let cols = info.row_len();
            let rows = info.rows();
            let element_count = checked_mul(rows, cols, "transposed F32 tensor elements")?;
            let values = decode_float_tensor_values(gguf, name, info.dtype, element_count)?;
            let mut transposed = vec![0.0f32; element_count];
            for row in 0..rows {
                let source_offset = row * cols;
                for col in 0..cols {
                    transposed[col * rows + row] = values[source_offset + col];
                }
            }

            Ok(GpuF32Tensor {
                name: format!("{name}:transposed"),
                dimensions: vec![rows, cols],
                buffer: self.upload_f32(&transposed)?,
            })
        }

        pub fn upload_q8_0_matrix(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ8_0Matrix> {
            let (scales, quants) = split_q8_0_matrix(matrix, rows, cols)?;
            Ok(CudaQ8_0Matrix {
                scales: self.upload_f32(&scales)?,
                quants: self.upload_bytes(&quants)?,
                rows,
                cols,
            })
        }

        pub fn upload_q8_0_tensor(&self, gguf: &GgufFile, name: &str) -> Result<CudaQ8_0Matrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q8_0 {
                return Err(XrtError::Unsupported(format!(
                    "resident Q8_0 tensor upload requires Q8_0 dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q8_0_matrix(gguf.tensor_data(name)?, info.rows(), info.row_len())
        }

        pub fn upload_q4_0_matrix(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ4_0Matrix> {
            let (scales, quants) = split_q4_0_matrix(matrix, rows, cols)?;
            Ok(CudaQ8_0Matrix {
                scales: self.upload_f32(&scales)?,
                quants: self.upload_bytes(&quants)?,
                rows,
                cols,
            })
        }

        pub fn upload_q4_0_tensor(&self, gguf: &GgufFile, name: &str) -> Result<CudaQ4_0Matrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q4_0 {
                return Err(XrtError::Unsupported(format!(
                    "resident Q4_0 tensor upload requires Q4_0 dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q4_0_matrix(gguf.tensor_data(name)?, info.rows(), info.row_len())
        }

        pub fn upload_q4_k_matrix(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ4KMatrix> {
            self.upload_q4_k_matrix_packed(matrix, rows, cols)
        }

        pub fn upload_q4_k_matrix_packed(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ4KMatrix> {
            let (d, dmin, scales, quants) = split_q4_k_matrix(matrix, rows, cols)?;
            Ok(CudaQ4KMatrix {
                storage: CudaKQuantMatrixStorage::Q4K {
                    d: self.upload_f32(&d)?,
                    dmin: self.upload_f32(&dmin)?,
                    scales: self.upload_bytes(&scales)?,
                    quants: self.upload_bytes(&quants)?,
                },
                rows,
                cols,
            })
        }

        pub fn upload_q4_k_embedding_matrix(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ4KMatrix> {
            let values_transposed = dequantize_q4_k_matrix_transposed(matrix, rows, cols)?;
            let values_row_major = transpose_row_major(&values_transposed, rows, cols)?;
            Ok(CudaQ4KMatrix {
                storage: CudaKQuantMatrixStorage::ExpandedF32 {
                    values_transposed: self.upload_f32(&values_transposed)?,
                    values_row_major: Some(self.upload_f32(&values_row_major)?),
                },
                rows,
                cols,
            })
        }

        pub fn upload_q4_k_tensor(&self, gguf: &GgufFile, name: &str) -> Result<CudaQ4KMatrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q4_K {
                return Err(XrtError::Unsupported(format!(
                    "resident Q4_K tensor upload requires Q4_K dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q4_k_matrix(gguf.tensor_data(name)?, info.rows(), info.row_len())
        }

        pub fn upload_q4_k_embedding_tensor(
            &self,
            gguf: &GgufFile,
            name: &str,
        ) -> Result<CudaQ4KMatrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q4_K {
                return Err(XrtError::Unsupported(format!(
                    "resident Q4_K embedding upload requires Q4_K dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q4_k_embedding_matrix(gguf.tensor_data(name)?, info.rows(), info.row_len())
        }

        pub fn upload_q5_k_matrix(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ5KMatrix> {
            self.upload_q5_k_matrix_with_embedding_rows(matrix, rows, cols, false)
        }

        pub fn upload_q5_k_embedding_matrix(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ5KMatrix> {
            self.upload_q5_k_matrix_with_embedding_rows(matrix, rows, cols, true)
        }

        fn upload_q5_k_matrix_with_embedding_rows(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
            include_row_major: bool,
        ) -> Result<CudaQ5KMatrix> {
            // ponytail: reuse resident F32 matmul until Q5_K has a proven faster CUDA kernel.
            let values_transposed = dequantize_q5_k_matrix_transposed(matrix, rows, cols)?;
            let values_row_major = if include_row_major {
                Some(transpose_row_major(&values_transposed, rows, cols)?)
            } else {
                None
            };
            Ok(CudaQ4KMatrix {
                storage: CudaKQuantMatrixStorage::ExpandedF32 {
                    values_transposed: self.upload_f32(&values_transposed)?,
                    values_row_major: values_row_major
                        .as_deref()
                        .map(|values| self.upload_f32(values))
                        .transpose()?,
                },
                rows,
                cols,
            })
        }

        pub fn upload_q5_k_tensor(&self, gguf: &GgufFile, name: &str) -> Result<CudaQ5KMatrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q5_K {
                return Err(XrtError::Unsupported(format!(
                    "resident Q5_K tensor upload requires Q5_K dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q5_k_matrix(gguf.tensor_data(name)?, info.rows(), info.row_len())
        }

        pub fn upload_q5_k_embedding_tensor(
            &self,
            gguf: &GgufFile,
            name: &str,
        ) -> Result<CudaQ5KMatrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q5_K {
                return Err(XrtError::Unsupported(format!(
                    "resident Q5_K embedding upload requires Q5_K dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q5_k_embedding_matrix(gguf.tensor_data(name)?, info.rows(), info.row_len())
        }

        pub fn upload_q6_k_matrix(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ6KMatrix> {
            self.upload_q6_k_matrix_with_embedding_rows(matrix, rows, cols, false)
        }

        pub fn upload_q6_k_embedding_matrix(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ6KMatrix> {
            self.upload_q6_k_matrix_with_embedding_rows(matrix, rows, cols, true)
        }

        pub fn upload_q6_k_embedding_matrix_packed(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
        ) -> Result<CudaQ6KMatrix> {
            let d = q6_k_block_scales(matrix, rows, cols)?;
            Ok(CudaQ4KMatrix {
                storage: CudaKQuantMatrixStorage::Q6K {
                    d: self.upload_f32(&d)?,
                    blocks: self.upload_bytes(matrix)?,
                },
                rows,
                cols,
            })
        }

        fn upload_q6_k_matrix_with_embedding_rows(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
            include_row_major: bool,
        ) -> Result<CudaQ6KMatrix> {
            // ponytail: packed Q6_K regressed VibeThinker; dequantize once until a faster kernel exists.
            let values_transposed = dequantize_q6_k_matrix_transposed(matrix, rows, cols)?;
            let values_row_major = if include_row_major {
                Some(transpose_row_major(&values_transposed, rows, cols)?)
            } else {
                None
            };
            Ok(CudaQ4KMatrix {
                storage: CudaKQuantMatrixStorage::ExpandedF32 {
                    values_transposed: self.upload_f32(&values_transposed)?,
                    values_row_major: values_row_major
                        .as_deref()
                        .map(|values| self.upload_f32(values))
                        .transpose()?,
                },
                rows,
                cols,
            })
        }

        pub fn upload_q6_k_tensor(&self, gguf: &GgufFile, name: &str) -> Result<CudaQ6KMatrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q6_K {
                return Err(XrtError::Unsupported(format!(
                    "resident Q6_K tensor upload requires Q6_K dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q6_k_matrix(gguf.tensor_data(name)?, info.rows(), info.row_len())
        }

        pub fn upload_q6_k_embedding_tensor(
            &self,
            gguf: &GgufFile,
            name: &str,
        ) -> Result<CudaQ6KMatrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q6_K {
                return Err(XrtError::Unsupported(format!(
                    "resident Q6_K embedding upload requires Q6_K dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q6_k_embedding_matrix(gguf.tensor_data(name)?, info.rows(), info.row_len())
        }

        pub fn upload_q6_k_embedding_tensor_packed(
            &self,
            gguf: &GgufFile,
            name: &str,
        ) -> Result<CudaQ6KMatrix> {
            let info = gguf.require_tensor(name)?;
            if info.dtype != DType::Q6_K {
                return Err(XrtError::Unsupported(format!(
                    "resident packed Q6_K embedding upload requires Q6_K dtype, tensor `{name}` is {:?}",
                    info.dtype
                )));
            }
            self.upload_q6_k_embedding_matrix_packed(
                gguf.tensor_data(name)?,
                info.rows(),
                info.row_len(),
            )
        }

        pub fn rmsnorm(
            &self,
            input: &[f32],
            weight: &[f32],
            rows: usize,
            cols: usize,
            eps: f32,
        ) -> Result<Vec<f32>> {
            let expected = checked_mul(rows, cols, "rmsnorm elements")?;
            expect_len(input.len(), expected, "rmsnorm input")?;
            expect_len(weight.len(), cols, "rmsnorm weight")?;
            if expected == 0 {
                return Ok(Vec::new());
            }

            let rows_u32 = to_u32(rows, "rmsnorm rows")?;
            let cols_u32 = to_u32(cols, "rmsnorm cols")?;
            let input_dev = self
                .device
                .htod_copy(input.to_vec())
                .map_err(|err| cuda_error("failed to copy rmsnorm input to device", err))?;
            let weight_dev = self
                .device
                .htod_copy(weight.to_vec())
                .map_err(|err| cuda_error("failed to copy rmsnorm weight to device", err))?;
            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(expected)
                .map_err(|err| cuda_error("failed to allocate rmsnorm output", err))?;

            let func = self.function(self.modules.rmsnorm, "rmsnorm_kernel")?;
            unsafe {
                func.launch(
                    row_launch(rows_u32),
                    (
                        &input_dev,
                        &weight_dev,
                        &mut output_dev,
                        rows_u32,
                        cols_u32,
                        eps,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch rmsnorm kernel", err))?;

            self.device
                .sync_reclaim(output_dev)
                .map_err(|err| cuda_error("failed to reclaim rmsnorm output", err))
        }

        pub fn rmsnorm_resident_weight(
            &self,
            input: &[f32],
            weight: &CudaF32Buffer,
            rows: usize,
            cols: usize,
            eps: f32,
        ) -> Result<Vec<f32>> {
            let input = self.upload_f32(input)?;
            self.rmsnorm_device(&input, weight, rows, cols, eps)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn rmsnorm_device(
            &self,
            input: &CudaF32Buffer,
            weight: &CudaF32Buffer,
            rows: usize,
            cols: usize,
            eps: f32,
        ) -> Result<CudaF32Buffer> {
            let expected = checked_mul(rows, cols, "rmsnorm elements")?;
            expect_len(input.len(), expected, "rmsnorm input")?;
            expect_len(weight.len(), cols, "rmsnorm resident weight")?;
            if expected == 0 {
                return self.zeros_f32(0);
            }

            let mut output = self.zeros_f32(expected)?;
            self.rmsnorm_device_into(input, weight, rows, cols, eps, &mut output)?;
            Ok(output)
        }

        pub fn rmsnorm_device_into(
            &self,
            input: &CudaF32Buffer,
            weight: &CudaF32Buffer,
            rows: usize,
            cols: usize,
            eps: f32,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            let expected = checked_mul(rows, cols, "rmsnorm elements")?;
            expect_len(input.len(), expected, "rmsnorm input")?;
            expect_len(weight.len(), cols, "rmsnorm resident weight")?;
            expect_len(output.len(), expected, "rmsnorm output")?;
            if expected == 0 {
                return Ok(());
            }

            let rows_u32 = to_u32(rows, "rmsnorm rows")?;
            let cols_u32 = to_u32(cols, "rmsnorm cols")?;

            let func = self.function(self.modules.rmsnorm, "rmsnorm_kernel")?;
            unsafe {
                func.launch(
                    row_launch(rows_u32),
                    (
                        &input.data,
                        &weight.data,
                        &mut output.data,
                        rows_u32,
                        cols_u32,
                        eps,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch rmsnorm kernel", err))?;
            Ok(())
        }

        pub fn rmsnorm_unweighted_device(
            &self,
            input: &CudaF32Buffer,
            rows: usize,
            cols: usize,
            eps: f32,
        ) -> Result<CudaF32Buffer> {
            let expected = checked_mul(rows, cols, "unweighted rmsnorm elements")?;
            expect_len(input.len(), expected, "unweighted rmsnorm input")?;
            if expected == 0 {
                return self.zeros_f32(0);
            }

            let mut output = self.zeros_f32(expected)?;
            self.rmsnorm_unweighted_device_into(input, rows, cols, eps, &mut output)?;
            Ok(output)
        }

        pub fn rmsnorm_unweighted_device_into(
            &self,
            input: &CudaF32Buffer,
            rows: usize,
            cols: usize,
            eps: f32,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            let expected = checked_mul(rows, cols, "unweighted rmsnorm elements")?;
            expect_len(input.len(), expected, "unweighted rmsnorm input")?;
            expect_len(output.len(), expected, "unweighted rmsnorm output")?;
            if expected == 0 {
                return Ok(());
            }

            let rows_u32 = to_u32(rows, "unweighted rmsnorm rows")?;
            let cols_u32 = to_u32(cols, "unweighted rmsnorm cols")?;
            let func = self.function(self.modules.rmsnorm, "rmsnorm_unweighted_kernel")?;
            unsafe {
                func.launch(
                    row_launch(rows_u32),
                    (&input.data, &mut output.data, rows_u32, cols_u32, eps),
                )
            }
            .map_err(|err| cuda_error("failed to launch unweighted rmsnorm kernel", err))?;
            Ok(())
        }

        pub fn rope(
            &self,
            tensor: &[f32],
            n_heads: usize,
            head_dim: usize,
            position: usize,
            rope_dim: usize,
            base: f32,
            scale: f32,
        ) -> Result<Vec<f32>> {
            let expected = checked_mul(n_heads, head_dim, "rope tensor elements")?;
            expect_len(tensor.len(), expected, "rope tensor")?;
            if expected == 0 {
                return Ok(Vec::new());
            }

            let mut tensor_dev = self.upload_f32(tensor)?;
            self.rope_device(
                &mut tensor_dev,
                n_heads,
                head_dim,
                position,
                rope_dim,
                base,
                scale,
            )?;
            self.download_f32(&tensor_dev)
        }

        pub fn rope_device(
            &self,
            tensor: &mut CudaF32Buffer,
            n_heads: usize,
            head_dim: usize,
            position: usize,
            rope_dim: usize,
            base: f32,
            scale: f32,
        ) -> Result<()> {
            let expected = checked_mul(n_heads, head_dim, "rope tensor elements")?;
            expect_len(tensor.len(), expected, "rope tensor")?;
            if expected == 0 {
                return Ok(());
            }

            let rotary_width = rope_dim.min(head_dim);
            let half_width = rotary_width / 2;
            if half_width == 0 {
                return Ok(());
            }

            let total_pairs = checked_mul(n_heads, half_width, "rope pair count")?;
            let n_heads_u32 = to_u32(n_heads, "rope head count")?;
            let head_dim_u32 = to_u32(head_dim, "rope head dimension")?;
            let position_u32 = to_u32(position, "rope position")?;
            let rotary_width_u32 = to_u32(rotary_width, "rope dimension")?;
            let total_pairs_u32 = to_u32(total_pairs, "rope work items")?;

            let func = self.function(self.modules.rope, "rope_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(total_pairs_u32),
                    (
                        &mut tensor.data,
                        n_heads_u32,
                        head_dim_u32,
                        position_u32,
                        rotary_width_u32,
                        base,
                        scale,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch rope kernel", err))?;

            Ok(())
        }

        pub fn softmax(&self, values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
            let expected = checked_mul(rows, cols, "softmax elements")?;
            expect_len(values.len(), expected, "softmax input")?;
            if expected == 0 {
                return Ok(values.to_vec());
            }

            let mut values_dev = self.upload_f32(values)?;
            self.softmax_device(&mut values_dev, rows, cols)?;
            self.download_f32(&values_dev)
        }

        pub fn softmax_device(
            &self,
            values: &mut CudaF32Buffer,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let expected = checked_mul(rows, cols, "softmax elements")?;
            expect_len(values.len(), expected, "softmax input")?;
            if expected == 0 {
                return Ok(());
            }

            let rows_u32 = to_u32(rows, "softmax rows")?;
            let cols_u32 = to_u32(cols, "softmax cols")?;
            let func = self.function(self.modules.softmax, "softmax_kernel")?;
            unsafe { func.launch(row_launch(rows_u32), (&mut values.data, rows_u32, cols_u32)) }
                .map_err(|err| cuda_error("failed to launch softmax kernel", err))?;
            Ok(())
        }

        pub fn silu(&self, values: &[f32]) -> Result<Vec<f32>> {
            let values = self.upload_f32(values)?;
            self.silu_device(&values)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn silu_device(&self, values: &CudaF32Buffer) -> Result<CudaF32Buffer> {
            if values.is_empty() {
                return self.zeros_f32(0);
            }

            let mut output = self.zeros_f32(values.len())?;
            self.silu_device_into(values, &mut output)?;
            Ok(output)
        }

        pub fn silu_device_into(
            &self,
            values: &CudaF32Buffer,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            expect_len(output.len(), values.len(), "silu output")?;
            self.copy_f32_device(values, output)?;
            self.silu_assign_device(output)
        }

        pub fn silu_assign_device(&self, values: &mut CudaF32Buffer) -> Result<()> {
            if values.is_empty() {
                return Ok(());
            }

            let n_u32 = to_u32(values.len(), "silu element count")?;
            let func = self.function(self.modules.silu, "silu_kernel")?;
            unsafe { func.launch(one_dim_launch(n_u32), (&mut values.data, n_u32)) }
                .map_err(|err| cuda_error("failed to launch silu kernel", err))?;
            Ok(())
        }

        pub fn matmul(
            &self,
            a: &[f32],
            m: usize,
            k: usize,
            b: &[f32],
            n: usize,
        ) -> Result<Vec<f32>> {
            let a_expected = checked_mul(m, k, "matmul lhs elements")?;
            let b_expected = checked_mul(k, n, "matmul rhs elements")?;
            let output_len = checked_mul(m, n, "matmul output elements")?;
            expect_len(a.len(), a_expected, "matmul lhs")?;
            expect_len(b.len(), b_expected, "matmul rhs")?;

            if output_len == 0 {
                return Ok(Vec::new());
            }
            if k == 0 {
                return Ok(vec![0.0; output_len]);
            }

            let m_u32 = to_u32(m, "matmul rows")?;
            let k_u32 = to_u32(k, "matmul depth")?;
            let n_u32 = to_u32(n, "matmul cols")?;
            let a_dev = self
                .device
                .htod_copy(a.to_vec())
                .map_err(|err| cuda_error("failed to copy matmul lhs to device", err))?;
            let b_dev = self
                .device
                .htod_copy(b.to_vec())
                .map_err(|err| cuda_error("failed to copy matmul rhs to device", err))?;
            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(output_len)
                .map_err(|err| cuda_error("failed to allocate matmul output", err))?;

            let func = self.function(self.modules.matmul, "matmul_kernel")?;
            unsafe {
                func.launch(
                    matmul_launch(m_u32, n_u32),
                    (&a_dev, &b_dev, &mut output_dev, m_u32, k_u32, n_u32),
                )
            }
            .map_err(|err| cuda_error("failed to launch matmul kernel", err))?;

            self.device
                .sync_reclaim(output_dev)
                .map_err(|err| cuda_error("failed to reclaim matmul output", err))
        }

        pub fn matmul_resident_rhs(
            &self,
            a: &[f32],
            m: usize,
            k: usize,
            b: &CudaF32Buffer,
            n: usize,
        ) -> Result<Vec<f32>> {
            let a = self.upload_f32(a)?;
            self.matmul_resident_rhs_device(&a, m, k, b, n)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn matmul_resident_rhs_device(
            &self,
            a: &CudaF32Buffer,
            m: usize,
            k: usize,
            b: &CudaF32Buffer,
            n: usize,
        ) -> Result<CudaF32Buffer> {
            let a_expected = checked_mul(m, k, "matmul lhs elements")?;
            let b_expected = checked_mul(k, n, "matmul rhs elements")?;
            let output_len = checked_mul(m, n, "matmul output elements")?;
            expect_len(a.len(), a_expected, "matmul lhs")?;
            expect_len(b.len(), b_expected, "matmul resident rhs")?;

            if output_len == 0 {
                return self.zeros_f32(0);
            }
            if k == 0 {
                return self.zeros_f32(output_len);
            }

            let mut output = self.zeros_f32(output_len)?;
            self.matmul_resident_rhs_device_into(a, m, k, b, n, &mut output)?;
            Ok(output)
        }

        pub fn matmul_resident_rhs_device_into(
            &self,
            a: &CudaF32Buffer,
            m: usize,
            k: usize,
            b: &CudaF32Buffer,
            n: usize,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            let a_expected = checked_mul(m, k, "matmul lhs elements")?;
            let b_expected = checked_mul(k, n, "matmul rhs elements")?;
            let output_len = checked_mul(m, n, "matmul output elements")?;
            expect_len(a.len(), a_expected, "matmul lhs")?;
            expect_len(b.len(), b_expected, "matmul resident rhs")?;
            expect_len(output.len(), output_len, "matmul output")?;
            if output_len == 0 {
                return Ok(());
            }
            if k == 0 {
                return self
                    .device
                    .memset_zeros(&mut output.data)
                    .map_err(|err| cuda_error("failed to zero matmul output", err));
            }
            let m_u32 = to_u32(m, "matmul rows")?;
            let k_u32 = to_u32(k, "matmul depth")?;
            let n_u32 = to_u32(n, "matmul cols")?;

            let func = self.function(self.modules.matmul, "matmul_kernel")?;
            unsafe {
                func.launch(
                    matmul_launch(m_u32, n_u32),
                    (&a.data, &b.data, &mut output.data, m_u32, k_u32, n_u32),
                )
            }
            .map_err(|err| cuda_error("failed to launch matmul kernel", err))?;
            Ok(())
        }

        pub fn matvec_q8_0(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let resident = self.upload_q8_0_matrix(matrix, rows, cols)?;
            self.matvec_q8_0_resident(&resident, input)
        }

        pub fn matvec_q8_0_resident(
            &self,
            matrix: &CudaQ8_0Matrix,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let input = self.upload_f32(input)?;
            self.matvec_q8_0_resident_device(matrix, &input)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn matvec_q8_0_resident_device(
            &self,
            matrix: &CudaQ8_0Matrix,
            input: &CudaF32Buffer,
        ) -> Result<CudaF32Buffer> {
            expect_len(input.len(), matrix.cols, "Q8_0 matvec input")?;
            if matrix.rows == 0 {
                return self.zeros_f32(0);
            }

            let mut output = self.zeros_f32(matrix.rows)?;
            self.matvec_q8_0_resident_device_into(matrix, input, &mut output)?;
            Ok(output)
        }

        pub fn matvec_q8_0_resident_device_into(
            &self,
            matrix: &CudaQ8_0Matrix,
            input: &CudaF32Buffer,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            expect_len(input.len(), matrix.cols, "Q8_0 matvec input")?;
            expect_len(output.len(), matrix.rows, "Q8_0 matvec output")?;
            if matrix.rows == 0 {
                return Ok(());
            }

            let rows_u32 = to_u32(matrix.rows, "Q8_0 matvec rows")?;
            let cols_u32 = to_u32(matrix.cols, "Q8_0 matvec cols")?;

            let func = self.function(self.modules.q8_0_matvec, "q8_0_matvec_kernel")?;
            unsafe {
                func.launch(
                    row_launch(rows_u32),
                    (
                        &matrix.scales.data,
                        &matrix.quants.data,
                        &input.data,
                        &mut output.data,
                        rows_u32,
                        cols_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch Q8_0 matvec kernel", err))?;
            Ok(())
        }

        pub fn matvec_q4_0(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let resident = self.upload_q4_0_matrix(matrix, rows, cols)?;
            self.matvec_q4_0_resident(&resident, input)
        }

        pub fn matvec_q4_0_resident(
            &self,
            matrix: &CudaQ4_0Matrix,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let input = self.upload_f32(input)?;
            self.matvec_q4_0_resident_device(matrix, &input)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn matvec_q4_0_resident_device(
            &self,
            matrix: &CudaQ4_0Matrix,
            input: &CudaF32Buffer,
        ) -> Result<CudaF32Buffer> {
            self.matvec_q8_0_resident_device(matrix, input)
        }

        pub fn matvec_q4_0_resident_device_into(
            &self,
            matrix: &CudaQ4_0Matrix,
            input: &CudaF32Buffer,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            self.matvec_q8_0_resident_device_into(matrix, input, output)
        }

        pub fn matvec_q4_k(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let resident = self.upload_q4_k_matrix(matrix, rows, cols)?;
            self.matvec_q4_k_resident(&resident, input)
        }

        pub fn matvec_q4_k_resident(
            &self,
            matrix: &CudaQ4KMatrix,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let input = self.upload_f32(input)?;
            self.matvec_q4_k_resident_device(matrix, &input)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn matvec_q4_k_resident_device(
            &self,
            matrix: &CudaQ4KMatrix,
            input: &CudaF32Buffer,
        ) -> Result<CudaF32Buffer> {
            let mut output = self.zeros_f32(matrix.rows)?;
            self.matvec_q4_k_resident_device_into(matrix, input, &mut output)?;
            Ok(output)
        }

        pub fn matvec_q4_k_resident_device_into(
            &self,
            matrix: &CudaQ4KMatrix,
            input: &CudaF32Buffer,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            expect_len(input.len(), matrix.cols, "Q4_K matvec input")?;
            expect_len(output.len(), matrix.rows, "Q4_K matvec output")?;
            if matrix.rows == 0 {
                return Ok(());
            }
            match &matrix.storage {
                CudaKQuantMatrixStorage::Q4K {
                    d,
                    dmin,
                    scales,
                    quants,
                } => {
                    let rows_u32 = to_u32(matrix.rows, "Q4_K matvec rows")?;
                    let cols_u32 = to_u32(matrix.cols, "Q4_K matvec cols")?;

                    let func = self.function(self.modules.q4_k_matvec, "q4_k_matvec_kernel")?;
                    unsafe {
                        func.launch(
                            row_launch(rows_u32),
                            (
                                &d.data,
                                &dmin.data,
                                &scales.data,
                                &quants.data,
                                &input.data,
                                &mut output.data,
                                rows_u32,
                                cols_u32,
                            ),
                        )
                    }
                    .map_err(|err| cuda_error("failed to launch Q4_K matvec kernel", err))?;
                    Ok(())
                }
                CudaKQuantMatrixStorage::Q6K { .. } => Err(XrtError::InvalidTensor(
                    "Q4_K matvec received packed Q6_K storage".to_string(),
                )),
                CudaKQuantMatrixStorage::ExpandedF32 {
                    values_transposed, ..
                } => self.matmul_resident_rhs_device_into(
                    input,
                    1,
                    matrix.cols,
                    values_transposed,
                    matrix.rows,
                    output,
                ),
            }
        }

        pub fn matvec_q5_k(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let resident = self.upload_q5_k_matrix(matrix, rows, cols)?;
            self.matvec_q5_k_resident(&resident, input)
        }

        pub fn matvec_q5_k_resident(
            &self,
            matrix: &CudaQ5KMatrix,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let input = self.upload_f32(input)?;
            self.matvec_q5_k_resident_device(matrix, &input)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn matvec_q5_k_resident_device(
            &self,
            matrix: &CudaQ5KMatrix,
            input: &CudaF32Buffer,
        ) -> Result<CudaF32Buffer> {
            self.matvec_q6_k_resident_device(matrix, input)
        }

        pub fn matvec_q5_k_resident_device_into(
            &self,
            matrix: &CudaQ5KMatrix,
            input: &CudaF32Buffer,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            self.matvec_q6_k_resident_device_into(matrix, input, output)
        }

        pub fn matvec_q6_k(
            &self,
            matrix: &[u8],
            rows: usize,
            cols: usize,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let resident = self.upload_q6_k_matrix(matrix, rows, cols)?;
            self.matvec_q6_k_resident(&resident, input)
        }

        pub fn matvec_q6_k_resident(
            &self,
            matrix: &CudaQ6KMatrix,
            input: &[f32],
        ) -> Result<Vec<f32>> {
            let input = self.upload_f32(input)?;
            self.matvec_q6_k_resident_device(matrix, &input)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn matvec_q6_k_resident_device(
            &self,
            matrix: &CudaQ6KMatrix,
            input: &CudaF32Buffer,
        ) -> Result<CudaF32Buffer> {
            let mut output = self.zeros_f32(matrix.rows)?;
            self.matvec_q6_k_resident_device_into(matrix, input, &mut output)?;
            Ok(output)
        }

        pub fn matvec_q6_k_resident_device_into(
            &self,
            matrix: &CudaQ6KMatrix,
            input: &CudaF32Buffer,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            expect_len(input.len(), matrix.cols, "Q6_K matvec input")?;
            expect_len(output.len(), matrix.rows, "Q6_K matvec output")?;
            if matrix.rows == 0 {
                return Ok(());
            }
            match &matrix.storage {
                CudaKQuantMatrixStorage::Q6K { d, blocks } => {
                    let rows_u32 = to_u32(matrix.rows, "Q6_K matvec rows")?;
                    let cols_u32 = to_u32(matrix.cols, "Q6_K matvec cols")?;
                    let func = self.function(self.modules.q6_k_matvec, "q6_k_matvec_kernel")?;
                    unsafe {
                        func.launch(
                            row_launch(rows_u32),
                            (
                                &d.data,
                                &blocks.data,
                                &input.data,
                                &mut output.data,
                                rows_u32,
                                cols_u32,
                            ),
                        )
                    }
                    .map_err(|err| cuda_error("failed to launch Q6_K matvec kernel", err))?;
                    Ok(())
                }
                CudaKQuantMatrixStorage::Q4K { .. } => Err(XrtError::InvalidTensor(
                    "Q6_K matvec received packed Q4_K storage".to_string(),
                )),
                CudaKQuantMatrixStorage::ExpandedF32 {
                    values_transposed, ..
                } => self.matmul_resident_rhs_device_into(
                    input,
                    1,
                    matrix.cols,
                    values_transposed,
                    matrix.rows,
                    output,
                ),
            }
        }

        pub fn add(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
            let lhs = self.upload_f32(lhs)?;
            let rhs = self.upload_f32(rhs)?;
            self.add_device(&lhs, &rhs)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn add_device(
            &self,
            lhs: &CudaF32Buffer,
            rhs: &CudaF32Buffer,
        ) -> Result<CudaF32Buffer> {
            if lhs.len() != rhs.len() {
                return Err(XrtError::Shape(format!(
                    "add inputs must have identical lengths, found {} and {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            if lhs.is_empty() {
                return self.zeros_f32(0);
            }

            let mut output = self.zeros_f32(lhs.len())?;
            self.add_device_into(lhs, rhs, &mut output)?;
            Ok(output)
        }

        pub fn add_device_into(
            &self,
            lhs: &CudaF32Buffer,
            rhs: &CudaF32Buffer,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            if lhs.len() != rhs.len() {
                return Err(XrtError::Shape(format!(
                    "add inputs must have identical lengths, found {} and {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            expect_len(output.len(), lhs.len(), "add output")?;
            self.copy_f32_device(lhs, output)?;
            self.add_assign_device(output, rhs)
        }

        pub fn add_assign_device(
            &self,
            lhs: &mut CudaF32Buffer,
            rhs: &CudaF32Buffer,
        ) -> Result<()> {
            if lhs.len() != rhs.len() {
                return Err(XrtError::Shape(format!(
                    "add inputs must have identical lengths, found {} and {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            if lhs.is_empty() {
                return Ok(());
            }

            let n_u32 = to_u32(lhs.len(), "add element count")?;

            let func = self.function(self.modules.add, "elementwise_add_kernel")?;
            unsafe { func.launch(one_dim_launch(n_u32), (&mut lhs.data, &rhs.data, n_u32)) }
                .map_err(|err| cuda_error("failed to launch add kernel", err))?;
            Ok(())
        }

        pub fn mul_device(
            &self,
            lhs: &CudaF32Buffer,
            rhs: &CudaF32Buffer,
        ) -> Result<CudaF32Buffer> {
            if lhs.len() != rhs.len() {
                return Err(XrtError::Shape(format!(
                    "mul inputs must have identical lengths, found {} and {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            if lhs.is_empty() {
                return self.zeros_f32(0);
            }

            let mut output = self.zeros_f32(lhs.len())?;
            self.mul_device_into(lhs, rhs, &mut output)?;
            Ok(output)
        }

        pub fn mul_device_into(
            &self,
            lhs: &CudaF32Buffer,
            rhs: &CudaF32Buffer,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            if lhs.len() != rhs.len() {
                return Err(XrtError::Shape(format!(
                    "mul inputs must have identical lengths, found {} and {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            expect_len(output.len(), lhs.len(), "mul output")?;
            self.copy_f32_device(lhs, output)?;
            self.mul_assign_device(output, rhs)
        }

        pub fn mul_assign_device(
            &self,
            lhs: &mut CudaF32Buffer,
            rhs: &CudaF32Buffer,
        ) -> Result<()> {
            if lhs.len() != rhs.len() {
                return Err(XrtError::Shape(format!(
                    "mul inputs must have identical lengths, found {} and {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            if lhs.is_empty() {
                return Ok(());
            }

            let n_u32 = to_u32(lhs.len(), "mul element count")?;

            let func = self.function(self.modules.mul, "elementwise_mul_kernel")?;
            unsafe { func.launch(one_dim_launch(n_u32), (&mut lhs.data, &rhs.data, n_u32)) }
                .map_err(|err| cuda_error("failed to launch mul kernel", err))?;
            Ok(())
        }

        pub fn scale_assign_device(&self, values: &mut CudaF32Buffer, scale: f32) -> Result<()> {
            if values.is_empty() {
                return Ok(());
            }
            if !scale.is_finite() {
                return Err(XrtError::Model(format!(
                    "CUDA scalar multiplier must be finite, found {scale}"
                )));
            }

            let n_u32 = to_u32(values.len(), "scale element count")?;
            let func = self.function(self.modules.activation, "scale_assign_kernel")?;
            unsafe { func.launch(one_dim_launch(n_u32), (&mut values.data, n_u32, scale)) }
                .map_err(|err| cuda_error("failed to launch scale kernel", err))?;
            Ok(())
        }

        pub fn geglu_pytorch_tanh_assign_device(
            &self,
            gate: &mut CudaF32Buffer,
            up: &CudaF32Buffer,
        ) -> Result<()> {
            if gate.len() != up.len() {
                return Err(XrtError::Shape(format!(
                    "GeGLU inputs must have identical lengths, found {} and {}",
                    gate.len(),
                    up.len()
                )));
            }
            if gate.is_empty() {
                return Ok(());
            }

            let n_u32 = to_u32(gate.len(), "GeGLU element count")?;
            let func = self.function(self.modules.activation, "geglu_pytorch_tanh_kernel")?;
            unsafe { func.launch(one_dim_launch(n_u32), (&mut gate.data, &up.data, n_u32)) }
                .map_err(|err| cuda_error("failed to launch PyTorch tanh GeGLU kernel", err))?;
            Ok(())
        }

        pub fn logit_softcap_assign_device(
            &self,
            values: &mut CudaF32Buffer,
            softcap: f32,
        ) -> Result<()> {
            if values.is_empty() {
                return Ok(());
            }
            if !softcap.is_finite() || softcap <= 0.0 {
                return Err(XrtError::Model(format!(
                    "CUDA logit softcap must be finite and positive, found {softcap}"
                )));
            }

            let n_u32 = to_u32(values.len(), "logit softcap element count")?;
            let func = self.function(self.modules.activation, "logit_softcap_assign_kernel")?;
            unsafe { func.launch(one_dim_launch(n_u32), (&mut values.data, n_u32, softcap)) }
                .map_err(|err| cuda_error("failed to launch logit softcap kernel", err))?;
            Ok(())
        }

        pub fn repeat_kv_for_gqa_device(
            &self,
            values: &CudaF32Buffer,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
        ) -> Result<CudaF32Buffer> {
            if n_heads == 0 || n_kv_heads == 0 {
                expect_len(values.len(), 0, "repeat-kv input")?;
                return self.zeros_f32(0);
            }
            if n_heads % n_kv_heads != 0 {
                return Err(XrtError::Shape(format!(
                    "attention head count {n_heads} is not divisible by KV head count {n_kv_heads}"
                )));
            }

            let input_len = checked_mul(n_kv_heads, head_dim, "repeat-kv input elements")?;
            let output_len = checked_mul(n_heads, head_dim, "repeat-kv output elements")?;
            expect_len(values.len(), input_len, "repeat-kv input")?;
            if output_len == 0 {
                return self.zeros_f32(0);
            }

            let n_heads_u32 = to_u32(n_heads, "repeat-kv head count")?;
            let n_kv_heads_u32 = to_u32(n_kv_heads, "repeat-kv KV head count")?;
            let head_dim_u32 = to_u32(head_dim, "repeat-kv head dimension")?;
            let output_len_u32 = to_u32(output_len, "repeat-kv output elements")?;

            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(output_len)
                .map_err(|err| cuda_error("failed to allocate repeat-kv output", err))?;
            let func = self.function(self.modules.repeat_kv, "repeat_kv_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(output_len_u32),
                    (
                        &values.data,
                        &mut output_dev,
                        n_heads_u32,
                        n_kv_heads_u32,
                        head_dim_u32,
                        output_len_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch repeat-kv kernel", err))?;

            Ok(CudaF32Buffer {
                data: output_dev,
                len: output_len,
            })
        }

        pub fn single_query_attention_device(
            &self,
            query: &CudaF32Buffer,
            cache: &CudaLayerKvCache,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
        ) -> Result<CudaF32Buffer> {
            self.single_query_attention_windowed_device(
                query,
                cache,
                n_heads,
                n_kv_heads,
                head_dim,
                0,
                1.0f32 / (head_dim as f32).sqrt(),
            )
        }

        pub fn single_query_attention_windowed_device(
            &self,
            query: &CudaF32Buffer,
            cache: &CudaLayerKvCache,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            attend_start: usize,
            scale: f32,
        ) -> Result<CudaF32Buffer> {
            if cache.is_empty() {
                return Err(XrtError::Runtime(
                    "CUDA attention requires at least one KV cache entry".to_string(),
                ));
            }
            if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
                return Err(XrtError::Shape(format!(
                    "invalid attention head counts: heads={n_heads}, kv_heads={n_kv_heads}"
                )));
            }
            if attend_start >= cache.len {
                return Err(XrtError::Shape(format!(
                    "attention start {attend_start} must be less than cache length {}",
                    cache.len
                )));
            }
            if !scale.is_finite() || scale <= 0.0 {
                return Err(XrtError::Shape(format!(
                    "attention scale must be finite and positive, found {scale}"
                )));
            }

            let q_len = checked_mul(n_heads, head_dim, "attention query elements")?;
            let kv_width = checked_mul(n_kv_heads, head_dim, "attention KV width")?;
            expect_len(query.len(), q_len, "attention query")?;
            expect_len(cache.width, kv_width, "attention KV width")?;

            let output_len = q_len;
            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(output_len)
                .map_err(|err| cuda_error("failed to allocate attention output", err))?;

            let n_heads_u32 = to_u32(n_heads, "attention head count")?;
            let n_kv_heads_u32 = to_u32(n_kv_heads, "attention KV head count")?;
            let head_dim_u32 = to_u32(head_dim, "attention head dimension")?;
            let cache_len_u32 = to_u32(cache.len, "attention cache length")?;
            let kv_width_u32 = to_u32(cache.width, "attention KV width")?;
            let output_len_u32 = to_u32(output_len, "attention output elements")?;
            let page_tokens_u32 = to_u32(cache.page_tokens, "CUDA KV page tokens")?;
            let attend_start_u32 = to_u32(attend_start, "attention start position")?;

            let use_online_kernel = head_dim <= BLOCK_SIZE as usize;
            let kernel_name = if use_online_kernel {
                "single_query_attention_online_kernel"
            } else {
                "single_query_attention_kernel"
            };
            let launch = if use_online_kernel {
                row_launch(n_heads_u32)
            } else {
                one_dim_launch(output_len_u32)
            };
            let func = self.function(self.modules.attention, kernel_name)?;
            // cudarc 0.12 provides typed launch tuples through 12 arguments. The
            // page table and window start extend this kernel ABI to 14, so construct the documented
            // raw parameter-pointer list explicitly.
            let mut params = vec![
                (&query.data).as_kernel_param(),
                (&cache.keys.data).as_kernel_param(),
                (&cache.values.data).as_kernel_param(),
                (&mut output_dev).as_kernel_param(),
                n_heads_u32.as_kernel_param(),
                n_kv_heads_u32.as_kernel_param(),
                head_dim_u32.as_kernel_param(),
                cache_len_u32.as_kernel_param(),
                kv_width_u32.as_kernel_param(),
                output_len_u32.as_kernel_param(),
                scale.as_kernel_param(),
                (&cache.page_table).as_kernel_param(),
                page_tokens_u32.as_kernel_param(),
                attend_start_u32.as_kernel_param(),
            ];
            unsafe { func.launch(launch, &mut params) }.map_err(|err| {
                cuda_error(
                    &format!(
                        "failed to launch paged single-query attention kernel `{kernel_name}`"
                    ),
                    err,
                )
            })?;

            Ok(CudaF32Buffer {
                data: output_dev,
                len: output_len,
            })
        }

        pub fn embed(
            &self,
            table: &[f32],
            vocab_size: usize,
            hidden_dim: usize,
            token_ids: &[u32],
        ) -> Result<Vec<f32>> {
            let table_expected = checked_mul(vocab_size, hidden_dim, "embedding table elements")?;
            expect_len(table.len(), table_expected, "embedding table")?;
            let output_len = checked_mul(token_ids.len(), hidden_dim, "embedding output elements")?;
            if output_len == 0 {
                return Ok(Vec::new());
            }

            if let Some(token) = token_ids
                .iter()
                .copied()
                .find(|token| (*token as usize) >= vocab_size)
            {
                return Err(XrtError::Model(format!(
                    "token id {token} exceeds embedding rows {vocab_size}"
                )));
            }

            let num_tokens_u32 = to_u32(token_ids.len(), "embedding token count")?;
            let hidden_dim_u32 = to_u32(hidden_dim, "embedding width")?;
            let vocab_size_u32 = to_u32(vocab_size, "embedding vocab size")?;
            let output_len_u32 = to_u32(output_len, "embedding output elements")?;

            let table_dev = self
                .device
                .htod_copy(table.to_vec())
                .map_err(|err| cuda_error("failed to copy embedding table to device", err))?;
            let token_dev = self
                .device
                .htod_copy(token_ids.to_vec())
                .map_err(|err| cuda_error("failed to copy token ids to device", err))?;
            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(output_len)
                .map_err(|err| cuda_error("failed to allocate embedding output", err))?;

            let func = self.function(self.modules.embed, "embedding_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(output_len_u32),
                    (
                        &table_dev,
                        &token_dev,
                        &mut output_dev,
                        num_tokens_u32,
                        hidden_dim_u32,
                        vocab_size_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch embedding kernel", err))?;

            self.device
                .sync_reclaim(output_dev)
                .map_err(|err| cuda_error("failed to reclaim embedding output", err))
        }

        pub fn embed_resident(
            &self,
            table: &CudaF32Buffer,
            vocab_size: usize,
            hidden_dim: usize,
            token_ids: &[u32],
        ) -> Result<Vec<f32>> {
            self.embed_resident_device(table, vocab_size, hidden_dim, token_ids)
                .and_then(|output| self.download_f32(&output))
        }

        pub fn embed_resident_device(
            &self,
            table: &CudaF32Buffer,
            vocab_size: usize,
            hidden_dim: usize,
            token_ids: &[u32],
        ) -> Result<CudaF32Buffer> {
            let table_expected = checked_mul(vocab_size, hidden_dim, "embedding table elements")?;
            expect_len(table.len(), table_expected, "embedding resident table")?;
            let output_len = checked_mul(token_ids.len(), hidden_dim, "embedding output elements")?;
            if output_len == 0 {
                return self.zeros_f32(0);
            }

            if let Some(token) = token_ids
                .iter()
                .copied()
                .find(|token| (*token as usize) >= vocab_size)
            {
                return Err(XrtError::Model(format!(
                    "token id {token} exceeds embedding rows {vocab_size}"
                )));
            }

            let num_tokens_u32 = to_u32(token_ids.len(), "embedding token count")?;
            let hidden_dim_u32 = to_u32(hidden_dim, "embedding width")?;
            let vocab_size_u32 = to_u32(vocab_size, "embedding vocab size")?;
            let output_len_u32 = to_u32(output_len, "embedding output elements")?;

            let token_dev = self
                .device
                .htod_copy(token_ids.to_vec())
                .map_err(|err| cuda_error("failed to copy token ids to device", err))?;
            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(output_len)
                .map_err(|err| cuda_error("failed to allocate embedding output", err))?;

            let func = self.function(self.modules.embed, "embedding_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(output_len_u32),
                    (
                        &table.data,
                        &token_dev,
                        &mut output_dev,
                        num_tokens_u32,
                        hidden_dim_u32,
                        vocab_size_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch embedding kernel", err))?;

            Ok(CudaF32Buffer {
                data: output_dev,
                len: output_len,
            })
        }

        pub fn embed_q8_0_resident_device(
            &self,
            table: &CudaQ8_0Matrix,
            token_ids: &[u32],
        ) -> Result<CudaF32Buffer> {
            let vocab_size = table.rows;
            let hidden_dim = table.cols;
            let output_len = checked_mul(
                token_ids.len(),
                hidden_dim,
                "Q8_0 embedding output elements",
            )?;
            if output_len == 0 {
                return self.zeros_f32(0);
            }
            if let Some(token) = token_ids
                .iter()
                .copied()
                .find(|token| (*token as usize) >= vocab_size)
            {
                return Err(XrtError::Model(format!(
                    "token id {token} exceeds embedding rows {vocab_size}"
                )));
            }

            let num_tokens_u32 = to_u32(token_ids.len(), "Q8_0 embedding token count")?;
            let hidden_dim_u32 = to_u32(hidden_dim, "Q8_0 embedding width")?;
            let vocab_size_u32 = to_u32(vocab_size, "Q8_0 embedding vocab size")?;
            let output_len_u32 = to_u32(output_len, "Q8_0 embedding output elements")?;
            let token_dev = self.device.htod_copy(token_ids.to_vec()).map_err(|err| {
                cuda_error("failed to copy Q8_0 embedding token ids to device", err)
            })?;
            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(output_len)
                .map_err(|err| cuda_error("failed to allocate Q8_0 embedding output", err))?;

            let func = self.function(self.modules.embed, "q8_0_embedding_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(output_len_u32),
                    (
                        &table.scales.data,
                        &table.quants.data,
                        &token_dev,
                        &mut output_dev,
                        num_tokens_u32,
                        hidden_dim_u32,
                        vocab_size_u32,
                    ),
                )
            }
            .map_err(|err| cuda_error("failed to launch Q8_0 embedding kernel", err))?;

            Ok(CudaF32Buffer {
                data: output_dev,
                len: output_len,
            })
        }

        pub fn embed_q4_k_resident_device(
            &self,
            table: &CudaQ4KMatrix,
            token_ids: &[u32],
        ) -> Result<CudaF32Buffer> {
            let vocab_size = table.rows;
            let hidden_dim = table.cols;
            let output_len = checked_mul(
                token_ids.len(),
                hidden_dim,
                "Q4_K embedding output elements",
            )?;
            if output_len == 0 {
                return self.zeros_f32(0);
            }
            if let Some(token) = token_ids
                .iter()
                .copied()
                .find(|token| (*token as usize) >= vocab_size)
            {
                return Err(XrtError::Model(format!(
                    "token id {token} exceeds embedding rows {vocab_size}"
                )));
            }

            let num_tokens_u32 = to_u32(token_ids.len(), "Q4_K embedding token count")?;
            let hidden_dim_u32 = to_u32(hidden_dim, "Q4_K embedding width")?;
            let vocab_size_u32 = to_u32(vocab_size, "Q4_K embedding vocab size")?;
            let output_len_u32 = to_u32(output_len, "Q4_K embedding output elements")?;
            let token_dev = self.device.htod_copy(token_ids.to_vec()).map_err(|err| {
                cuda_error("failed to copy Q4_K embedding token ids to device", err)
            })?;
            let mut output_dev = self
                .device
                .alloc_zeros::<f32>(output_len)
                .map_err(|err| cuda_error("failed to allocate Q4_K embedding output", err))?;

            match &table.storage {
                CudaKQuantMatrixStorage::Q4K {
                    d,
                    dmin,
                    scales,
                    quants,
                } => {
                    let func = self.function(self.modules.embed, "q4_k_packed_embedding_kernel")?;
                    unsafe {
                        func.launch(
                            one_dim_launch(output_len_u32),
                            (
                                &d.data,
                                &dmin.data,
                                &scales.data,
                                &quants.data,
                                &token_dev,
                                &mut output_dev,
                                num_tokens_u32,
                                hidden_dim_u32,
                                vocab_size_u32,
                            ),
                        )
                    }
                    .map_err(|err| {
                        cuda_error("failed to launch packed Q4_K embedding kernel", err)
                    })?;
                }
                CudaKQuantMatrixStorage::Q6K { d, blocks } => {
                    let func = self.function(self.modules.embed, "q6_k_packed_embedding_kernel")?;
                    unsafe {
                        func.launch(
                            one_dim_launch(output_len_u32),
                            (
                                &d.data,
                                &blocks.data,
                                &token_dev,
                                &mut output_dev,
                                num_tokens_u32,
                                hidden_dim_u32,
                                vocab_size_u32,
                            ),
                        )
                    }
                    .map_err(|err| {
                        cuda_error("failed to launch packed Q6_K embedding kernel", err)
                    })?;
                }
                CudaKQuantMatrixStorage::ExpandedF32 {
                    values_row_major, ..
                } => {
                    let values_row_major = values_row_major.as_ref().ok_or_else(|| {
                        XrtError::InvalidTensor(
                            "expanded K-quant embedding requires row-major values".to_string(),
                        )
                    })?;
                    let func = self.function(self.modules.embed, "q4_k_embedding_kernel")?;
                    unsafe {
                        func.launch(
                            one_dim_launch(output_len_u32),
                            (
                                &values_row_major.data,
                                &token_dev,
                                &mut output_dev,
                                num_tokens_u32,
                                hidden_dim_u32,
                                vocab_size_u32,
                            ),
                        )
                    }
                    .map_err(|err| {
                        cuda_error("failed to launch expanded K-quant embedding kernel", err)
                    })?;
                }
            }

            Ok(CudaF32Buffer {
                data: output_dev,
                len: output_len,
            })
        }

        pub fn embed_q6_k_resident_device(
            &self,
            table: &CudaQ6KMatrix,
            token_ids: &[u32],
        ) -> Result<CudaF32Buffer> {
            self.embed_q4_k_resident_device(table, token_ids)
        }

        pub fn embed_q5_k_resident_device(
            &self,
            table: &CudaQ5KMatrix,
            token_ids: &[u32],
        ) -> Result<CudaF32Buffer> {
            self.embed_q4_k_resident_device(table, token_ids)
        }

        fn function(&self, module_name: &str, function_name: &str) -> Result<CudaFunction> {
            if let Some(function) = self.device.get_func(module_name, function_name) {
                return Ok(function);
            }
            self.load_module_for_function(module_name)?;
            self.device
                .get_func(module_name, function_name)
                .ok_or_else(|| {
                    XrtError::Cuda(format!(
                        "failed to fetch kernel `{function_name}` from module `{module_name}`"
                    ))
                })
        }

        fn load_module_for_function(&self, module_name: &str) -> Result<()> {
            if module_name == self.modules.rmsnorm {
                load_module(
                    &self.device,
                    MODULES.rmsnorm,
                    RMSNORM_PTX,
                    &["rmsnorm_kernel", "rmsnorm_unweighted_kernel"],
                )
            } else if module_name == self.modules.rope {
                load_module(&self.device, MODULES.rope, ROPE_PTX, &["rope_kernel"])
            } else if module_name == self.modules.softmax {
                load_module(
                    &self.device,
                    MODULES.softmax,
                    SOFTMAX_PTX,
                    &["softmax_kernel"],
                )
            } else if module_name == self.modules.silu {
                load_module(&self.device, MODULES.silu, SILU_PTX, &["silu_kernel"])
            } else if module_name == self.modules.matmul {
                load_module(&self.device, MODULES.matmul, MATMUL_PTX, &["matmul_kernel"])
            } else if module_name == self.modules.q8_0_matvec {
                load_module(
                    &self.device,
                    MODULES.q8_0_matvec,
                    Q8_0_MATVEC_PTX,
                    &["q8_0_matvec_kernel"],
                )
            } else if module_name == self.modules.q4_k_matvec {
                load_module(
                    &self.device,
                    MODULES.q4_k_matvec,
                    Q4_K_MATVEC_PTX,
                    &["q4_k_matvec_kernel"],
                )
            } else if module_name == self.modules.q6_k_matvec {
                load_module(
                    &self.device,
                    MODULES.q6_k_matvec,
                    Q6_K_MATVEC_PTX,
                    &["q6_k_matvec_kernel"],
                )
            } else if module_name == self.modules.add {
                load_module(
                    &self.device,
                    MODULES.add,
                    ADD_PTX,
                    &["elementwise_add_kernel"],
                )
            } else if module_name == self.modules.mul {
                load_module(
                    &self.device,
                    MODULES.mul,
                    MUL_PTX,
                    &["elementwise_mul_kernel"],
                )
            } else if module_name == self.modules.activation {
                load_module(
                    &self.device,
                    MODULES.activation,
                    ACTIVATION_PTX,
                    &[
                        "scale_assign_kernel",
                        "geglu_pytorch_tanh_kernel",
                        "logit_softcap_assign_kernel",
                    ],
                )
            } else if module_name == self.modules.repeat_kv {
                load_module(
                    &self.device,
                    MODULES.repeat_kv,
                    REPEAT_KV_PTX,
                    &["repeat_kv_kernel"],
                )
            } else if module_name == self.modules.attention {
                load_module(
                    &self.device,
                    MODULES.attention,
                    ATTENTION_PTX,
                    &[
                        "adaptive_kv_route_write_kernel",
                        "kv_cache_append_kernel",
                        "paged_kv_cache_append_kernel",
                        "paged_kv_cache_gather_kernel",
                        "q8_kv_cache_append_kernel",
                        "q8_kv_cache_dequantize_kernel",
                        "kq4_vq8_kv_cache_append_kernel",
                        "kq4_vq8_kv_cache_copy_row_kernel",
                        "kq4_vq8_kv_cache_dequantize_kernel",
                        "attention_scores_kernel",
                        "attention_values_kernel",
                        "single_query_attention_online_kernel",
                        "single_query_attention_kernel",
                        "single_query_attention_q8_online_kernel",
                        "single_query_attention_q8_kernel",
                        "single_query_attention_kq4_vq8_online_kernel",
                        "single_query_attention_kq4_vq8_kernel",
                        "single_query_attention_mixed_kq4_vq8_kernel",
                    ],
                )
            } else if module_name == self.modules.embed {
                load_module(
                    &self.device,
                    MODULES.embed,
                    EMBEDDING_PTX,
                    &[
                        "embedding_kernel",
                        "q8_0_embedding_kernel",
                        "q4_k_embedding_kernel",
                        "q4_k_packed_embedding_kernel",
                        "q6_k_packed_embedding_kernel",
                    ],
                )
            } else {
                Err(XrtError::Cuda(format!(
                    "unknown CUDA module `{module_name}` requested"
                )))
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{
    CudaAdaptiveKvRoutes, CudaBackend, CudaBytes, CudaDevice, CudaF32Buffer,
    CudaKeyQ4ValueQ8LayerKvCache, CudaLayerKvCache, CudaQ4KMatrix, CudaQ4_0Matrix, CudaQ5KMatrix,
    CudaQ6KMatrix, CudaQ8LayerKvCache, CudaQ8_0Matrix, GpuF32Tensor, GpuModelWeights, GpuTensor,
};

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaDevice;

#[cfg(not(feature = "cuda"))]
pub type CudaBackend = CudaDevice;

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaBytes {
    len: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaBytes {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaF32Buffer {
    len: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaF32Buffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone)]
pub struct GpuTensor {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub dtype: DType,
    pub byte_len: usize,
    buffer: CudaBytes,
}

#[cfg(not(feature = "cuda"))]
impl GpuTensor {
    pub fn buffer(&self) -> &CudaBytes {
        &self.buffer
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone)]
pub struct GpuF32Tensor {
    pub name: String,
    pub dimensions: Vec<usize>,
    buffer: CudaF32Buffer,
}

#[cfg(not(feature = "cuda"))]
impl GpuF32Tensor {
    pub fn buffer(&self) -> &CudaF32Buffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn byte_len(&self) -> usize {
        self.buffer.byte_len()
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Default)]
pub struct CudaQ8_0Matrix {
    scales: CudaF32Buffer,
    quants: CudaBytes,
    rows: usize,
    cols: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaQ8_0Matrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn scale_count(&self) -> usize {
        self.scales.len()
    }

    pub fn quant_byte_len(&self) -> usize {
        self.quants.byte_len()
    }
}

#[cfg(not(feature = "cuda"))]
pub type CudaQ4_0Matrix = CudaQ8_0Matrix;

#[cfg(not(feature = "cuda"))]
pub type CudaQ5KMatrix = CudaQ4KMatrix;

#[cfg(not(feature = "cuda"))]
pub type CudaQ6KMatrix = CudaQ4KMatrix;

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Default)]
pub struct CudaQ4KMatrix {
    rows: usize,
    cols: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaQ4KMatrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn byte_len(&self) -> usize {
        0
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Default)]
pub struct CudaAdaptiveKvRoutes {
    capacity: usize,
    len: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaAdaptiveKvRoutes {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allocated_bytes(&self) -> u64 {
        0
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.len = self.len.min(new_len);
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Default)]
pub struct CudaLayerKvCache {
    capacity: usize,
    len: usize,
    width: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaLayerKvCache {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn page_tokens(&self) -> usize {
        self.capacity.max(1)
    }

    pub fn page_count(&self) -> usize {
        usize::from(self.capacity != 0)
    }

    pub fn allocated_bytes(&self) -> u64 {
        0
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.len = self.len.min(new_len);
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Default)]
pub struct CudaQ8LayerKvCache {
    capacity: usize,
    len: usize,
    width: usize,
    page_tokens: usize,
    page_count: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaQ8LayerKvCache {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens.max(1)
    }

    pub fn page_count(&self) -> usize {
        self.page_count
    }

    pub fn allocated_bytes(&self) -> u64 {
        0
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.len = self.len.min(new_len);
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Default)]
pub struct CudaKeyQ4ValueQ8LayerKvCache {
    capacity: usize,
    len: usize,
    width: usize,
    page_tokens: usize,
    page_count: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaKeyQ4ValueQ8LayerKvCache {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens.max(1)
    }

    pub fn page_count(&self) -> usize {
        self.page_count
    }

    pub fn allocated_bytes(&self) -> u64 {
        0
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.len = self.len.min(new_len);
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Default)]
pub struct GpuModelWeights {
    tensors: Vec<GpuTensor>,
    total_bytes: u64,
}

#[cfg(not(feature = "cuda"))]
impl GpuModelWeights {
    pub fn from_gguf(_device: &CudaDevice, _gguf: &GgufFile) -> Result<Self> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn tensors(&self) -> &[GpuTensor] {
        &self.tensors
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

#[cfg(not(feature = "cuda"))]
impl CudaDevice {
    pub fn new(_ordinal: usize) -> Result<Self> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn name(&self) -> Result<String> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn memory_info(&self) -> Result<(u64, u64)> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_bytes(&self, _bytes: &[u8]) -> Result<CudaBytes> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_f32(&self, _values: &[f32]) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_f32_into(&self, _values: &[f32], _destination: &mut CudaF32Buffer) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn copy_f32_device(
        &self,
        _source: &CudaF32Buffer,
        _destination: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn download_f32(&self, _buffer: &CudaF32Buffer) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn zeros_f32(&self, _len: usize) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn zeros_bytes(&self, _len: usize) -> Result<CudaBytes> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn alloc_adaptive_kv_routes(&self, _capacity: usize) -> Result<CudaAdaptiveKvRoutes> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn grow_adaptive_kv_routes(
        &self,
        _routes: &mut CudaAdaptiveKvRoutes,
        _new_capacity: usize,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn replace_adaptive_kv_routes(
        &self,
        _routes: &mut CudaAdaptiveKvRoutes,
        _hot_mask: &[u8],
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn append_adaptive_kv_route(
        &self,
        _routes: &mut CudaAdaptiveKvRoutes,
        _is_hot: bool,
        _local_position: usize,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn alloc_layer_kv_cache(
        &self,
        _capacity: usize,
        _width: usize,
    ) -> Result<CudaLayerKvCache> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn alloc_paged_layer_kv_cache(
        &self,
        _capacity: usize,
        _width: usize,
        _page_tokens: usize,
    ) -> Result<CudaLayerKvCache> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn remap_paged_layer_kv_pages(
        &self,
        _cache: &mut CudaLayerKvCache,
        _page_map: &[u32],
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn alloc_q8_layer_kv_cache(
        &self,
        _capacity: usize,
        _width: usize,
    ) -> Result<CudaQ8LayerKvCache> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn alloc_paged_q8_layer_kv_cache(
        &self,
        _capacity: usize,
        _width: usize,
        _page_tokens: usize,
    ) -> Result<CudaQ8LayerKvCache> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn remap_paged_q8_layer_kv_pages(
        &self,
        _cache: &mut CudaQ8LayerKvCache,
        _page_map: &[u32],
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn alloc_key_q4_value_q8_layer_kv_cache(
        &self,
        _capacity: usize,
        _width: usize,
    ) -> Result<CudaKeyQ4ValueQ8LayerKvCache> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn alloc_paged_key_q4_value_q8_layer_kv_cache(
        &self,
        _capacity: usize,
        _width: usize,
        _page_tokens: usize,
    ) -> Result<CudaKeyQ4ValueQ8LayerKvCache> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn remap_paged_key_q4_value_q8_layer_kv_pages(
        &self,
        _cache: &mut CudaKeyQ4ValueQ8LayerKvCache,
        _page_map: &[u32],
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn grow_layer_kv_cache(
        &self,
        _cache: &mut CudaLayerKvCache,
        _new_capacity: usize,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn grow_q8_layer_kv_cache(
        &self,
        _cache: &mut CudaQ8LayerKvCache,
        _new_capacity: usize,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn grow_key_q4_value_q8_layer_kv_cache(
        &self,
        _cache: &mut CudaKeyQ4ValueQ8LayerKvCache,
        _new_capacity: usize,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn append_layer_kv(
        &self,
        _cache: &mut CudaLayerKvCache,
        _key: &CudaF32Buffer,
        _value: &CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn gather_paged_layer_kv(
        &self,
        _cache: &CudaLayerKvCache,
        _start_position: usize,
        _count: usize,
    ) -> Result<(CudaF32Buffer, CudaF32Buffer)> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn copy_layer_kv(
        &self,
        _cache: &CudaLayerKvCache,
        _position: usize,
    ) -> Result<(CudaF32Buffer, CudaF32Buffer)> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn append_q8_layer_kv(
        &self,
        _cache: &mut CudaQ8LayerKvCache,
        _key: &CudaF32Buffer,
        _value: &CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn append_key_q4_value_q8_layer_kv(
        &self,
        _cache: &mut CudaKeyQ4ValueQ8LayerKvCache,
        _key: &CudaF32Buffer,
        _value: &CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn copy_key_q4_value_q8_layer_kv_row(
        &self,
        _source: &CudaKeyQ4ValueQ8LayerKvCache,
        _source_position: usize,
        _destination: &mut CudaKeyQ4ValueQ8LayerKvCache,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn dequantize_q8_layer_kv(
        &self,
        _cache: &CudaQ8LayerKvCache,
        _position: usize,
    ) -> Result<(CudaF32Buffer, CudaF32Buffer)> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn dequantize_key_q4_value_q8_layer_kv(
        &self,
        _cache: &CudaKeyQ4ValueQ8LayerKvCache,
        _position: usize,
    ) -> Result<(CudaF32Buffer, CudaF32Buffer)> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_f32_tensor(&self, _gguf: &GgufFile, _name: &str) -> Result<GpuF32Tensor> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_f32_tensor_transposed_2d(
        &self,
        _gguf: &GgufFile,
        _name: &str,
    ) -> Result<GpuF32Tensor> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q8_0_matrix(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ8_0Matrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q8_0_tensor(&self, _gguf: &GgufFile, _name: &str) -> Result<CudaQ8_0Matrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q4_0_matrix(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ4_0Matrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q4_0_tensor(&self, _gguf: &GgufFile, _name: &str) -> Result<CudaQ4_0Matrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q4_k_matrix(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ4KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q4_k_matrix_packed(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ4KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q4_k_embedding_matrix(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ4KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q4_k_tensor(&self, _gguf: &GgufFile, _name: &str) -> Result<CudaQ4KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q4_k_embedding_tensor(
        &self,
        _gguf: &GgufFile,
        _name: &str,
    ) -> Result<CudaQ4KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q5_k_matrix(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ5KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q5_k_embedding_matrix(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ5KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q5_k_tensor(&self, _gguf: &GgufFile, _name: &str) -> Result<CudaQ5KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q5_k_embedding_tensor(
        &self,
        _gguf: &GgufFile,
        _name: &str,
    ) -> Result<CudaQ5KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q6_k_matrix(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ6KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q6_k_embedding_matrix(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ6KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q6_k_embedding_matrix_packed(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
    ) -> Result<CudaQ6KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q6_k_tensor(&self, _gguf: &GgufFile, _name: &str) -> Result<CudaQ6KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q6_k_embedding_tensor(
        &self,
        _gguf: &GgufFile,
        _name: &str,
    ) -> Result<CudaQ6KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn upload_q6_k_embedding_tensor_packed(
        &self,
        _gguf: &GgufFile,
        _name: &str,
    ) -> Result<CudaQ6KMatrix> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn rmsnorm(
        &self,
        _input: &[f32],
        _weight: &[f32],
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn rmsnorm_resident_weight(
        &self,
        _input: &[f32],
        _weight: &CudaF32Buffer,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn rmsnorm_device(
        &self,
        _input: &CudaF32Buffer,
        _weight: &CudaF32Buffer,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn rmsnorm_device_into(
        &self,
        _input: &CudaF32Buffer,
        _weight: &CudaF32Buffer,
        _rows: usize,
        _cols: usize,
        _eps: f32,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn rmsnorm_unweighted_device(
        &self,
        _input: &CudaF32Buffer,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn rmsnorm_unweighted_device_into(
        &self,
        _input: &CudaF32Buffer,
        _rows: usize,
        _cols: usize,
        _eps: f32,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn rope(
        &self,
        _tensor: &[f32],
        _n_heads: usize,
        _head_dim: usize,
        _position: usize,
        _rope_dim: usize,
        _base: f32,
        _scale: f32,
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn rope_device(
        &self,
        _tensor: &mut CudaF32Buffer,
        _n_heads: usize,
        _head_dim: usize,
        _position: usize,
        _rope_dim: usize,
        _base: f32,
        _scale: f32,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn softmax(&self, _values: &[f32], _rows: usize, _cols: usize) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn softmax_device(
        &self,
        _values: &mut CudaF32Buffer,
        _rows: usize,
        _cols: usize,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn silu(&self, _values: &[f32]) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn silu_device(&self, _values: &CudaF32Buffer) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn silu_device_into(
        &self,
        _values: &CudaF32Buffer,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn silu_assign_device(&self, _values: &mut CudaF32Buffer) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matmul(
        &self,
        _a: &[f32],
        _m: usize,
        _k: usize,
        _b: &[f32],
        _n: usize,
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matmul_resident_rhs(
        &self,
        _a: &[f32],
        _m: usize,
        _k: usize,
        _b: &CudaF32Buffer,
        _n: usize,
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matmul_resident_rhs_device(
        &self,
        _a: &CudaF32Buffer,
        _m: usize,
        _k: usize,
        _b: &CudaF32Buffer,
        _n: usize,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matmul_resident_rhs_device_into(
        &self,
        _a: &CudaF32Buffer,
        _m: usize,
        _k: usize,
        _b: &CudaF32Buffer,
        _n: usize,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q8_0(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q8_0_resident(
        &self,
        _matrix: &CudaQ8_0Matrix,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q8_0_resident_device(
        &self,
        _matrix: &CudaQ8_0Matrix,
        _input: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q8_0_resident_device_into(
        &self,
        _matrix: &CudaQ8_0Matrix,
        _input: &CudaF32Buffer,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q4_0(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q4_0_resident(
        &self,
        _matrix: &CudaQ4_0Matrix,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q4_0_resident_device(
        &self,
        _matrix: &CudaQ4_0Matrix,
        _input: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q4_0_resident_device_into(
        &self,
        _matrix: &CudaQ4_0Matrix,
        _input: &CudaF32Buffer,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q4_k(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q4_k_resident(
        &self,
        _matrix: &CudaQ4KMatrix,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q4_k_resident_device(
        &self,
        _matrix: &CudaQ4KMatrix,
        _input: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q4_k_resident_device_into(
        &self,
        _matrix: &CudaQ4KMatrix,
        _input: &CudaF32Buffer,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q5_k(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q5_k_resident(
        &self,
        _matrix: &CudaQ5KMatrix,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q5_k_resident_device(
        &self,
        _matrix: &CudaQ5KMatrix,
        _input: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q5_k_resident_device_into(
        &self,
        _matrix: &CudaQ5KMatrix,
        _input: &CudaF32Buffer,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q6_k(
        &self,
        _matrix: &[u8],
        _rows: usize,
        _cols: usize,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q6_k_resident(
        &self,
        _matrix: &CudaQ6KMatrix,
        _input: &[f32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q6_k_resident_device(
        &self,
        _matrix: &CudaQ6KMatrix,
        _input: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn matvec_q6_k_resident_device_into(
        &self,
        _matrix: &CudaQ6KMatrix,
        _input: &CudaF32Buffer,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn add(&self, _lhs: &[f32], _rhs: &[f32]) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn add_device(&self, _lhs: &CudaF32Buffer, _rhs: &CudaF32Buffer) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn add_device_into(
        &self,
        _lhs: &CudaF32Buffer,
        _rhs: &CudaF32Buffer,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn add_assign_device(&self, _lhs: &mut CudaF32Buffer, _rhs: &CudaF32Buffer) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn mul_device(&self, _lhs: &CudaF32Buffer, _rhs: &CudaF32Buffer) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn mul_device_into(
        &self,
        _lhs: &CudaF32Buffer,
        _rhs: &CudaF32Buffer,
        _output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn mul_assign_device(&self, _lhs: &mut CudaF32Buffer, _rhs: &CudaF32Buffer) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn scale_assign_device(&self, _values: &mut CudaF32Buffer, _scale: f32) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn geglu_pytorch_tanh_assign_device(
        &self,
        _gate: &mut CudaF32Buffer,
        _up: &CudaF32Buffer,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn logit_softcap_assign_device(
        &self,
        _values: &mut CudaF32Buffer,
        _softcap: f32,
    ) -> Result<()> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn repeat_kv_for_gqa_device(
        &self,
        _values: &CudaF32Buffer,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn single_query_attention_device(
        &self,
        _query: &CudaF32Buffer,
        _cache: &CudaLayerKvCache,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn single_query_attention_windowed_device(
        &self,
        _query: &CudaF32Buffer,
        _cache: &CudaLayerKvCache,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _attend_start: usize,
        _scale: f32,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn single_query_attention_q8_device(
        &self,
        _query: &CudaF32Buffer,
        _cache: &CudaQ8LayerKvCache,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn single_query_attention_q8_windowed_device(
        &self,
        _query: &CudaF32Buffer,
        _cache: &CudaQ8LayerKvCache,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _attend_start: usize,
        _scale: f32,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn single_query_attention_key_q4_value_q8_device(
        &self,
        _query: &CudaF32Buffer,
        _cache: &CudaKeyQ4ValueQ8LayerKvCache,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn single_query_attention_key_q4_value_q8_windowed_device(
        &self,
        _query: &CudaF32Buffer,
        _cache: &CudaKeyQ4ValueQ8LayerKvCache,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _attend_start: usize,
        _scale: f32,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn single_query_attention_mixed_key_q4_value_q8_device(
        &self,
        _query: &CudaF32Buffer,
        _hot_cache: &CudaLayerKvCache,
        _cold_cache: &CudaKeyQ4ValueQ8LayerKvCache,
        _routes: &CudaAdaptiveKvRoutes,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn single_query_attention_mixed_key_q4_value_q8_windowed_device(
        &self,
        _query: &CudaF32Buffer,
        _hot_cache: &CudaLayerKvCache,
        _cold_cache: &CudaKeyQ4ValueQ8LayerKvCache,
        _routes: &CudaAdaptiveKvRoutes,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _attend_start: usize,
        _scale: f32,
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn embed(
        &self,
        _table: &[f32],
        _vocab_size: usize,
        _hidden_dim: usize,
        _token_ids: &[u32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn embed_resident(
        &self,
        _table: &CudaF32Buffer,
        _vocab_size: usize,
        _hidden_dim: usize,
        _token_ids: &[u32],
    ) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn embed_resident_device(
        &self,
        _table: &CudaF32Buffer,
        _vocab_size: usize,
        _hidden_dim: usize,
        _token_ids: &[u32],
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn embed_q8_0_resident_device(
        &self,
        _table: &CudaQ8_0Matrix,
        _token_ids: &[u32],
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn embed_q4_k_resident_device(
        &self,
        _table: &CudaQ4KMatrix,
        _token_ids: &[u32],
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn embed_q5_k_resident_device(
        &self,
        _table: &CudaQ5KMatrix,
        _token_ids: &[u32],
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn embed_q6_k_resident_device(
        &self,
        _table: &CudaQ6KMatrix,
        _token_ids: &[u32],
    ) -> Result<CudaF32Buffer> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    fn assert_cuda_disabled<T>(result: Result<T>) {
        match result {
            Err(XrtError::Cuda(message)) => assert_eq!(message, CUDA_DISABLED_MESSAGE),
            Err(err) => panic!("expected CUDA-disabled error, got {err}"),
            Ok(_) => panic!("expected CUDA-disabled error, got Ok"),
        }
    }

    #[test]
    fn resident_api_stubs_fail_clearly_without_cuda_feature() {
        let device = CudaDevice;
        let buffer = CudaF32Buffer { len: 4 };

        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.byte_len(), 16);
        assert_cuda_disabled(CudaDevice::new(0));
        assert_cuda_disabled(device.name());
        assert_cuda_disabled(device.memory_info());
        assert_cuda_disabled(device.download_f32(&buffer));
        let mut mutable_buffer = buffer;
        assert_cuda_disabled(device.upload_f32_into(&[0.0; 4], &mut mutable_buffer));
        assert_cuda_disabled(device.copy_f32_device(&buffer, &mut mutable_buffer));
        assert_cuda_disabled(device.zeros_bytes(4));
        assert_cuda_disabled(device.alloc_layer_kv_cache(1, 4));
        assert_cuda_disabled(device.alloc_paged_layer_kv_cache(2, 4, 1));
        assert_cuda_disabled(device.alloc_q8_layer_kv_cache(1, 4));
        assert_cuda_disabled(device.alloc_paged_q8_layer_kv_cache(2, 4, 1));
        assert_cuda_disabled(device.alloc_key_q4_value_q8_layer_kv_cache(1, 4));
        assert_cuda_disabled(device.alloc_paged_key_q4_value_q8_layer_kv_cache(2, 4, 1));
        assert_cuda_disabled(device.alloc_adaptive_kv_routes(2));
        let mut routes = CudaAdaptiveKvRoutes {
            capacity: 2,
            len: 1,
        };
        assert_eq!(routes.capacity(), 2);
        assert_eq!(routes.len(), 1);
        assert!(!routes.is_empty());
        assert_eq!(routes.allocated_bytes(), 0);
        routes.truncate(1);
        routes.clear();
        assert!(routes.is_empty());
        assert_cuda_disabled(device.grow_adaptive_kv_routes(&mut routes, 4));
        assert_cuda_disabled(device.replace_adaptive_kv_routes(&mut routes, &[1, 0]));
        assert_cuda_disabled(device.append_adaptive_kv_route(&mut routes, true, 0));
        let mut cache = CudaLayerKvCache {
            capacity: 1,
            len: 0,
            width: 4,
        };
        assert_eq!(cache.capacity(), 1);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.width(), 4);
        assert_cuda_disabled(device.grow_layer_kv_cache(&mut cache, 2));
        assert_cuda_disabled(device.append_layer_kv(&mut cache, &buffer, &buffer));
        assert_cuda_disabled(device.copy_layer_kv(&cache, 0));
        let mut q8_cache = CudaQ8LayerKvCache {
            capacity: 1,
            len: 0,
            width: 4,
            page_tokens: 1,
            page_count: 1,
        };
        assert_eq!(q8_cache.capacity(), 1);
        assert_eq!(q8_cache.len(), 0);
        assert!(q8_cache.is_empty());
        assert_eq!(q8_cache.width(), 4);
        assert_eq!(q8_cache.page_tokens(), 1);
        assert_eq!(q8_cache.page_count(), 1);
        assert_eq!(q8_cache.allocated_bytes(), 0);
        q8_cache.truncate(0);
        q8_cache.clear();
        assert_cuda_disabled(device.grow_q8_layer_kv_cache(&mut q8_cache, 2));
        assert_cuda_disabled(device.remap_paged_q8_layer_kv_pages(&mut q8_cache, &[0]));
        assert_cuda_disabled(device.append_q8_layer_kv(&mut q8_cache, &buffer, &buffer));
        assert_cuda_disabled(device.dequantize_q8_layer_kv(&q8_cache, 0));
        let mut kq4_vq8_cache = CudaKeyQ4ValueQ8LayerKvCache {
            capacity: 1,
            len: 0,
            width: 4,
            page_tokens: 1,
            page_count: 1,
        };
        assert_eq!(kq4_vq8_cache.capacity(), 1);
        assert_eq!(kq4_vq8_cache.len(), 0);
        assert!(kq4_vq8_cache.is_empty());
        assert_eq!(kq4_vq8_cache.width(), 4);
        assert_eq!(kq4_vq8_cache.page_tokens(), 1);
        assert_eq!(kq4_vq8_cache.page_count(), 1);
        assert_eq!(kq4_vq8_cache.allocated_bytes(), 0);
        kq4_vq8_cache.truncate(0);
        kq4_vq8_cache.clear();
        assert_cuda_disabled(device.grow_key_q4_value_q8_layer_kv_cache(&mut kq4_vq8_cache, 2));
        assert_cuda_disabled(
            device.remap_paged_key_q4_value_q8_layer_kv_pages(&mut kq4_vq8_cache, &[0]),
        );
        assert_cuda_disabled(device.append_key_q4_value_q8_layer_kv(
            &mut kq4_vq8_cache,
            &buffer,
            &buffer,
        ));
        let mut kq4_vq8_destination = CudaKeyQ4ValueQ8LayerKvCache {
            capacity: 1,
            len: 0,
            width: 4,
            page_tokens: 1,
            page_count: 1,
        };
        assert_cuda_disabled(device.copy_key_q4_value_q8_layer_kv_row(
            &kq4_vq8_cache,
            0,
            &mut kq4_vq8_destination,
        ));
        assert_cuda_disabled(device.dequantize_key_q4_value_q8_layer_kv(&kq4_vq8_cache, 0));
        assert_cuda_disabled(device.upload_q8_0_matrix(&[], 0, 32));
        assert_cuda_disabled(device.rmsnorm_resident_weight(
            &[1.0, 2.0, 3.0, 4.0],
            &buffer,
            1,
            4,
            1e-5,
        ));
        assert_cuda_disabled(device.rmsnorm_device(&buffer, &buffer, 1, 4, 1e-5));
        assert_cuda_disabled(device.rmsnorm_device_into(
            &buffer,
            &buffer,
            1,
            4,
            1e-5,
            &mut mutable_buffer,
        ));
        assert_cuda_disabled(device.rmsnorm_unweighted_device(&buffer, 1, 4, 1e-5));
        assert_cuda_disabled(device.rmsnorm_unweighted_device_into(
            &buffer,
            1,
            4,
            1e-5,
            &mut mutable_buffer,
        ));
        assert_cuda_disabled(device.matmul_resident_rhs(&[1.0, 2.0], 1, 2, &buffer, 2));
        assert_cuda_disabled(device.matmul_resident_rhs_device(&buffer, 1, 4, &buffer, 1));
        assert_cuda_disabled(device.matmul_resident_rhs_device_into(
            &buffer,
            1,
            4,
            &buffer,
            1,
            &mut mutable_buffer,
        ));
        assert_cuda_disabled(device.embed_resident(&buffer, 2, 2, &[0, 1]));
        assert_cuda_disabled(device.embed_resident_device(&buffer, 2, 2, &[0, 1]));
        assert_cuda_disabled(device.add_device(&buffer, &buffer));
        assert_cuda_disabled(device.add_device_into(&buffer, &buffer, &mut mutable_buffer));
        assert_cuda_disabled(device.add_assign_device(&mut mutable_buffer, &buffer));
        assert_cuda_disabled(device.rope_device(&mut mutable_buffer, 1, 4, 0, 4, 10000.0, 1.0));
        assert_cuda_disabled(device.softmax_device(&mut mutable_buffer, 1, 4));
        assert_cuda_disabled(device.silu_device(&buffer));
        assert_cuda_disabled(device.silu_device_into(&buffer, &mut mutable_buffer));
        assert_cuda_disabled(device.silu_assign_device(&mut mutable_buffer));
        assert_cuda_disabled(device.mul_device(&buffer, &buffer));
        assert_cuda_disabled(device.mul_device_into(&buffer, &buffer, &mut mutable_buffer));
        assert_cuda_disabled(device.mul_assign_device(&mut mutable_buffer, &buffer));
        assert_cuda_disabled(device.scale_assign_device(&mut mutable_buffer, 2.0));
        assert_cuda_disabled(device.geglu_pytorch_tanh_assign_device(&mut mutable_buffer, &buffer));
        assert_cuda_disabled(device.logit_softcap_assign_device(&mut mutable_buffer, 30.0));
        assert_cuda_disabled(device.repeat_kv_for_gqa_device(&buffer, 2, 1, 4));
        assert_cuda_disabled(device.single_query_attention_device(&buffer, &cache, 2, 1, 2));
        assert_cuda_disabled(
            device.single_query_attention_windowed_device(&buffer, &cache, 2, 1, 2, 0, 1.0),
        );
        assert_cuda_disabled(device.single_query_attention_q8_device(&buffer, &q8_cache, 2, 1, 2));
        assert_cuda_disabled(
            device.single_query_attention_q8_windowed_device(&buffer, &q8_cache, 2, 1, 2, 0, 1.0),
        );
        assert_cuda_disabled(device.single_query_attention_key_q4_value_q8_device(
            &buffer,
            &kq4_vq8_cache,
            2,
            1,
            2,
        ));
        assert_cuda_disabled(
            device.single_query_attention_key_q4_value_q8_windowed_device(
                &buffer,
                &kq4_vq8_cache,
                2,
                1,
                2,
                0,
                1.0,
            ),
        );
        assert_cuda_disabled(device.single_query_attention_mixed_key_q4_value_q8_device(
            &buffer,
            &cache,
            &kq4_vq8_cache,
            &routes,
            2,
            1,
            2,
        ));
        assert_cuda_disabled(
            device.single_query_attention_mixed_key_q4_value_q8_windowed_device(
                &buffer,
                &cache,
                &kq4_vq8_cache,
                &routes,
                2,
                1,
                2,
                0,
                1.0,
            ),
        );

        let q8 = CudaQ8_0Matrix {
            scales: CudaF32Buffer { len: 1 },
            quants: CudaBytes { len: 32 },
            rows: 1,
            cols: 32,
        };
        assert_eq!(q8.rows(), 1);
        assert_eq!(q8.cols(), 32);
        assert_eq!(q8.scale_count(), 1);
        assert_eq!(q8.quant_byte_len(), 32);
        assert_cuda_disabled(device.matvec_q8_0_resident(&q8, &[0.0; 32]));
        assert_cuda_disabled(device.matvec_q8_0_resident_device(&q8, &buffer));
        assert_cuda_disabled(device.matvec_q8_0_resident_device_into(
            &q8,
            &buffer,
            &mut mutable_buffer,
        ));
        assert_cuda_disabled(device.embed_q8_0_resident_device(&q8, &[0]));
        assert_cuda_disabled(device.upload_q4_0_matrix(&[], 0, 32));
        assert_cuda_disabled(device.matvec_q4_0_resident(&q8, &[0.0; 32]));
        assert_cuda_disabled(device.matvec_q4_0_resident_device(&q8, &buffer));
        assert_cuda_disabled(device.matvec_q4_0_resident_device_into(
            &q8,
            &buffer,
            &mut mutable_buffer,
        ));
        let q4k = CudaQ4KMatrix { rows: 1, cols: 256 };
        assert_eq!(q4k.rows(), 1);
        assert_eq!(q4k.cols(), 256);
        assert_eq!(q4k.byte_len(), 0);
        assert_cuda_disabled(device.upload_q4_k_matrix(&[], 0, 256));
        assert_cuda_disabled(device.upload_q4_k_matrix_packed(&[], 0, 256));
        assert_cuda_disabled(device.upload_q4_k_embedding_matrix(&[], 0, 256));
        assert_cuda_disabled(device.matvec_q4_k_resident(&q4k, &[0.0; 256]));
        assert_cuda_disabled(device.matvec_q4_k_resident_device(&q4k, &buffer));
        assert_cuda_disabled(device.matvec_q4_k_resident_device_into(
            &q4k,
            &buffer,
            &mut mutable_buffer,
        ));
        assert_cuda_disabled(device.embed_q4_k_resident_device(&q4k, &[0]));
        assert_cuda_disabled(device.upload_q5_k_matrix(&[], 0, 256));
        assert_cuda_disabled(device.upload_q5_k_embedding_matrix(&[], 0, 256));
        assert_cuda_disabled(device.matvec_q5_k_resident(&q4k, &[0.0; 256]));
        assert_cuda_disabled(device.matvec_q5_k_resident_device(&q4k, &buffer));
        assert_cuda_disabled(device.matvec_q5_k_resident_device_into(
            &q4k,
            &buffer,
            &mut mutable_buffer,
        ));
        assert_cuda_disabled(device.embed_q5_k_resident_device(&q4k, &[0]));
        assert_cuda_disabled(device.upload_q6_k_matrix(&[], 0, 256));
        assert_cuda_disabled(device.upload_q6_k_embedding_matrix(&[], 0, 256));
        assert_cuda_disabled(device.upload_q6_k_embedding_matrix_packed(&[], 0, 256));
        assert_cuda_disabled(device.matvec_q6_k_resident(&q4k, &[0.0; 256]));
        assert_cuda_disabled(device.matvec_q6_k_resident_device(&q4k, &buffer));
        assert_cuda_disabled(device.matvec_q6_k_resident_device_into(
            &q4k,
            &buffer,
            &mut mutable_buffer,
        ));
        assert_cuda_disabled(device.embed_q6_k_resident_device(&q4k, &[0]));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod allocation_tests {
    #[test]
    fn q8_kv_allocated_bytes_formula_is_smaller_than_f32() {
        let capacity = 16usize;
        let width = 64usize;
        let q8_bytes = super::cuda_impl::q8_layer_kv_allocated_bytes(capacity, width).unwrap();
        let f32_bytes = (2 * capacity * width * std::mem::size_of::<f32>()) as u64;

        assert_eq!(q8_bytes, 2176);
        assert_eq!(f32_bytes, 8192);
        assert!(q8_bytes < f32_bytes);
    }

    #[test]
    fn kq4_vq8_kv_allocated_bytes_formula_is_smaller_than_q8() {
        let capacity = 16usize;
        let width = 64usize;
        let q8_bytes = super::cuda_impl::q8_layer_kv_allocated_bytes(capacity, width).unwrap();
        let kq4_vq8_bytes =
            super::cuda_impl::kq4_vq8_layer_kv_allocated_bytes(capacity, width).unwrap();

        assert_eq!(kq4_vq8_bytes, 1664);
        assert!(kq4_vq8_bytes < q8_bytes);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::cuda_impl::ptx_jit_error_log;
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "value {idx} differs: actual={actual}, expected={expected}, delta={delta}"
            );
        }
    }

    fn append_q8_0_block(bytes: &mut Vec<u8>, scale_bits: u16, quants: [i8; 32]) {
        bytes.extend_from_slice(&scale_bits.to_le_bytes());
        bytes.extend(quants.iter().map(|value| *value as u8));
    }

    fn append_q4_0_block(bytes: &mut Vec<u8>, scale_bits: u16, quants: [i8; 32]) {
        bytes.extend_from_slice(&scale_bits.to_le_bytes());
        for idx in 0..16 {
            let low = (quants[idx] + 8) as u8 & 0x0f;
            let high = ((quants[idx + 16] + 8) as u8 & 0x0f) << 4;
            bytes.push(low | high);
        }
    }

    fn append_q4_k_block(bytes: &mut Vec<u8>, d_bits: u16, dmin_bits: u16, scales: [u8; 12]) {
        bytes.extend_from_slice(&d_bits.to_le_bytes());
        bytes.extend_from_slice(&dmin_bits.to_le_bytes());
        bytes.extend_from_slice(&scales);
        bytes.extend((0..128).map(|idx| {
            let low = (idx as u8).wrapping_mul(3) & 0x0f;
            let high = ((idx as u8).wrapping_mul(5).wrapping_add(1) & 0x0f) << 4;
            low | high
        }));
    }

    fn make_q5_k_block(seed: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(DType::Q5_K.block_bytes());
        block.extend_from_slice(&0x3c00u16.to_le_bytes());
        block.extend_from_slice(&0x3800u16.to_le_bytes());
        block.extend((0..12).map(|idx| seed.wrapping_add(idx as u8).wrapping_mul(7)));
        block.extend((0..32).map(|idx| seed.wrapping_add(idx as u8).wrapping_mul(3)));
        block.extend((0..128).map(|idx| seed.wrapping_add(idx as u8).wrapping_mul(5)));
        block
    }

    fn make_q6_k_block(seed: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(DType::Q6_K.block_bytes());
        block.extend((0..128).map(|idx| seed.wrapping_add(idx as u8).wrapping_mul(5)));
        block.extend((0..64).map(|idx| seed.wrapping_add(idx as u8).wrapping_mul(3)));
        block.extend((0..16).map(|idx| seed.wrapping_add(idx as u8) as i8 as u8));
        block.extend_from_slice(&0x3c00u16.to_le_bytes());
        block
    }

    #[test]
    fn q4_k_matrix_dequantizes_to_transposed_cpu_layout_without_cuda_device() -> Result<()> {
        let mut matrix = Vec::new();
        append_q4_k_block(
            &mut matrix,
            0x3800,
            0x2e66,
            [1, 2, 3, 4, 5, 6, 7, 8, 17, 34, 51, 68],
        );
        append_q4_k_block(
            &mut matrix,
            0x3400,
            0x2a66,
            [8, 7, 6, 5, 4, 3, 2, 1, 68, 51, 34, 17],
        );

        let transposed = super::cuda_impl::dequantize_q4_k_matrix_transposed(&matrix, 2, 256)?;
        let row_major = super::cuda_impl::transpose_row_major(&transposed, 2, 256)?;

        assert_eq!(transposed.len(), 512);
        assert_eq!(row_major.len(), 512);
        assert!(transposed.iter().any(|value| *value != 0.0));
        for row in 0..2 {
            let start = row * DType::Q4_K.block_bytes();
            let mut expected = vec![0.0f32; 256];
            xrt_kernels::cpu::dequantize_q4_k_row(
                &matrix[start..start + DType::Q4_K.block_bytes()],
                &mut expected,
            )?;
            for col in 0..256 {
                assert_eq!(transposed[col * 2 + row], expected[col]);
                assert_eq!(row_major[row * 256 + col], expected[col]);
            }
        }
        assert!(matches!(
            super::cuda_impl::dequantize_q4_k_matrix_transposed(&matrix, 2, 255),
            Err(XrtError::InvalidTensor(_))
        ));

        Ok(())
    }

    #[test]
    fn q5_k_matrix_dequantizes_to_transposed_cpu_layout_without_cuda_device() -> Result<()> {
        let mut matrix = make_q5_k_block(1);
        matrix.extend(make_q5_k_block(17));
        let transposed = super::cuda_impl::dequantize_q5_k_matrix_transposed(&matrix, 2, 256)?;
        let row_major = super::cuda_impl::transpose_row_major(&transposed, 2, 256)?;

        assert_eq!(transposed.len(), 512);
        assert_eq!(row_major.len(), 512);
        assert!(transposed.iter().any(|value| *value != 0.0));
        for row in 0..2 {
            let start = row * DType::Q5_K.block_bytes();
            let mut expected = vec![0.0f32; 256];
            xrt_kernels::cpu::dequantize_q5_k_row(
                &matrix[start..start + DType::Q5_K.block_bytes()],
                &mut expected,
            )?;
            for col in 0..256 {
                assert_eq!(transposed[col * 2 + row], expected[col]);
                assert_eq!(row_major[row * 256 + col], expected[col]);
            }
        }
        assert!(matches!(
            super::cuda_impl::dequantize_q5_k_matrix_transposed(&matrix, 2, 255),
            Err(XrtError::InvalidTensor(_))
        ));

        Ok(())
    }

    #[test]
    fn q6_k_matrix_dequantizes_to_transposed_cpu_layout_without_cuda_device() -> Result<()> {
        let mut matrix = make_q6_k_block(3);
        matrix.extend(make_q6_k_block(19));
        assert_eq!(
            super::cuda_impl::q6_k_block_scales(&matrix, 2, 256)?,
            vec![1.0, 1.0]
        );
        let transposed = super::cuda_impl::dequantize_q6_k_matrix_transposed(&matrix, 2, 256)?;
        let row_major = super::cuda_impl::transpose_row_major(&transposed, 2, 256)?;

        assert_eq!(transposed.len(), 512);
        assert_eq!(row_major.len(), 512);
        assert!(transposed.iter().any(|value| *value != 0.0));
        for row in 0..2 {
            let start = row * DType::Q6_K.block_bytes();
            let mut expected = vec![0.0f32; 256];
            xrt_kernels::cpu::dequantize_q6_k_row(
                &matrix[start..start + DType::Q6_K.block_bytes()],
                &mut expected,
            )?;
            for col in 0..256 {
                assert_eq!(transposed[col * 2 + row], expected[col]);
                assert_eq!(row_major[row * 256 + col], expected[col]);
            }
        }
        assert!(matches!(
            super::cuda_impl::dequantize_q6_k_matrix_transposed(&matrix, 2, 255),
            Err(XrtError::InvalidTensor(_))
        ));
        assert!(matches!(
            super::cuda_impl::q6_k_block_scales(&matrix, 2, 255),
            Err(XrtError::InvalidTensor(_))
        ));

        Ok(())
    }

    #[test]
    fn float_tensor_bytes_decode_supported_dtypes_without_cuda_device() -> Result<()> {
        let f32_bytes = [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat();
        assert_eq!(
            super::cuda_impl::decode_float_tensor_bytes(&f32_bytes, "f32", DType::F32, 2)?,
            vec![1.0, -2.0]
        );

        let f16_bytes = [0x3c00u16.to_le_bytes(), 0xc000u16.to_le_bytes()].concat();
        assert_eq!(
            super::cuda_impl::decode_float_tensor_bytes(&f16_bytes, "f16", DType::F16, 2)?,
            vec![1.0, -2.0]
        );

        let bf16_bytes = [0x3f80u16.to_le_bytes(), 0xc000u16.to_le_bytes()].concat();
        assert_eq!(
            super::cuda_impl::decode_float_tensor_bytes(&bf16_bytes, "bf16", DType::BF16, 2)?,
            vec![1.0, -2.0]
        );

        assert!(matches!(
            super::cuda_impl::decode_float_tensor_bytes(&f16_bytes, "bad", DType::F16, 3),
            Err(XrtError::Shape(_))
        ));

        Ok(())
    }

    fn q4_k_scale_min(index: usize, packed: &[u8]) -> (u8, u8) {
        if index < 4 {
            (packed[index] & 0x3f, packed[index + 4] & 0x3f)
        } else {
            (
                ((packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4)) & 0x3f,
                ((packed[index + 4] >> 4) | ((packed[index] >> 6) << 4)) & 0x3f,
            )
        }
    }

    fn q8_0_matvec_reference(
        matrix: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let blocks_per_row = cols / 32;
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            for block in 0..blocks_per_row {
                let global_block = row * blocks_per_row + block;
                let quant_offset = global_block * 34 + 2;
                let input_offset = block * 32;
                let scale = scales[global_block];
                for lane in 0..32 {
                    let quant = matrix[quant_offset + lane] as i8 as f32;
                    sum += scale * quant * input[input_offset + lane];
                }
            }
            output[row] = sum;
        }
        output
    }

    fn q4_0_matvec_reference(
        matrix: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let blocks_per_row = cols / 32;
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            for block in 0..blocks_per_row {
                let global_block = row * blocks_per_row + block;
                let quant_offset = global_block * 18 + 2;
                let input_offset = block * 32;
                let scale = scales[global_block];
                for lane in 0..16 {
                    let packed = matrix[quant_offset + lane];
                    let low = (packed & 0x0f) as i8 - 8;
                    let high = ((packed >> 4) & 0x0f) as i8 - 8;
                    sum += scale * low as f32 * input[input_offset + lane];
                    sum += scale * high as f32 * input[input_offset + 16 + lane];
                }
            }
            output[row] = sum;
        }
        output
    }

    fn q4_k_matvec_reference(matrix: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        let blocks_per_row = cols / 256;
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let mut sum = 0.0f32;
            for block_index in 0..blocks_per_row {
                let block_offset = (row * blocks_per_row + block_index) * 144;
                let block = &matrix[block_offset..block_offset + 144];
                let d = xrt_core::decode_f16(&block[0..2]).expect("valid Q4_K d");
                let dmin = xrt_core::decode_f16(&block[2..4]).expect("valid Q4_K dmin");
                let scales = &block[4..16];
                let qs = &block[16..144];
                for group in 0..4 {
                    let q = &qs[group * 32..(group + 1) * 32];
                    let (sc1, m1) = q4_k_scale_min(group * 2, scales);
                    let (sc2, m2) = q4_k_scale_min(group * 2 + 1, scales);
                    let d1 = d * sc1 as f32;
                    let d2 = d * sc2 as f32;
                    let min1 = dmin * m1 as f32;
                    let min2 = dmin * m2 as f32;
                    let base = block_index * 256 + group * 64;
                    for lane in 0..32 {
                        sum += (d1 * (q[lane] & 0x0f) as f32 - min1) * input[base + lane];
                        sum += (d2 * (q[lane] >> 4) as f32 - min2) * input[base + 32 + lane];
                    }
                }
            }
            output[row] = sum;
        }
        output
    }

    fn q4_k_rows_reference(matrix: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let blocks_per_row = cols / 256;
        let mut output = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for block_index in 0..blocks_per_row {
                let block_offset = (row * blocks_per_row + block_index) * 144;
                let block = &matrix[block_offset..block_offset + 144];
                let d = xrt_core::decode_f16(&block[0..2]).expect("valid Q4_K d");
                let dmin = xrt_core::decode_f16(&block[2..4]).expect("valid Q4_K dmin");
                let scales = &block[4..16];
                let qs = &block[16..144];
                for group in 0..4 {
                    let q = &qs[group * 32..(group + 1) * 32];
                    let (sc1, m1) = q4_k_scale_min(group * 2, scales);
                    let (sc2, m2) = q4_k_scale_min(group * 2 + 1, scales);
                    let d1 = d * sc1 as f32;
                    let d2 = d * sc2 as f32;
                    let min1 = dmin * m1 as f32;
                    let min2 = dmin * m2 as f32;
                    let base = block_index * 256 + group * 64;
                    for lane in 0..32 {
                        output[row * cols + base + lane] = d1 * (q[lane] & 0x0f) as f32 - min1;
                        output[row * cols + base + 32 + lane] = d2 * (q[lane] >> 4) as f32 - min2;
                    }
                }
            }
        }
        output
    }

    fn rope_reference(
        mut tensor: Vec<f32>,
        n_heads: usize,
        head_dim: usize,
        position: usize,
        rope_dim: usize,
        base: f32,
        scale: f32,
    ) -> Vec<f32> {
        let rotary_width = rope_dim.min(head_dim);
        let half_width = rotary_width / 2;
        for head in 0..n_heads {
            let head_offset = head * head_dim;
            for pair in 0..half_width {
                let first = head_offset + pair;
                let second = first + half_width;
                let theta =
                    position as f32 * scale * base.powf(-(2.0 * pair as f32) / rotary_width as f32);
                let (sin, cos) = theta.sin_cos();
                let x0 = tensor[first];
                let x1 = tensor[second];
                tensor[first] = x0 * cos - x1 * sin;
                tensor[second] = x0 * sin + x1 * cos;
            }
        }
        tensor
    }

    fn single_query_attention_reference(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        cache_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        single_query_attention_windowed_reference(
            query,
            keys,
            values,
            cache_len,
            n_heads,
            n_kv_heads,
            head_dim,
            0,
            1.0f32 / (head_dim as f32).sqrt(),
        )
    }

    fn single_query_attention_windowed_reference(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        cache_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        attend_start: usize,
        scale: f32,
    ) -> Vec<f32> {
        let kv_width = n_kv_heads * head_dim;
        let head_group = n_heads / n_kv_heads;
        let mut output = vec![0.0f32; n_heads * head_dim];
        for head in 0..n_heads {
            let kv_head = head / head_group;
            let mut scores = vec![0.0f32; cache_len - attend_start];
            for (score, pos) in scores.iter_mut().zip(attend_start..cache_len) {
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += query[head * head_dim + dim]
                        * keys[pos * kv_width + kv_head * head_dim + dim];
                }
                *score = dot * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                denom += *score;
            }
            for dim in 0..head_dim {
                let mut sum = 0.0f32;
                for (score, pos) in scores.iter().zip(attend_start..cache_len) {
                    let prob = *score / denom;
                    sum += prob * values[pos * kv_width + kv_head * head_dim + dim];
                }
                output[head * head_dim + dim] = sum;
            }
        }
        output
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn resident_f32_kernels_match_host_upload_path() -> Result<()> {
        let device = CudaDevice::new(0)?;

        let rms_input = [1.0, -2.0, 3.0, -4.0];
        let rms_weight = [0.5, 1.0, 1.5, 2.0];
        let rms_resident_weight = device.upload_f32(&rms_weight)?;
        let rms_host = device.rmsnorm(&rms_input, &rms_weight, 1, 4, 1e-5)?;
        let rms_resident =
            device.rmsnorm_resident_weight(&rms_input, &rms_resident_weight, 1, 4, 1e-5)?;
        assert_close(&rms_resident, &rms_host, 1e-5);
        let rms_input_device = device.upload_f32(&rms_input)?;
        let mut rms_scratch = device.zeros_f32(rms_input.len())?;
        device.rmsnorm_device_into(
            &rms_input_device,
            &rms_resident_weight,
            1,
            4,
            1e-5,
            &mut rms_scratch,
        )?;
        assert_close(&device.download_f32(&rms_scratch)?, &rms_host, 1e-5);

        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_resident = device.upload_f32(&rhs)?;
        let matmul_host = device.matmul(&lhs, 2, 3, &rhs, 2)?;
        let matmul_resident = device.matmul_resident_rhs(&lhs, 2, 3, &rhs_resident, 2)?;
        assert_close(&matmul_resident, &matmul_host, 1e-5);
        let lhs_device = device.upload_f32(&lhs)?;
        let mut matmul_scratch = device.zeros_f32(matmul_host.len())?;
        device.matmul_resident_rhs_device_into(
            &lhs_device,
            2,
            3,
            &rhs_resident,
            2,
            &mut matmul_scratch,
        )?;
        assert_close(&device.download_f32(&matmul_scratch)?, &matmul_host, 1e-5);

        let table = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let table_resident = device.upload_f32(&table)?;
        let embed_host = device.embed(&table, 3, 2, &[2, 0])?;
        let embed_resident = device.embed_resident(&table_resident, 3, 2, &[2, 0])?;
        assert_close(&embed_resident, &embed_host, 1e-5);

        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn silu_mul_device_path_matches_scalar_reference() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let gate = [-2.0f32, -0.5, 0.0, 1.5];
        let up = [3.0f32, -4.0, 5.0, 0.25];
        let gate_dev = device.upload_f32(&gate)?;
        let up_dev = device.upload_f32(&up)?;
        let silu_gate_dev = device.silu_device(&gate_dev)?;
        let swiglu_dev = device.mul_device(&silu_gate_dev, &up_dev)?;
        let swiglu = device.download_f32(&swiglu_dev)?;
        let expected_swiglu = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        assert_close(&swiglu, &expected_swiglu, 1e-5);
        let mut reusable_gate = device.zeros_f32(gate.len())?;
        device.upload_f32_into(&gate, &mut reusable_gate)?;
        device.silu_assign_device(&mut reusable_gate)?;
        device.mul_assign_device(&mut reusable_gate, &up_dev)?;
        assert_close(
            &device.download_f32(&reusable_gate)?,
            &expected_swiglu,
            1e-5,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn gemma4_activation_primitives_match_cpu_reference() -> Result<()> {
        let device = CudaDevice::new(0)?;

        let scaled_input = [1.0f32, -2.0, 0.5, 4.0];
        let mut scaled = device.upload_f32(&scaled_input)?;
        device.scale_assign_device(&mut scaled, 3.0)?;
        assert_close(
            &device.download_f32(&scaled)?,
            &[3.0, -6.0, 1.5, 12.0],
            1e-6,
        );

        let rms_input = [1.0f32, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5];
        let rms_input_device = device.upload_f32(&rms_input)?;
        let rms = device.rmsnorm_unweighted_device(&rms_input_device, 2, 4, 1e-6)?;
        let rms_expected = rms_input
            .chunks_exact(4)
            .flat_map(|row| {
                let inv_rms =
                    1.0 / (row.iter().map(|value| value * value).sum::<f32>() / 4.0 + 1e-6).sqrt();
                row.iter().map(move |value| value * inv_rms)
            })
            .collect::<Vec<_>>();
        assert_close(&device.download_f32(&rms)?, &rms_expected, 1e-5);

        let gate = [-2.0f32, -0.5, 0.0, 1.5, 3.0];
        let up = [3.0f32, -4.0, 5.0, 0.25, -0.75];
        let mut gate_device = device.upload_f32(&gate)?;
        let up_device = device.upload_f32(&up)?;
        device.geglu_pytorch_tanh_assign_device(&mut gate_device, &up_device)?;
        let geglu_expected = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| {
                let gelu = 0.5
                    * gate
                    * (1.0 + (0.797_884_6 * (gate + 0.044_715 * gate * gate * gate)).tanh());
                gelu * up
            })
            .collect::<Vec<_>>();
        assert_close(&device.download_f32(&gate_device)?, &geglu_expected, 2e-4);

        let logits = [-60.0f32, -6.0, 0.0, 6.0, 60.0];
        let mut logits_device = device.upload_f32(&logits)?;
        device.logit_softcap_assign_device(&mut logits_device, 30.0)?;
        let softcap_expected = logits
            .iter()
            .map(|value| (value / 30.0).tanh() * 30.0)
            .collect::<Vec<_>>();
        assert_close(
            &device.download_f32(&logits_device)?,
            &softcap_expected,
            2e-4,
        );

        assert!(device
            .logit_softcap_assign_device(&mut logits_device, 0.0)
            .is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn rope_device_path_matches_scalar_reference() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let tensor = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.25];
        let expected = rope_reference(tensor.clone(), 2, 4, 3, 4, 10000.0, 1.0);
        let mut tensor_dev = device.upload_f32(&tensor)?;
        device.rope_device(&mut tensor_dev, 2, 4, 3, 4, 10000.0, 1.0)?;
        let actual = device.download_f32(&tensor_dev)?;
        assert_close(&actual, &expected, 2e-3);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn repeat_kv_for_gqa_device_matches_scalar_reference() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let values = [10.0f32, 11.0, 20.0, 21.0];
        let expected = [10.0f32, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0];
        let values_dev = device.upload_f32(&values)?;
        let repeated_dev = device.repeat_kv_for_gqa_device(&values_dev, 4, 2, 2)?;
        let repeated = device.download_f32(&repeated_dev)?;
        assert_close(&repeated, &expected, 1e-6);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn single_query_attention_device_matches_scalar_reference() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 2;
        let keys = vec![1.0f32, 0.0, 0.0, 1.0];
        let values = vec![10.0f32, 20.0, 30.0, 40.0];
        let query = vec![1.0f32, 0.0, 0.0, 1.0];

        let mut cache = device.alloc_paged_layer_kv_cache(2, n_kv_heads * head_dim, 1)?;
        assert_eq!(cache.page_count(), 2);
        device.remap_paged_layer_kv_pages(&mut cache, &[1, 0])?;
        for pos in 0..2 {
            let start = pos * n_kv_heads * head_dim;
            let end = start + n_kv_heads * head_dim;
            let key = device.upload_f32(&keys[start..end])?;
            let value = device.upload_f32(&values[start..end])?;
            device.append_layer_kv(&mut cache, &key, &value)?;
        }
        assert_eq!(cache.len(), 2);

        let query_dev = device.upload_f32(&query)?;
        let output_dev = device
            .single_query_attention_device(&query_dev, &cache, n_heads, n_kv_heads, head_dim)?;
        let output = device.download_f32(&output_dev)?;
        let expected = single_query_attention_reference(
            &query, &keys, &values, 2, n_heads, n_kv_heads, head_dim,
        );
        assert_close(&output, &expected, 2e-2);

        let windowed_dev = device.single_query_attention_windowed_device(
            &query_dev, &cache, n_heads, n_kv_heads, head_dim, 1, 1.0,
        )?;
        let windowed = device.download_f32(&windowed_dev)?;
        assert_close(&windowed, &[30.0, 40.0, 30.0, 40.0], 2e-2);

        let wide_n_heads = 4;
        let wide_n_kv_heads = 2;
        let wide_head_dim = 128;
        let wide_cache_len = 5;
        let wide_kv_width = wide_n_kv_heads * wide_head_dim;
        let wide_query = (0..wide_n_heads * wide_head_dim)
            .map(|idx| ((idx * 17 % 37) as f32 - 18.0) / 23.0)
            .collect::<Vec<_>>();
        let wide_keys = (0..wide_cache_len * wide_kv_width)
            .map(|idx| ((idx * 13 % 41) as f32 - 20.0) / 29.0)
            .collect::<Vec<_>>();
        let wide_values = (0..wide_cache_len * wide_kv_width)
            .map(|idx| ((idx * 19 % 43) as f32 - 21.0) / 31.0)
            .collect::<Vec<_>>();
        let mut wide_cache = device.alloc_paged_layer_kv_cache(6, wide_kv_width, 2)?;
        device.remap_paged_layer_kv_pages(&mut wide_cache, &[2, 0, 1])?;
        for pos in 0..wide_cache_len {
            let start = pos * wide_kv_width;
            let end = start + wide_kv_width;
            let key = device.upload_f32(&wide_keys[start..end])?;
            let value = device.upload_f32(&wide_values[start..end])?;
            device.append_layer_kv(&mut wide_cache, &key, &value)?;
        }
        let wide_query_dev = device.upload_f32(&wide_query)?;
        let wide_output_dev = device.single_query_attention_windowed_device(
            &wide_query_dev,
            &wide_cache,
            wide_n_heads,
            wide_n_kv_heads,
            wide_head_dim,
            1,
            0.75,
        )?;
        let wide_output = device.download_f32(&wide_output_dev)?;
        let wide_expected = single_query_attention_windowed_reference(
            &wide_query,
            &wide_keys,
            &wide_values,
            wide_cache_len,
            wide_n_heads,
            wide_n_kv_heads,
            wide_head_dim,
            1,
            0.75,
        );
        assert_close(&wide_output, &wide_expected, 3e-2);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn q8_layer_kv_append_dequantize_matches_scalar_reference() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let key = [0.0f32, 0.5, -1.0, 1.25, -0.25, 0.75, -0.5, 0.125];
        let value = [1.0f32, -0.75, 0.25, -0.125, 0.5, -1.25, 0.0, 0.875];
        let key_2 = [0.25f32, -0.375, 0.875, -1.5, 0.125, -0.25, 0.5, -0.625];
        let value_2 = [-0.5f32, 0.25, 1.0, -0.875, 0.625, 0.0, -1.125, 0.375];
        let key_dev = device.upload_f32(&key)?;
        let value_dev = device.upload_f32(&value)?;
        let key_2_dev = device.upload_f32(&key_2)?;
        let value_2_dev = device.upload_f32(&value_2)?;
        let mut growth_cache = device.alloc_paged_q8_layer_kv_cache(1, key.len(), 1)?;
        device.append_q8_layer_kv(&mut growth_cache, &key_dev, &value_dev)?;
        device.grow_q8_layer_kv_cache(&mut growth_cache, 2)?;
        assert_eq!(growth_cache.capacity(), 2);
        assert_eq!(growth_cache.len(), 1);
        let (grown_key_dev, grown_value_dev) = device.dequantize_q8_layer_kv(&growth_cache, 0)?;
        assert_close(&device.download_f32(&grown_key_dev)?, &key, 1.0 / 127.0);
        assert_close(&device.download_f32(&grown_value_dev)?, &value, 1.0 / 127.0);

        let mut cache = device.alloc_paged_q8_layer_kv_cache(2, key.len(), 1)?;
        assert_eq!(cache.page_count(), 2);
        device.remap_paged_q8_layer_kv_pages(&mut cache, &[1, 0])?;
        device.append_q8_layer_kv(&mut cache, &key_dev, &value_dev)?;
        device.append_q8_layer_kv(&mut cache, &key_2_dev, &value_2_dev)?;

        assert_eq!(cache.len(), 2);
        let (roundtrip_key_dev, roundtrip_value_dev) = device.dequantize_q8_layer_kv(&cache, 0)?;
        let roundtrip_key = device.download_f32(&roundtrip_key_dev)?;
        let roundtrip_value = device.download_f32(&roundtrip_value_dev)?;
        assert_close(&roundtrip_key, &key, 1.0 / 127.0);
        assert_close(&roundtrip_value, &value, 1.0 / 127.0);

        let (roundtrip_key_2_dev, roundtrip_value_2_dev) =
            device.dequantize_q8_layer_kv(&cache, 1)?;
        let roundtrip_key_2 = device.download_f32(&roundtrip_key_2_dev)?;
        let roundtrip_value_2 = device.download_f32(&roundtrip_value_2_dev)?;
        assert_close(&roundtrip_key_2, &key_2, 1.0 / 127.0);
        assert_close(&roundtrip_value_2, &value_2, 1.0 / 127.0);

        let query = [0.125f32, 0.5, -0.75, 1.0, -0.25, 0.375, -0.5, 0.875];
        let query_dev = device.upload_f32(&query)?;
        let attention_dev =
            device.single_query_attention_q8_device(&query_dev, &cache, 1, 1, key.len())?;
        let attention = device.download_f32(&attention_dev)?;
        let mut dequantized_keys = Vec::with_capacity(key.len() * 2);
        dequantized_keys.extend_from_slice(&roundtrip_key);
        dequantized_keys.extend_from_slice(&roundtrip_key_2);
        let mut dequantized_values = Vec::with_capacity(value.len() * 2);
        dequantized_values.extend_from_slice(&roundtrip_value);
        dequantized_values.extend_from_slice(&roundtrip_value_2);
        let expected_attention = single_query_attention_reference(
            &query,
            &dequantized_keys,
            &dequantized_values,
            2,
            1,
            1,
            key.len(),
        );
        assert_close(&attention, &expected_attention, 2e-2);

        let windowed_attention_dev = device.single_query_attention_q8_windowed_device(
            &query_dev,
            &cache,
            1,
            1,
            key.len(),
            1,
            1.0,
        )?;
        let windowed_attention = device.download_f32(&windowed_attention_dev)?;
        let expected_windowed_attention = single_query_attention_windowed_reference(
            &query,
            &dequantized_keys,
            &dequantized_values,
            2,
            1,
            1,
            key.len(),
            1,
            1.0,
        );
        assert_close(&windowed_attention, &expected_windowed_attention, 2e-2);
        let gemma_scale_attention_dev = device.single_query_attention_q8_windowed_device(
            &query_dev,
            &cache,
            1,
            1,
            key.len(),
            0,
            1.0,
        )?;
        let expected_gemma_scale_attention = single_query_attention_windowed_reference(
            &query,
            &dequantized_keys,
            &dequantized_values,
            2,
            1,
            1,
            key.len(),
            0,
            1.0,
        );
        assert_close(
            &device.download_f32(&gemma_scale_attention_dev)?,
            &expected_gemma_scale_attention,
            2e-2,
        );

        let wide_head_dim = 128;
        let wide_n_heads = 2;
        let wide_n_kv_heads = 1;
        let wide_cache_len = 3;
        let wide_keys = (0..wide_cache_len * wide_head_dim)
            .map(|idx| ((idx * 11 % 31) as f32 - 15.0) / 19.0)
            .collect::<Vec<_>>();
        let wide_values = (0..wide_cache_len * wide_head_dim)
            .map(|idx| ((idx * 7 % 29) as f32 - 14.0) / 17.0)
            .collect::<Vec<_>>();
        let wide_query = (0..wide_n_heads * wide_head_dim)
            .map(|idx| ((idx * 13 % 37) as f32 - 18.0) / 23.0)
            .collect::<Vec<_>>();
        let mut wide_cache = device.alloc_paged_q8_layer_kv_cache(4, wide_head_dim, 2)?;
        device.remap_paged_q8_layer_kv_pages(&mut wide_cache, &[1, 0])?;
        for pos in 0..wide_cache_len {
            let start = pos * wide_head_dim;
            let end = start + wide_head_dim;
            let key = device.upload_f32(&wide_keys[start..end])?;
            let value = device.upload_f32(&wide_values[start..end])?;
            device.append_q8_layer_kv(&mut wide_cache, &key, &value)?;
        }
        let mut wide_dequantized_keys = Vec::with_capacity(wide_keys.len());
        let mut wide_dequantized_values = Vec::with_capacity(wide_values.len());
        for pos in 0..wide_cache_len {
            let (key, value) = device.dequantize_q8_layer_kv(&wide_cache, pos)?;
            wide_dequantized_keys.extend(device.download_f32(&key)?);
            wide_dequantized_values.extend(device.download_f32(&value)?);
        }
        let wide_query_dev = device.upload_f32(&wide_query)?;
        let wide_attention_dev = device.single_query_attention_q8_windowed_device(
            &wide_query_dev,
            &wide_cache,
            wide_n_heads,
            wide_n_kv_heads,
            wide_head_dim,
            1,
            0.5,
        )?;
        let wide_expected = single_query_attention_windowed_reference(
            &wide_query,
            &wide_dequantized_keys,
            &wide_dequantized_values,
            wide_cache_len,
            wide_n_heads,
            wide_n_kv_heads,
            wide_head_dim,
            1,
            0.5,
        );
        assert_close(
            &device.download_f32(&wide_attention_dev)?,
            &wide_expected,
            3e-2,
        );

        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn kq4_vq8_layer_kv_append_dequantize_matches_scalar_reference() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let key = [0.0f32, 0.5, -1.0, 1.25, -0.25, 0.75, -0.5, 0.125];
        let value = [1.0f32, -0.75, 0.25, -0.125, 0.5, -1.25, 0.0, 0.875];
        let key_2 = [0.25f32, -0.375, 0.875, -1.5, 0.125, -0.25, 0.5, -0.625];
        let value_2 = [-0.5f32, 0.25, 1.0, -0.875, 0.625, 0.0, -1.125, 0.375];
        let key_dev = device.upload_f32(&key)?;
        let value_dev = device.upload_f32(&value)?;
        let key_2_dev = device.upload_f32(&key_2)?;
        let value_2_dev = device.upload_f32(&value_2)?;
        let mut growth_cache =
            device.alloc_paged_key_q4_value_q8_layer_kv_cache(1, key.len(), 1)?;
        device.append_key_q4_value_q8_layer_kv(&mut growth_cache, &key_dev, &value_dev)?;
        device.grow_key_q4_value_q8_layer_kv_cache(&mut growth_cache, 2)?;
        assert_eq!(growth_cache.capacity(), 2);
        assert_eq!(growth_cache.len(), 1);
        let (grown_key_dev, grown_value_dev) =
            device.dequantize_key_q4_value_q8_layer_kv(&growth_cache, 0)?;
        assert_close(&device.download_f32(&grown_key_dev)?, &key, 0.25);
        assert_close(&device.download_f32(&grown_value_dev)?, &value, 1.0 / 127.0);

        let mut cache = device.alloc_paged_key_q4_value_q8_layer_kv_cache(2, key.len(), 1)?;
        assert_eq!(cache.page_count(), 2);
        device.remap_paged_key_q4_value_q8_layer_kv_pages(&mut cache, &[1, 0])?;
        device.append_key_q4_value_q8_layer_kv(&mut cache, &key_dev, &value_dev)?;
        device.append_key_q4_value_q8_layer_kv(&mut cache, &key_2_dev, &value_2_dev)?;

        assert_eq!(cache.len(), 2);
        let (roundtrip_key_dev, roundtrip_value_dev) =
            device.dequantize_key_q4_value_q8_layer_kv(&cache, 0)?;
        let roundtrip_key = device.download_f32(&roundtrip_key_dev)?;
        let roundtrip_value = device.download_f32(&roundtrip_value_dev)?;
        assert_close(&roundtrip_key, &key, 0.25);
        assert_close(&roundtrip_value, &value, 1.0 / 127.0);

        let (roundtrip_key_2_dev, roundtrip_value_2_dev) =
            device.dequantize_key_q4_value_q8_layer_kv(&cache, 1)?;
        let roundtrip_key_2 = device.download_f32(&roundtrip_key_2_dev)?;
        let roundtrip_value_2 = device.download_f32(&roundtrip_value_2_dev)?;
        assert_close(&roundtrip_key_2, &key_2, 0.25);
        assert_close(&roundtrip_value_2, &value_2, 1.0 / 127.0);

        let mut copied_cache =
            device.alloc_paged_key_q4_value_q8_layer_kv_cache(2, key.len(), 1)?;
        device.remap_paged_key_q4_value_q8_layer_kv_pages(&mut copied_cache, &[1, 0])?;
        device.copy_key_q4_value_q8_layer_kv_row(&cache, 1, &mut copied_cache)?;
        assert_eq!(copied_cache.len(), 1);
        let (copied_key_dev, copied_value_dev) =
            device.dequantize_key_q4_value_q8_layer_kv(&copied_cache, 0)?;
        assert_close(
            &device.download_f32(&copied_key_dev)?,
            &roundtrip_key_2,
            1e-6,
        );
        assert_close(
            &device.download_f32(&copied_value_dev)?,
            &roundtrip_value_2,
            1e-6,
        );

        let query = [0.125f32, 0.5, -0.75, 1.0, -0.25, 0.375, -0.5, 0.875];
        let query_dev = device.upload_f32(&query)?;
        let attention_dev = device.single_query_attention_key_q4_value_q8_device(
            &query_dev,
            &cache,
            1,
            1,
            key.len(),
        )?;
        let attention = device.download_f32(&attention_dev)?;
        let mut dequantized_keys = Vec::with_capacity(key.len() * 2);
        dequantized_keys.extend_from_slice(&roundtrip_key);
        dequantized_keys.extend_from_slice(&roundtrip_key_2);
        let mut dequantized_values = Vec::with_capacity(value.len() * 2);
        dequantized_values.extend_from_slice(&roundtrip_value);
        dequantized_values.extend_from_slice(&roundtrip_value_2);
        let expected_attention = single_query_attention_reference(
            &query,
            &dequantized_keys,
            &dequantized_values,
            2,
            1,
            1,
            key.len(),
        );
        assert_close(&attention, &expected_attention, 2e-2);

        let windowed_attention_dev = device
            .single_query_attention_key_q4_value_q8_windowed_device(
                &query_dev,
                &cache,
                1,
                1,
                key.len(),
                1,
                1.0,
            )?;
        let windowed_attention = device.download_f32(&windowed_attention_dev)?;
        let expected_windowed_attention = single_query_attention_windowed_reference(
            &query,
            &dequantized_keys,
            &dequantized_values,
            2,
            1,
            1,
            key.len(),
            1,
            1.0,
        );
        assert_close(&windowed_attention, &expected_windowed_attention, 2e-2);
        let gemma_scale_attention_dev = device
            .single_query_attention_key_q4_value_q8_windowed_device(
                &query_dev,
                &cache,
                1,
                1,
                key.len(),
                0,
                1.0,
            )?;
        let expected_gemma_scale_attention = single_query_attention_windowed_reference(
            &query,
            &dequantized_keys,
            &dequantized_values,
            2,
            1,
            1,
            key.len(),
            0,
            1.0,
        );
        assert_close(
            &device.download_f32(&gemma_scale_attention_dev)?,
            &expected_gemma_scale_attention,
            2e-2,
        );

        let mut hot_cache = device.alloc_layer_kv_cache(1, key.len())?;
        let mut cold_cache = device.alloc_key_q4_value_q8_layer_kv_cache(1, key.len())?;
        device.append_layer_kv(&mut hot_cache, &key_dev, &value_dev)?;
        device.append_key_q4_value_q8_layer_kv(&mut cold_cache, &key_2_dev, &value_2_dev)?;
        let mut routes = device.alloc_adaptive_kv_routes(2)?;
        device.append_adaptive_kv_route(&mut routes, true, 0)?;
        device.append_adaptive_kv_route(&mut routes, false, 0)?;
        assert_eq!(routes.len(), 2);
        assert_eq!(routes.allocated_bytes(), 8);
        device.grow_adaptive_kv_routes(&mut routes, 4)?;
        assert_eq!(routes.capacity(), 4);
        device.replace_adaptive_kv_routes(&mut routes, &[1, 0])?;
        let mixed_attention_dev = device.single_query_attention_mixed_key_q4_value_q8_device(
            &query_dev,
            &hot_cache,
            &cold_cache,
            &routes,
            1,
            1,
            key.len(),
        )?;
        let mixed_attention = device.download_f32(&mixed_attention_dev)?;
        let mut mixed_keys = Vec::with_capacity(key.len() * 2);
        mixed_keys.extend_from_slice(&key);
        mixed_keys.extend_from_slice(&roundtrip_key_2);
        let mut mixed_values = Vec::with_capacity(value.len() * 2);
        mixed_values.extend_from_slice(&value);
        mixed_values.extend_from_slice(&roundtrip_value_2);
        let expected_mixed_attention = single_query_attention_reference(
            &query,
            &mixed_keys,
            &mixed_values,
            2,
            1,
            1,
            key.len(),
        );
        assert_close(&mixed_attention, &expected_mixed_attention, 2e-2);
        let mixed_windowed_attention_dev = device
            .single_query_attention_mixed_key_q4_value_q8_windowed_device(
                &query_dev,
                &hot_cache,
                &cold_cache,
                &routes,
                1,
                1,
                key.len(),
                1,
                1.0,
            )?;
        let expected_mixed_windowed_attention = single_query_attention_windowed_reference(
            &query,
            &mixed_keys,
            &mixed_values,
            2,
            1,
            1,
            key.len(),
            1,
            1.0,
        );
        assert_close(
            &device.download_f32(&mixed_windowed_attention_dev)?,
            &expected_mixed_windowed_attention,
            2e-2,
        );

        let mut remapped_hot_cache = device.alloc_paged_layer_kv_cache(2, key.len(), 1)?;
        let mut remapped_cold_cache =
            device.alloc_paged_key_q4_value_q8_layer_kv_cache(2, key.len(), 1)?;
        device.remap_paged_layer_kv_pages(&mut remapped_hot_cache, &[1, 0])?;
        device.remap_paged_key_q4_value_q8_layer_kv_pages(&mut remapped_cold_cache, &[1, 0])?;
        device.append_layer_kv(&mut remapped_hot_cache, &key_dev, &value_dev)?;
        device.append_layer_kv(&mut remapped_hot_cache, &key_2_dev, &value_2_dev)?;
        device.append_key_q4_value_q8_layer_kv(
            &mut remapped_cold_cache,
            &key_2_dev,
            &value_2_dev,
        )?;
        device.append_key_q4_value_q8_layer_kv(&mut remapped_cold_cache, &key_dev, &value_dev)?;
        let mut remapped_routes = device.alloc_adaptive_kv_routes(4)?;
        for (is_hot, local_position) in [(true, 0), (false, 0), (true, 1), (false, 1)] {
            device.append_adaptive_kv_route(&mut remapped_routes, is_hot, local_position)?;
        }
        let (cold_key_0, cold_value_0) =
            device.dequantize_key_q4_value_q8_layer_kv(&remapped_cold_cache, 0)?;
        let (cold_key_1, cold_value_1) =
            device.dequantize_key_q4_value_q8_layer_kv(&remapped_cold_cache, 1)?;
        let mut remapped_keys = key.to_vec();
        remapped_keys.extend_from_slice(&device.download_f32(&cold_key_0)?);
        remapped_keys.extend_from_slice(&key_2);
        remapped_keys.extend_from_slice(&device.download_f32(&cold_key_1)?);
        let mut remapped_values = value.to_vec();
        remapped_values.extend_from_slice(&device.download_f32(&cold_value_0)?);
        remapped_values.extend_from_slice(&value_2);
        remapped_values.extend_from_slice(&device.download_f32(&cold_value_1)?);
        let remapped_attention = device
            .single_query_attention_mixed_key_q4_value_q8_windowed_device(
                &query_dev,
                &remapped_hot_cache,
                &remapped_cold_cache,
                &remapped_routes,
                1,
                1,
                key.len(),
                1,
                1.0,
            )?;
        let expected_remapped_attention = single_query_attention_windowed_reference(
            &query,
            &remapped_keys,
            &remapped_values,
            4,
            1,
            1,
            key.len(),
            1,
            1.0,
        );
        assert_close(
            &device.download_f32(&remapped_attention)?,
            &expected_remapped_attention,
            2e-2,
        );

        // Cross a real 128-wide attention head so scale indexing cannot accidentally
        // collapse to one scale per head or use the old 32-element grouping.
        let wide_key = (0..128)
            .map(|index| match index {
                0..=31 => 0.25,
                32..=63 => 8.0,
                64..=95 => -0.25,
                _ => -8.0,
            })
            .collect::<Vec<_>>();
        let wide_key_2 = wide_key.iter().rev().copied().collect::<Vec<_>>();
        let wide_value = (0..128)
            .map(|index| (index as f32 - 63.5) / 64.0)
            .collect::<Vec<_>>();
        let wide_value_2 = wide_value.iter().map(|value| -*value).collect::<Vec<_>>();
        let mut wide_cache = device.alloc_key_q4_value_q8_layer_kv_cache(2, 128)?;
        device.append_key_q4_value_q8_layer_kv(
            &mut wide_cache,
            &device.upload_f32(&wide_key)?,
            &device.upload_f32(&wide_value)?,
        )?;
        device.append_key_q4_value_q8_layer_kv(
            &mut wide_cache,
            &device.upload_f32(&wide_key_2)?,
            &device.upload_f32(&wide_value_2)?,
        )?;
        let (wide_roundtrip_key_dev, wide_roundtrip_value_dev) =
            device.dequantize_key_q4_value_q8_layer_kv(&wide_cache, 0)?;
        let (wide_roundtrip_key_2_dev, wide_roundtrip_value_2_dev) =
            device.dequantize_key_q4_value_q8_layer_kv(&wide_cache, 1)?;
        let wide_roundtrip_key = device.download_f32(&wide_roundtrip_key_dev)?;
        assert_close(&wide_roundtrip_key[..32], &[0.0; 32], 1e-6);
        assert_close(&wide_roundtrip_key[32..64], &[7.0; 32], 1e-6);
        assert_close(&wide_roundtrip_key[64..96], &[0.0; 32], 1e-6);
        assert_close(&wide_roundtrip_key[96..], &[-8.0; 32], 1e-6);

        let wide_query = vec![1.0f32; 128];
        let wide_attention_dev = device.single_query_attention_key_q4_value_q8_device(
            &device.upload_f32(&wide_query)?,
            &wide_cache,
            1,
            1,
            128,
        )?;
        let wide_attention = device.download_f32(&wide_attention_dev)?;
        let wide_roundtrip_key_2 = device.download_f32(&wide_roundtrip_key_2_dev)?;
        let mut wide_keys = wide_roundtrip_key;
        wide_keys.extend_from_slice(&wide_roundtrip_key_2);
        let mut wide_values = device.download_f32(&wide_roundtrip_value_dev)?;
        wide_values.extend_from_slice(&device.download_f32(&wide_roundtrip_value_2_dev)?);
        let expected_wide_attention =
            single_query_attention_reference(&wide_query, &wide_keys, &wide_values, 2, 1, 1, 128);
        assert_close(&wide_attention, &expected_wide_attention, 2e-2);

        let mut wide_hot_cache = device.alloc_layer_kv_cache(1, 128)?;
        device.append_layer_kv(
            &mut wide_hot_cache,
            &device.upload_f32(&wide_key_2)?,
            &device.upload_f32(&wide_value_2)?,
        )?;
        let mut wide_mixed_routes = device.alloc_adaptive_kv_routes(3)?;
        for (is_hot, local_position) in [(false, 0), (true, 0), (false, 1)] {
            device.append_adaptive_kv_route(&mut wide_mixed_routes, is_hot, local_position)?;
        }
        let mut wide_gqa_query = wide_query.clone();
        wide_gqa_query.extend(wide_query.iter().map(|value| -*value));
        let mut wide_mixed_keys = wide_keys[..128].to_vec();
        wide_mixed_keys.extend_from_slice(&wide_key_2);
        wide_mixed_keys.extend_from_slice(&wide_keys[128..]);
        let mut wide_mixed_values = wide_values[..128].to_vec();
        wide_mixed_values.extend_from_slice(&wide_value_2);
        wide_mixed_values.extend_from_slice(&wide_values[128..]);
        let wide_mixed_attention = device
            .single_query_attention_mixed_key_q4_value_q8_windowed_device(
                &device.upload_f32(&wide_gqa_query)?,
                &wide_hot_cache,
                &wide_cache,
                &wide_mixed_routes,
                2,
                1,
                128,
                1,
                1.0,
            )?;
        let expected_wide_mixed_attention = single_query_attention_windowed_reference(
            &wide_gqa_query,
            &wide_mixed_keys,
            &wide_mixed_values,
            3,
            2,
            1,
            128,
            1,
            1.0,
        );
        assert_close(
            &device.download_f32(&wide_mixed_attention)?,
            &expected_wide_mixed_attention,
            2e-2,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn q8_0_matvec_kernel_matches_scalar_reference() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let matvec_tolerance = 1e-3;

        let mut q8_matrix = Vec::new();
        let scales = [0.5, 1.0, 0.25, 2.0];
        append_q8_0_block(
            &mut q8_matrix,
            0x3800,
            core::array::from_fn(|idx| idx as i8 - 16),
        );
        append_q8_0_block(
            &mut q8_matrix,
            0x3c00,
            core::array::from_fn(|idx| 15 - idx as i8),
        );
        append_q8_0_block(
            &mut q8_matrix,
            0x3400,
            core::array::from_fn(|idx| (idx as i8 % 7) - 3),
        );
        append_q8_0_block(
            &mut q8_matrix,
            0x4000,
            core::array::from_fn(|idx| (idx as i8 % 11) - 5),
        );
        let q8_input = (0..64)
            .map(|idx| (idx as f32 - 31.5) / 17.0)
            .collect::<Vec<_>>();
        let q8_expected = q8_0_matvec_reference(&q8_matrix, &scales, 2, 64, &q8_input);
        let q8_host_upload = device.matvec_q8_0(&q8_matrix, 2, 64, &q8_input)?;
        assert_close(&q8_host_upload, &q8_expected, matvec_tolerance);

        let q8_resident = device.upload_q8_0_matrix(&q8_matrix, 2, 64)?;
        assert_eq!(q8_resident.rows(), 2);
        assert_eq!(q8_resident.cols(), 64);
        assert_eq!(q8_resident.scale_count(), 4);
        assert_eq!(q8_resident.quant_byte_len(), 128);
        let q8_input_dev = device.upload_f32(&q8_input)?;
        let q8_resident_output_dev =
            device.matvec_q8_0_resident_device(&q8_resident, &q8_input_dev)?;
        let q8_resident_output = device.download_f32(&q8_resident_output_dev)?;
        assert_close(&q8_resident_output, &q8_expected, matvec_tolerance);
        let mut q8_output_scratch = device.zeros_f32(q8_resident.rows())?;
        device.matvec_q8_0_resident_device_into(
            &q8_resident,
            &q8_input_dev,
            &mut q8_output_scratch,
        )?;
        assert_close(
            &device.download_f32(&q8_output_scratch)?,
            &q8_expected,
            matvec_tolerance,
        );

        let mut q4_matrix = Vec::new();
        append_q4_0_block(
            &mut q4_matrix,
            0x3800,
            core::array::from_fn(|idx| (idx as i8 % 16) - 8),
        );
        append_q4_0_block(
            &mut q4_matrix,
            0x3c00,
            core::array::from_fn(|idx| 7 - (idx as i8 % 16)),
        );
        append_q4_0_block(
            &mut q4_matrix,
            0x3400,
            core::array::from_fn(|idx| (idx as i8 % 9) - 4),
        );
        append_q4_0_block(
            &mut q4_matrix,
            0x4000,
            core::array::from_fn(|idx| (idx as i8 % 13) - 6),
        );
        let q4_expected = q4_0_matvec_reference(&q4_matrix, &scales, 2, 64, &q8_input);
        let q4_host_upload = device.matvec_q4_0(&q4_matrix, 2, 64, &q8_input)?;
        assert_close(&q4_host_upload, &q4_expected, matvec_tolerance);

        let q4_resident = device.upload_q4_0_matrix(&q4_matrix, 2, 64)?;
        assert_eq!(q4_resident.rows(), 2);
        assert_eq!(q4_resident.cols(), 64);
        assert_eq!(q4_resident.scale_count(), 4);
        assert_eq!(q4_resident.quant_byte_len(), 128);
        let q4_resident_output_dev =
            device.matvec_q4_0_resident_device(&q4_resident, &q8_input_dev)?;
        let q4_resident_output = device.download_f32(&q4_resident_output_dev)?;
        assert_close(&q4_resident_output, &q4_expected, matvec_tolerance);
        let mut q4_output_scratch = device.zeros_f32(q4_resident.rows())?;
        device.matvec_q4_0_resident_device_into(
            &q4_resident,
            &q8_input_dev,
            &mut q4_output_scratch,
        )?;
        assert_close(
            &device.download_f32(&q4_output_scratch)?,
            &q4_expected,
            matvec_tolerance,
        );

        let mut q4k_matrix = Vec::new();
        append_q4_k_block(
            &mut q4k_matrix,
            0x3800,
            0x2e66,
            [1, 2, 3, 4, 5, 6, 7, 8, 17, 34, 51, 68],
        );
        append_q4_k_block(
            &mut q4k_matrix,
            0x3400,
            0x2a66,
            [8, 7, 6, 5, 4, 3, 2, 1, 68, 51, 34, 17],
        );
        let q4k_input = (0..256)
            .map(|idx| (idx as f32 - 127.5) / 53.0)
            .collect::<Vec<_>>();
        let q4k_expected = q4_k_matvec_reference(&q4k_matrix, 2, 256, &q4k_input);
        let q4k_host_upload = device.matvec_q4_k(&q4k_matrix, 2, 256, &q4k_input)?;
        assert_close(&q4k_host_upload, &q4k_expected, matvec_tolerance);

        let q4k_resident = device.upload_q4_k_matrix_packed(&q4k_matrix, 2, 256)?;
        assert_eq!(q4k_resident.rows(), 2);
        assert_eq!(q4k_resident.cols(), 256);
        assert_eq!(q4k_resident.byte_len(), 2 * (4 + 4 + 12 + 128));
        let q4k_input_dev = device.upload_f32(&q4k_input)?;
        let q4k_resident_output_dev =
            device.matvec_q4_k_resident_device(&q4k_resident, &q4k_input_dev)?;
        let q4k_resident_output = device.download_f32(&q4k_resident_output_dev)?;
        assert_close(&q4k_resident_output, &q4k_expected, matvec_tolerance);
        let mut q4k_output_scratch = device.zeros_f32(q4k_resident.rows())?;
        device.matvec_q4_k_resident_device_into(
            &q4k_resident,
            &q4k_input_dev,
            &mut q4k_output_scratch,
        )?;
        assert_close(
            &device.download_f32(&q4k_output_scratch)?,
            &q4k_expected,
            matvec_tolerance,
        );

        let q4k_embedding_dev = device.embed_q4_k_resident_device(&q4k_resident, &[1, 0])?;
        let q4k_embedding = device.download_f32(&q4k_embedding_dev)?;
        let q4k_rows = q4_k_rows_reference(&q4k_matrix, 2, 256);
        assert_close(&q4k_embedding[0..256], &q4k_rows[256..512], 1e-4);
        assert_close(&q4k_embedding[256..512], &q4k_rows[0..256], 1e-4);

        let q4k_expanded = device.upload_q4_k_embedding_matrix(&q4k_matrix, 2, 256)?;
        let q4k_expanded_embedding_dev =
            device.embed_q4_k_resident_device(&q4k_expanded, &[1, 0])?;
        let q4k_expanded_embedding = device.download_f32(&q4k_expanded_embedding_dev)?;
        assert_close(&q4k_expanded_embedding[0..256], &q4k_rows[256..512], 1e-4);
        assert_close(&q4k_expanded_embedding[256..512], &q4k_rows[0..256], 1e-4);

        let mut q6k_matrix = make_q6_k_block(3);
        q6k_matrix.extend(make_q6_k_block(7));
        q6k_matrix.extend(make_q6_k_block(19));
        q6k_matrix.extend(make_q6_k_block(23));
        let q6k_input = (0..512)
            .map(|idx| (idx as f32 - 255.5) / 113.0)
            .collect::<Vec<_>>();
        let q6k_expected = (0..2)
            .map(|row| {
                let start = row * 2 * DType::Q6_K.block_bytes();
                xrt_kernels::cpu::q6_k_row_dot(
                    &q6k_matrix[start..start + 2 * DType::Q6_K.block_bytes()],
                    &q6k_input,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let q6k_resident = device.upload_q6_k_embedding_matrix_packed(&q6k_matrix, 2, 512)?;
        assert_eq!(q6k_resident.byte_len(), 4 * (4 + DType::Q6_K.block_bytes()));
        let q6k_input_dev = device.upload_f32(&q6k_input)?;
        let q6k_output_dev = device.matvec_q6_k_resident_device(&q6k_resident, &q6k_input_dev)?;
        assert_close(&device.download_f32(&q6k_output_dev)?, &q6k_expected, 1e-1);

        let q6k_embedding_dev = device.embed_q6_k_resident_device(&q6k_resident, &[1, 0])?;
        let q6k_embedding = device.download_f32(&q6k_embedding_dev)?;
        let mut q6k_rows = vec![0.0f32; 1024];
        for row in 0..2 {
            let start = row * 2 * DType::Q6_K.block_bytes();
            xrt_kernels::cpu::dequantize_q6_k_row(
                &q6k_matrix[start..start + 2 * DType::Q6_K.block_bytes()],
                &mut q6k_rows[row * 512..(row + 1) * 512],
            )?;
        }
        assert_close(&q6k_embedding[0..512], &q6k_rows[512..1024], 1e-4);
        assert_close(&q6k_embedding[512..1024], &q6k_rows[0..512], 1e-4);

        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn ptx_jit_log_reports_invalid_ptx() -> Result<()> {
        let _device = CudaDevice::new(0)?;
        let message = ptx_jit_error_log(
            ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry broken() {\n    invalid.op;\n    ret;\n}\n",
        )
        .expect("invalid PTX should produce a CUDA JIT log");
        assert!(
            !message.trim().is_empty(),
            "expected CUDA JIT error log for invalid PTX"
        );
        Ok(())
    }
}
