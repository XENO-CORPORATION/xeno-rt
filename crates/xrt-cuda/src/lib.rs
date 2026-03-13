use xrt_core::{Result, XrtError};

const CUDA_DISABLED_MESSAGE: &str =
    "CUDA backend requested but the xrt-cuda crate was built without the `cuda` feature";

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use cudarc::{
        driver::{CudaDevice as DriverCudaDevice, CudaFunction, LaunchAsync, LaunchConfig},
        nvrtc::Ptx,
    };
    use std::{fmt::Display, sync::Arc};
    use tracing::info;
    use xrt_core::checked_mul;

    const BLOCK_SIZE: u32 = 256;
    const MATMUL_TILE: u32 = 16;

    #[derive(Debug, Clone, Copy)]
    struct LoadedModules {
        rmsnorm: &'static str,
        rope: &'static str,
        softmax: &'static str,
        silu: &'static str,
        matmul: &'static str,
        add: &'static str,
        embed: &'static str,
    }

    const MODULES: LoadedModules = LoadedModules {
        rmsnorm: "xrt_cuda_rmsnorm",
        rope: "xrt_cuda_rope",
        softmax: "xrt_cuda_softmax",
        silu: "xrt_cuda_silu",
        matmul: "xrt_cuda_matmul",
        add: "xrt_cuda_add",
        embed: "xrt_cuda_embed",
    };

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
    .shared .align 4 .b8 reduce_buf[1024];
    .reg .pred %p<16>;
    .reg .f32 %f<20>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;

    ld.param.u64 %rd1, [rmsnorm_kernel_param_0];
    ld.param.u64 %rd2, [rmsnorm_kernel_param_1];
    ld.param.u64 %rd3, [rmsnorm_kernel_param_2];
    ld.param.u32 %r1, [rmsnorm_kernel_param_3];
    ld.param.u32 %r2, [rmsnorm_kernel_param_4];
    ld.param.f32 %f1, [rmsnorm_kernel_param_5];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;
    cvta.to.shared.u64 %rd7, reduce_buf;

    mov.u32 %r3, %ctaid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra RMS_DONE;

    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r6, %r3, %r2;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r7, %r4;

RMS_ACCUM:
    setp.ge.u32 %p2, %r7, %r2;
    @%p2 bra RMS_ACCUM_DONE;
    add.u32 %r8, %r6, %r7;
    mul.wide.u32 %rd8, %r8, 4;
    add.s64 %rd9, %rd4, %rd8;
    ld.global.f32 %f3, [%rd9];
    mul.f32 %f4, %f3, %f3;
    add.f32 %f2, %f2, %f4;
    add.u32 %r7, %r7, %r5;
    bra RMS_ACCUM;

RMS_ACCUM_DONE:
    mul.wide.u32 %rd10, %r4, 4;
    add.s64 %rd11, %rd7, %rd10;
    st.shared.f32 [%rd11], %f2;
    bar.sync 0;

    setp.ge.u32 %p3, %r4, 128;
    @%p3 bra RMS_REDUCE_128_DONE;
    add.u32 %r9, %r4, 128;
    mul.wide.u32 %rd12, %r9, 4;
    add.s64 %rd13, %rd7, %rd12;
    ld.shared.f32 %f5, [%rd11];
    ld.shared.f32 %f6, [%rd13];
    add.f32 %f7, %f5, %f6;
    st.shared.f32 [%rd11], %f7;
RMS_REDUCE_128_DONE:
    bar.sync 0;

    setp.ge.u32 %p4, %r4, 64;
    @%p4 bra RMS_REDUCE_64_DONE;
    add.u32 %r10, %r4, 64;
    mul.wide.u32 %rd14, %r10, 4;
    add.s64 %rd15, %rd7, %rd14;
    ld.shared.f32 %f8, [%rd11];
    ld.shared.f32 %f9, [%rd15];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd11], %f10;
RMS_REDUCE_64_DONE:
    bar.sync 0;

    setp.ge.u32 %p5, %r4, 32;
    @%p5 bra RMS_REDUCE_32_DONE;
    add.u32 %r11, %r4, 32;
    mul.wide.u32 %rd16, %r11, 4;
    add.s64 %rd17, %rd7, %rd16;
    ld.shared.f32 %f11, [%rd11];
    ld.shared.f32 %f12, [%rd17];
    add.f32 %f13, %f11, %f12;
    st.shared.f32 [%rd11], %f13;
RMS_REDUCE_32_DONE:
    bar.sync 0;

    setp.ge.u32 %p6, %r4, 16;
    @%p6 bra RMS_REDUCE_16_DONE;
    add.u32 %r12, %r4, 16;
    mul.wide.u32 %rd18, %r12, 4;
    add.s64 %rd19, %rd7, %rd18;
    ld.shared.f32 %f14, [%rd11];
    ld.shared.f32 %f15, [%rd19];
    add.f32 %f16, %f14, %f15;
    st.shared.f32 [%rd11], %f16;
RMS_REDUCE_16_DONE:
    bar.sync 0;

    setp.ge.u32 %p7, %r4, 8;
    @%p7 bra RMS_REDUCE_8_DONE;
    add.u32 %r13, %r4, 8;
    mul.wide.u32 %rd20, %r13, 4;
    add.s64 %rd21, %rd7, %rd20;
    ld.shared.f32 %f17, [%rd11];
    ld.shared.f32 %f18, [%rd21];
    add.f32 %f19, %f17, %f18;
    st.shared.f32 [%rd11], %f19;
RMS_REDUCE_8_DONE:
    bar.sync 0;

    setp.ge.u32 %p8, %r4, 4;
    @%p8 bra RMS_REDUCE_4_DONE;
    add.u32 %r14, %r4, 4;
    mul.wide.u32 %rd22, %r14, 4;
    add.s64 %rd23, %rd7, %rd22;
    ld.shared.f32 %f5, [%rd11];
    ld.shared.f32 %f6, [%rd23];
    add.f32 %f7, %f5, %f6;
    st.shared.f32 [%rd11], %f7;
RMS_REDUCE_4_DONE:
    bar.sync 0;

    setp.ge.u32 %p9, %r4, 2;
    @%p9 bra RMS_REDUCE_2_DONE;
    add.u32 %r15, %r4, 2;
    mul.wide.u32 %rd12, %r15, 4;
    add.s64 %rd13, %rd7, %rd12;
    ld.shared.f32 %f8, [%rd11];
    ld.shared.f32 %f9, [%rd13];
    add.f32 %f10, %f8, %f9;
    st.shared.f32 [%rd11], %f10;
RMS_REDUCE_2_DONE:
    bar.sync 0;

    setp.ge.u32 %p10, %r4, 1;
    @%p10 bra RMS_REDUCE_1_DONE;
    add.u32 %r16, %r4, 1;
    mul.wide.u32 %rd14, %r16, 4;
    add.s64 %rd15, %rd7, %rd14;
    ld.shared.f32 %f11, [%rd11];
    ld.shared.f32 %f12, [%rd15];
    add.f32 %f13, %f11, %f12;
    st.shared.f32 [%rd11], %f13;
RMS_REDUCE_1_DONE:
    bar.sync 0;

    ld.shared.f32 %f14, [%rd7];
    cvt.rn.f32.u32 %f15, %r2;
    div.rn.f32 %f16, %f14, %f15;
    add.f32 %f17, %f16, %f1;
    rsqrt.approx.f32 %f18, %f17;

    mov.u32 %r7, %r4;
RMS_WRITE:
    setp.ge.u32 %p11, %r7, %r2;
    @%p11 bra RMS_DONE;
    add.u32 %r8, %r6, %r7;
    mul.wide.u32 %rd8, %r8, 4;
    add.s64 %rd9, %rd4, %rd8;
    add.s64 %rd10, %rd6, %rd8;
    ld.global.f32 %f19, [%rd9];
    mul.wide.u32 %rd11, %r7, 4;
    add.s64 %rd12, %rd5, %rd11;
    ld.global.f32 %f2, [%rd12];
    mul.f32 %f3, %f19, %f18;
    mul.f32 %f4, %f3, %f2;
    st.global.f32 [%rd10], %f4;
    add.u32 %r7, %r7, %r5;
    bra RMS_WRITE;

RMS_DONE:
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
    .shared .align 4 .b8 tile_a[1024];
    .shared .align 4 .b8 tile_b[1024];
    .reg .pred %p<12>;
    .reg .f32 %f<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<24>;

    ld.param.u64 %rd1, [matmul_kernel_param_0];
    ld.param.u64 %rd2, [matmul_kernel_param_1];
    ld.param.u64 %rd3, [matmul_kernel_param_2];
    ld.param.u32 %r1, [matmul_kernel_param_3];
    ld.param.u32 %r2, [matmul_kernel_param_4];
    ld.param.u32 %r3, [matmul_kernel_param_5];

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;
    cvta.to.shared.u64 %rd7, tile_a;
    cvta.to.shared.u64 %rd8, tile_b;

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ctaid.y;
    mov.u32 %r6, %tid.x;
    mov.u32 %r7, %tid.y;

    mul.lo.u32 %r8, %r5, 16;
    add.u32 %r9, %r8, %r7;
    mul.lo.u32 %r10, %r4, 16;
    add.u32 %r11, %r10, %r6;
    mul.lo.u32 %r12, %r7, 16;
    add.u32 %r13, %r12, %r6;
    mul.wide.u32 %rd9, %r13, 4;
    add.s64 %rd10, %rd7, %rd9;
    add.s64 %rd11, %rd8, %rd9;

    add.u32 %r14, %r2, 15;
    shr.u32 %r15, %r14, 4;
    mov.u32 %r16, 0;
    mov.f32 %f1, 0f00000000;

MATMUL_TILE_LOOP:
    setp.ge.u32 %p1, %r16, %r15;
    @%p1 bra MATMUL_TILE_DONE;

    mul.lo.u32 %r17, %r16, 16;
    add.u32 %r18, %r17, %r6;
    setp.ge.u32 %p2, %r9, %r1;
    setp.ge.u32 %p3, %r18, %r2;
    or.pred %p4, %p2, %p3;
    @%p4 bra MATMUL_A_ZERO;
    mul.lo.u32 %r19, %r9, %r2;
    add.u32 %r20, %r19, %r18;
    mul.wide.u32 %rd12, %r20, 4;
    add.s64 %rd13, %rd4, %rd12;
    ld.global.f32 %f2, [%rd13];
    bra MATMUL_A_STORE;

MATMUL_A_ZERO:
    mov.f32 %f2, 0f00000000;

MATMUL_A_STORE:
    st.shared.f32 [%rd10], %f2;

    add.u32 %r21, %r17, %r7;
    setp.ge.u32 %p5, %r21, %r2;
    setp.ge.u32 %p6, %r11, %r3;
    or.pred %p7, %p5, %p6;
    @%p7 bra MATMUL_B_ZERO;
    mul.lo.u32 %r22, %r21, %r3;
    add.u32 %r23, %r22, %r11;
    mul.wide.u32 %rd14, %r23, 4;
    add.s64 %rd15, %rd5, %rd14;
    ld.global.f32 %f3, [%rd15];
    bra MATMUL_B_STORE;

MATMUL_B_ZERO:
    mov.f32 %f3, 0f00000000;

MATMUL_B_STORE:
    st.shared.f32 [%rd11], %f3;
    bar.sync 0;

    mov.u32 %r24, 0;
MATMUL_INNER_LOOP:
    setp.ge.u32 %p8, %r24, 16;
    @%p8 bra MATMUL_INNER_DONE;

    add.u32 %r25, %r12, %r24;
    mul.wide.u32 %rd16, %r25, 4;
    add.s64 %rd17, %rd7, %rd16;
    ld.shared.f32 %f4, [%rd17];

    mul.lo.u32 %r26, %r24, 16;
    add.u32 %r27, %r26, %r6;
    mul.wide.u32 %rd18, %r27, 4;
    add.s64 %rd19, %rd8, %rd18;
    ld.shared.f32 %f5, [%rd19];

    fma.rn.f32 %f1, %f4, %f5, %f1;
    add.u32 %r24, %r24, 1;
    bra MATMUL_INNER_LOOP;

MATMUL_INNER_DONE:
    bar.sync 0;
    add.u32 %r16, %r16, 1;
    bra MATMUL_TILE_LOOP;

MATMUL_TILE_DONE:
    setp.ge.u32 %p9, %r9, %r1;
    setp.ge.u32 %p10, %r11, %r3;
    or.pred %p11, %p9, %p10;
    @%p11 bra MATMUL_DONE;

    mul.lo.u32 %r28, %r9, %r3;
    add.u32 %r29, %r28, %r11;
    mul.wide.u32 %rd20, %r29, 4;
    add.s64 %rd21, %rd6, %rd20;
    st.global.f32 [%rd21], %f1;

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
    .reg .b64 %rd<7>;

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
"#;

    fn cuda_error(context: &str, err: impl Display) -> XrtError {
        XrtError::Cuda(format!("{context}: {err}"))
    }

    fn to_u32(value: usize, what: &str) -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| XrtError::Shape(format!("{what} {value} exceeds CUDA u32 limits")))
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

    fn load_module(
        device: &Arc<DriverCudaDevice>,
        module_name: &'static str,
        ptx: &'static str,
        functions: &[&'static str],
    ) -> Result<()> {
        device
            .load_ptx(Ptx::from_src(ptx), module_name, functions)
            .map_err(|err| cuda_error(&format!("failed to load PTX module `{module_name}`"), err))
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

    impl CudaDevice {
        pub fn new(ordinal: usize) -> Result<Self> {
            let device = DriverCudaDevice::new(ordinal).map_err(|err| {
                XrtError::Cuda(format!("failed to open CUDA device {ordinal}: {err}"))
            })?;

            load_module(&device, MODULES.rmsnorm, RMSNORM_PTX, &["rmsnorm_kernel"])?;
            load_module(&device, MODULES.rope, ROPE_PTX, &["rope_kernel"])?;
            load_module(&device, MODULES.softmax, SOFTMAX_PTX, &["softmax_kernel"])?;
            load_module(&device, MODULES.silu, SILU_PTX, &["silu_kernel"])?;
            load_module(&device, MODULES.matmul, MATMUL_PTX, &["matmul_kernel"])?;
            load_module(&device, MODULES.add, ADD_PTX, &["elementwise_add_kernel"])?;
            load_module(&device, MODULES.embed, EMBEDDING_PTX, &["embedding_kernel"])?;

            info!("initialized CUDA backend on device {}", ordinal);
            Ok(Self {
                device,
                modules: MODULES,
            })
        }

        pub fn inner(&self) -> &Arc<DriverCudaDevice> {
            &self.device
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

            let rotary_width = rope_dim.min(head_dim);
            let half_width = rotary_width / 2;
            if half_width == 0 {
                return Ok(tensor.to_vec());
            }

            let total_pairs = checked_mul(n_heads, half_width, "rope pair count")?;
            let n_heads_u32 = to_u32(n_heads, "rope head count")?;
            let head_dim_u32 = to_u32(head_dim, "rope head dimension")?;
            let position_u32 = to_u32(position, "rope position")?;
            let rotary_width_u32 = to_u32(rotary_width, "rope dimension")?;
            let total_pairs_u32 = to_u32(total_pairs, "rope work items")?;

            let mut tensor_dev = self
                .device
                .htod_copy(tensor.to_vec())
                .map_err(|err| cuda_error("failed to copy rope tensor to device", err))?;
            let func = self.function(self.modules.rope, "rope_kernel")?;
            unsafe {
                func.launch(
                    one_dim_launch(total_pairs_u32),
                    (
                        &mut tensor_dev,
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

            self.device
                .sync_reclaim(tensor_dev)
                .map_err(|err| cuda_error("failed to reclaim rope tensor", err))
        }

        pub fn softmax(&self, values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
            let expected = checked_mul(rows, cols, "softmax elements")?;
            expect_len(values.len(), expected, "softmax input")?;
            if expected == 0 {
                return Ok(values.to_vec());
            }

            let rows_u32 = to_u32(rows, "softmax rows")?;
            let cols_u32 = to_u32(cols, "softmax cols")?;
            let mut values_dev = self
                .device
                .htod_copy(values.to_vec())
                .map_err(|err| cuda_error("failed to copy softmax input to device", err))?;
            let func = self.function(self.modules.softmax, "softmax_kernel")?;
            unsafe { func.launch(row_launch(rows_u32), (&mut values_dev, rows_u32, cols_u32)) }
                .map_err(|err| cuda_error("failed to launch softmax kernel", err))?;

            self.device
                .sync_reclaim(values_dev)
                .map_err(|err| cuda_error("failed to reclaim softmax output", err))
        }

        pub fn silu(&self, values: &[f32]) -> Result<Vec<f32>> {
            if values.is_empty() {
                return Ok(Vec::new());
            }

            let n_u32 = to_u32(values.len(), "silu element count")?;
            let mut values_dev = self
                .device
                .htod_copy(values.to_vec())
                .map_err(|err| cuda_error("failed to copy silu input to device", err))?;
            let func = self.function(self.modules.silu, "silu_kernel")?;
            unsafe { func.launch(one_dim_launch(n_u32), (&mut values_dev, n_u32)) }
                .map_err(|err| cuda_error("failed to launch silu kernel", err))?;

            self.device
                .sync_reclaim(values_dev)
                .map_err(|err| cuda_error("failed to reclaim silu output", err))
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

        pub fn add(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
            if lhs.len() != rhs.len() {
                return Err(XrtError::Shape(format!(
                    "add inputs must have identical lengths, found {} and {}",
                    lhs.len(),
                    rhs.len()
                )));
            }
            if lhs.is_empty() {
                return Ok(Vec::new());
            }

            let n_u32 = to_u32(lhs.len(), "add element count")?;
            let mut dst_dev = self
                .device
                .htod_copy(lhs.to_vec())
                .map_err(|err| cuda_error("failed to copy add lhs to device", err))?;
            let src_dev = self
                .device
                .htod_copy(rhs.to_vec())
                .map_err(|err| cuda_error("failed to copy add rhs to device", err))?;

            let func = self.function(self.modules.add, "elementwise_add_kernel")?;
            unsafe { func.launch(one_dim_launch(n_u32), (&mut dst_dev, &src_dev, n_u32)) }
                .map_err(|err| cuda_error("failed to launch add kernel", err))?;

            self.device
                .sync_reclaim(dst_dev)
                .map_err(|err| cuda_error("failed to reclaim add output", err))
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

        fn function(&self, module_name: &str, function_name: &str) -> Result<CudaFunction> {
            self.device
                .get_func(module_name, function_name)
                .ok_or_else(|| {
                    XrtError::Cuda(format!(
                        "failed to fetch kernel `{function_name}` from module `{module_name}`"
                    ))
                })
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{CudaBackend, CudaDevice};

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaDevice;

#[cfg(not(feature = "cuda"))]
pub type CudaBackend = CudaDevice;

#[cfg(not(feature = "cuda"))]
impl CudaDevice {
    pub fn new(_ordinal: usize) -> Result<Self> {
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

    pub fn softmax(&self, _values: &[f32], _rows: usize, _cols: usize) -> Result<Vec<f32>> {
        Err(XrtError::Cuda(CUDA_DISABLED_MESSAGE.to_string()))
    }

    pub fn silu(&self, _values: &[f32]) -> Result<Vec<f32>> {
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

    pub fn add(&self, _lhs: &[f32], _rhs: &[f32]) -> Result<Vec<f32>> {
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
}
