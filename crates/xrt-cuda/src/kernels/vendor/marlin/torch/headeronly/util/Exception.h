#pragma once

// scalar_type.hpp uses this only for host-side construction invariants. The
// XRT kernel instantiation uses compile-time constants, so no Torch runtime is
// linked into xrt-cuda.
#define STD_TORCH_CHECK(condition, ...) ((void)0)
