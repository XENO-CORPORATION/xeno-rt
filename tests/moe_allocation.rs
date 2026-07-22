use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use xrt_models::moe::{
    group_route_slot_by_expert, route_top_k, MoeLayerDescriptor, MoeRoutingRow,
    MAX_SELECTED_EXPERTS,
};
use xrt_runtime::{build_moe_execution_plan, ExpertPlacementSnapshot, MoeWorkItem};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation delegates to the process system allocator without
// changing layout or pointer ownership. The counters are independent atomics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded with the exact layout received from the caller.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwarded with the exact pointer and layout from allocation.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarded with the exact pointer/layout and requested size.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn canonical_route_and_grouping_have_zero_steady_state_allocations() {
    const TOKENS: usize = 32;
    const EXPERTS: usize = 64;
    const TOP_K: usize = 8;

    let mut logits = [0.0f32; EXPERTS];
    let mut routes = [MoeRoutingRow::default(); TOKENS];
    let mut counts = [0usize; EXPERTS];
    let mut offsets = [0usize; EXPERTS + 1];
    let mut cursors = [0usize; EXPERTS];
    let mut token_indices = [0usize; TOKENS];
    assert!(TOP_K <= MAX_SELECTED_EXPERTS);

    for (token, route) in routes.iter_mut().enumerate() {
        for (expert, logit) in logits.iter_mut().enumerate() {
            *logit = ((token * 17 + expert * 29) % 101) as f32 * 0.03125;
        }
        route_top_k(&logits, TOP_K, route).unwrap();
    }
    group_route_slot_by_expert(
        &routes,
        0,
        EXPERTS,
        &mut counts,
        &mut offsets,
        &mut cursors,
        &mut token_indices,
    )
    .unwrap();

    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    for iteration in 0..256 {
        for (expert, logit) in logits.iter_mut().enumerate() {
            *logit = ((iteration * 11 + expert * 7) % 97) as f32 * 0.015625;
        }
        route_top_k(black_box(&logits), TOP_K, black_box(&mut routes[0])).unwrap();
        group_route_slot_by_expert(
            black_box(&routes),
            iteration % TOP_K,
            EXPERTS,
            black_box(&mut counts),
            black_box(&mut offsets),
            black_box(&mut cursors),
            black_box(&mut token_indices),
        )
        .unwrap();
    }
    COUNTING.store(false, Ordering::SeqCst);

    assert_eq!(
        ALLOCATIONS.load(Ordering::SeqCst),
        0,
        "route/dispatch planning allocated after scratch warmup"
    );
}

#[test]
fn placement_plan_has_zero_steady_state_allocations() {
    const TOKENS: usize = 16;
    const EXPERTS: usize = 8;
    const TOP_K: usize = 2;
    const WORK: usize = TOKENS * TOP_K;

    let layer = MoeLayerDescriptor::new(2, EXPERTS, TOP_K, 16, 32).unwrap();
    let placement = ExpertPlacementSnapshot::uniform(2, EXPERTS, 4, 7).unwrap();
    let mut routes = [MoeRoutingRow::default(); TOKENS];
    let mut logits = [0.0f32; EXPERTS];
    for (token, route) in routes.iter_mut().enumerate() {
        for (expert, logit) in logits.iter_mut().enumerate() {
            *logit = ((token * 11 + expert * 17) % 29) as f32;
        }
        route_top_k(&logits, TOP_K, route).unwrap();
    }
    let mut cpu = [MoeWorkItem::default(); WORK];
    let mut gpu = [MoeWorkItem::default(); WORK];

    let _ = build_moe_execution_plan(&layer, &placement, &routes, &mut cpu, &mut gpu).unwrap();
    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    for _ in 0..128 {
        let plan =
            build_moe_execution_plan(&layer, &placement, &routes, &mut cpu, &mut gpu).unwrap();
        black_box(plan);
    }
    COUNTING.store(false, Ordering::SeqCst);

    assert_eq!(
        ALLOCATIONS.load(Ordering::SeqCst),
        0,
        "steady-state placement planning unexpectedly allocated"
    );
}
