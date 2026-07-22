use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "moe-route-trace")]
use parking_lot::Mutex;
use xrt_core::{Result, XrtError};

pub const MAX_SELECTED_EXPERTS: usize = 32;
/// Router logits closer than this at the top-k boundary are numerically tied.
///
/// This narrow compatibility band stabilizes effectively equal top-k boundary
/// logits without coarsening ordinary finite routing decisions.
#[cfg(not(feature = "moe-router-exact-reference"))]
pub const MOE_ROUTER_TIE_EPSILON: f32 = 1.0e-5;
/// Feature-gated scalar semantic control used only by opt-in quality tests.
#[cfg(feature = "moe-router-exact-reference")]
pub const MOE_ROUTER_TIE_EPSILON: f32 = 0.0;

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum MoeCpuExecution {
    #[default]
    Legacy,
    Optimized,
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct MoeTelemetrySnapshot {
    pub routed_tokens: u64,
    pub selected_expert_calls: u64,
    pub legacy_batches: u64,
    pub grouped_batches: u64,
    pub grouped_tokens: u64,
    pub worker_failures: u64,
    pub expert_call_counts: Vec<u64>,
}

#[cfg(feature = "moe-route-trace")]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MoeRouteTraceEntry {
    layer_index: usize,
    logical_ids: [u32; MAX_SELECTED_EXPERTS],
    boundary_selected_id: u32,
    boundary_selected_logit_bits: u32,
    best_excluded_id: u32,
    best_excluded_logit_bits: u32,
    len: u8,
}

#[cfg(feature = "moe-route-trace")]
impl MoeRouteTraceEntry {
    fn new(layer_index: usize, route: &MoeRoutingRow) -> Self {
        let mut logical_ids = [0; MAX_SELECTED_EXPERTS];
        logical_ids[..route.len()].copy_from_slice(route.logical_ids());
        Self {
            layer_index,
            logical_ids,
            boundary_selected_id: route.boundary_selected_id,
            boundary_selected_logit_bits: route.boundary_selected_logit.to_bits(),
            best_excluded_id: route.best_excluded_id,
            best_excluded_logit_bits: route.best_excluded_logit.to_bits(),
            len: route.len() as u8,
        }
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn logical_ids(&self) -> &[u32] {
        &self.logical_ids[..usize::from(self.len)]
    }

    pub fn boundary_diagnostic(&self) -> (u32, f32, u32, f32) {
        (
            self.boundary_selected_id,
            f32::from_bits(self.boundary_selected_logit_bits),
            self.best_excluded_id,
            f32::from_bits(self.best_excluded_logit_bits),
        )
    }
}

#[cfg(feature = "moe-route-trace")]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MoeRouteTrace {
    max_entries: usize,
    overflowed: bool,
    entries: Vec<MoeRouteTraceEntry>,
}

#[cfg(feature = "moe-route-trace")]
impl MoeRouteTrace {
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    pub fn entries(&self) -> &[MoeRouteTraceEntry] {
        &self.entries
    }
}

#[cfg(feature = "moe-route-trace")]
#[derive(Debug)]
struct MoeRouteTraceState {
    max_entries: usize,
    overflowed: bool,
    entries: Vec<MoeRouteTraceEntry>,
}

#[derive(Debug)]
pub(crate) struct MoeTelemetry {
    routed_tokens: AtomicU64,
    selected_expert_calls: AtomicU64,
    legacy_batches: AtomicU64,
    grouped_batches: AtomicU64,
    grouped_tokens: AtomicU64,
    worker_failures: AtomicU64,
    expert_call_counts: Box<[AtomicU64]>,
    #[cfg(feature = "moe-route-trace")]
    route_trace: Mutex<Option<MoeRouteTraceState>>,
}

impl MoeTelemetry {
    pub(crate) fn new(expert_count: usize) -> Self {
        Self {
            routed_tokens: AtomicU64::new(0),
            selected_expert_calls: AtomicU64::new(0),
            legacy_batches: AtomicU64::new(0),
            grouped_batches: AtomicU64::new(0),
            grouped_tokens: AtomicU64::new(0),
            worker_failures: AtomicU64::new(0),
            expert_call_counts: (0..expert_count)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            #[cfg(feature = "moe-route-trace")]
            route_trace: Mutex::new(None),
        }
    }

    pub(crate) fn record_route(&self, route: &MoeRoutingRow) {
        self.routed_tokens.fetch_add(1, Ordering::Relaxed);
        self.selected_expert_calls
            .fetch_add(route.len() as u64, Ordering::Relaxed);
        for &expert_id in route.logical_ids() {
            if let Some(counter) = self.expert_call_counts.get(expert_id as usize) {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[cfg(feature = "moe-route-trace")]
    pub(crate) fn start_route_trace(&self, max_entries: usize) -> Result<()> {
        if max_entries == 0 {
            return Err(XrtError::InvalidTensor(
                "MoE route trace capacity must be greater than zero".to_string(),
            ));
        }
        *self.route_trace.lock() = Some(MoeRouteTraceState {
            max_entries,
            overflowed: false,
            entries: Vec::with_capacity(max_entries.min(4096)),
        });
        Ok(())
    }

    #[cfg(feature = "moe-route-trace")]
    pub(crate) fn record_route_trace(&self, layer_index: usize, route: &MoeRoutingRow) {
        let mut trace = self.route_trace.lock();
        let Some(trace) = trace.as_mut() else {
            return;
        };
        if trace.entries.len() >= trace.max_entries {
            trace.overflowed = true;
            return;
        }
        trace
            .entries
            .push(MoeRouteTraceEntry::new(layer_index, route));
    }

    #[cfg(feature = "moe-route-trace")]
    pub(crate) fn take_route_trace(&self) -> Option<MoeRouteTrace> {
        self.route_trace.lock().take().map(|trace| MoeRouteTrace {
            max_entries: trace.max_entries,
            overflowed: trace.overflowed,
            entries: trace.entries,
        })
    }

    pub(crate) fn record_legacy_batch(&self) {
        self.legacy_batches.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_grouped_batch(&self, tokens: usize) {
        self.grouped_batches.fetch_add(1, Ordering::Relaxed);
        self.grouped_tokens
            .fetch_add(tokens as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_worker_failure(&self) {
        self.worker_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> MoeTelemetrySnapshot {
        MoeTelemetrySnapshot {
            routed_tokens: self.routed_tokens.load(Ordering::Relaxed),
            selected_expert_calls: self.selected_expert_calls.load(Ordering::Relaxed),
            legacy_batches: self.legacy_batches.load(Ordering::Relaxed),
            grouped_batches: self.grouped_batches.load(Ordering::Relaxed),
            grouped_tokens: self.grouped_tokens.load(Ordering::Relaxed),
            worker_failures: self.worker_failures.load(Ordering::Relaxed),
            expert_call_counts: self
                .expert_call_counts
                .iter()
                .map(|counter| counter.load(Ordering::Relaxed))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeLayerDescriptor {
    layer_index: usize,
    expert_count: usize,
    selected_per_token: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

impl MoeLayerDescriptor {
    pub fn new(
        layer_index: usize,
        expert_count: usize,
        selected_per_token: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        if expert_count == 0
            || selected_per_token == 0
            || hidden_size == 0
            || intermediate_size == 0
        {
            return Err(XrtError::InvalidMetadata(
                "MoE layer dimensions and expert counts must be non-zero".to_string(),
            ));
        }
        if selected_per_token > expert_count {
            return Err(XrtError::InvalidMetadata(format!(
                "MoE layer {layer_index} selects {selected_per_token} experts from only {expert_count}"
            )));
        }
        if selected_per_token > MAX_SELECTED_EXPERTS {
            return Err(XrtError::Unsupported(format!(
                "MoE layer {layer_index} selects {selected_per_token} experts, exceeding the fixed routing capacity of {MAX_SELECTED_EXPERTS}"
            )));
        }
        if expert_count > u32::MAX as usize {
            return Err(XrtError::Unsupported(format!(
                "MoE layer {layer_index} has {expert_count} experts, exceeding the logical expert ID width"
            )));
        }
        Ok(Self {
            layer_index,
            expert_count,
            selected_per_token,
            hidden_size,
            intermediate_size,
        })
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn expert_count(&self) -> usize {
        self.expert_count
    }

    pub fn selected_per_token(&self) -> usize {
        self.selected_per_token
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeRoutingRow {
    logical_ids: [u32; MAX_SELECTED_EXPERTS],
    weights: [f32; MAX_SELECTED_EXPERTS],
    len: usize,
    #[cfg(feature = "moe-route-trace")]
    boundary_selected_id: u32,
    #[cfg(feature = "moe-route-trace")]
    boundary_selected_logit: f32,
    #[cfg(feature = "moe-route-trace")]
    best_excluded_id: u32,
    #[cfg(feature = "moe-route-trace")]
    best_excluded_logit: f32,
}

impl Default for MoeRoutingRow {
    fn default() -> Self {
        Self {
            logical_ids: [0; MAX_SELECTED_EXPERTS],
            weights: [0.0; MAX_SELECTED_EXPERTS],
            len: 0,
            #[cfg(feature = "moe-route-trace")]
            boundary_selected_id: u32::MAX,
            #[cfg(feature = "moe-route-trace")]
            boundary_selected_logit: f32::NEG_INFINITY,
            #[cfg(feature = "moe-route-trace")]
            best_excluded_id: u32::MAX,
            #[cfg(feature = "moe-route-trace")]
            best_excluded_logit: f32::NEG_INFINITY,
        }
    }
}

impl MoeRoutingRow {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn logical_ids(&self) -> &[u32] {
        &self.logical_ids[..self.len]
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights[..self.len]
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (usize, f32)> + '_ {
        self.logical_ids()
            .iter()
            .copied()
            .zip(self.weights().iter().copied())
            .map(|(id, weight)| (id as usize, weight))
    }
}

pub fn route_top_k(
    logits: &[f32],
    selected_per_token: usize,
    output: &mut MoeRoutingRow,
) -> Result<()> {
    if selected_per_token == 0 || selected_per_token > logits.len() {
        return Err(XrtError::InvalidTensor(format!(
            "MoE router top-k {selected_per_token} is invalid for {} logits",
            logits.len()
        )));
    }
    if selected_per_token > MAX_SELECTED_EXPERTS {
        return Err(XrtError::Unsupported(format!(
            "MoE router top-k {selected_per_token} exceeds fixed capacity {MAX_SELECTED_EXPERTS}"
        )));
    }
    if logits.len() > u32::MAX as usize {
        return Err(XrtError::Unsupported(format!(
            "MoE router has {} logits, exceeding logical expert ID width",
            logits.len()
        )));
    }

    output.len = 0;
    for (logical_id, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            continue;
        }
        let mut insertion = output.len;
        for index in 0..output.len {
            let existing_logit = output.weights[index];
            let existing_id = output.logical_ids[index] as usize;
            if logit > existing_logit || (logit == existing_logit && logical_id < existing_id) {
                insertion = index;
                break;
            }
        }
        if insertion >= selected_per_token {
            continue;
        }
        let next_len = (output.len + 1).min(selected_per_token);
        for index in (insertion + 1..next_len).rev() {
            output.weights[index] = output.weights[index - 1];
            output.logical_ids[index] = output.logical_ids[index - 1];
        }
        output.weights[insertion] = logit;
        output.logical_ids[insertion] = logical_id as u32;
        output.len = next_len;
    }

    if output.len < selected_per_token {
        output.len = 0;
        return Err(XrtError::Runtime(format!(
            "MoE router produced only {} finite candidates for top-k {selected_per_token}",
            logits.iter().filter(|value| value.is_finite()).count()
        )));
    }

    // Canonicalize only the exact top-k boundary. Values clearly above the
    // band retain logit order; boundary candidates are treated as tied and the
    // largest logical IDs fill the remaining slots. Scanning IDs in descending
    // order keeps this allocation-free and transitive even when many logits are
    // equal or form a chain of sub-epsilon differences.
    let boundary = output.weights[selected_per_token - 1];
    let strict_len = output.weights[..selected_per_token]
        .iter()
        .take_while(|&&logit| logit > boundary + MOE_ROUTER_TIE_EPSILON)
        .count();
    output.len = strict_len;
    for logical_id in (0..logits.len()).rev() {
        if output.len >= selected_per_token {
            break;
        }
        let logit = logits[logical_id];
        if logit.is_finite()
            && logit >= boundary - MOE_ROUTER_TIE_EPSILON
            && logit <= boundary + MOE_ROUTER_TIE_EPSILON
        {
            output.logical_ids[output.len] = logical_id as u32;
            output.weights[output.len] = logit;
            output.len += 1;
        }
    }
    if output.len < selected_per_token {
        output.len = 0;
        return Err(XrtError::Runtime(
            "MoE router boundary tie canonicalization produced too few candidates".to_string(),
        ));
    }

    #[cfg(feature = "moe-route-trace")]
    {
        let mut boundary_selected_id = u32::MAX;
        let mut boundary_selected_logit = f32::INFINITY;
        for index in 0..output.len {
            let logical_id = output.logical_ids[index];
            let logit = output.weights[index];
            if logit < boundary_selected_logit
                || (logit == boundary_selected_logit && logical_id < boundary_selected_id)
            {
                boundary_selected_id = logical_id;
                boundary_selected_logit = logit;
            }
        }

        let mut best_excluded_id = u32::MAX;
        let mut best_excluded_logit = f32::NEG_INFINITY;
        for (logical_id, &logit) in logits.iter().enumerate() {
            let logical_id = logical_id as u32;
            if !logit.is_finite() || output.logical_ids[..output.len].contains(&logical_id) {
                continue;
            }
            if logit > best_excluded_logit
                || (logit == best_excluded_logit && logical_id < best_excluded_id)
            {
                best_excluded_id = logical_id;
                best_excluded_logit = logit;
            }
        }

        output.boundary_selected_id = boundary_selected_id;
        output.boundary_selected_logit = boundary_selected_logit;
        output.best_excluded_id = best_excluded_id;
        output.best_excluded_logit = best_excluded_logit;
    }

    let max_logit = output.weights[0];
    let mut sum = 0.0f32;
    for weight in &mut output.weights[..output.len] {
        *weight = (*weight - max_logit).exp();
        sum += *weight;
    }
    if !sum.is_finite() || sum <= 0.0 {
        output.len = 0;
        return Err(XrtError::Runtime(
            "MoE router selected logits could not be normalized".to_string(),
        ));
    }
    let inverse_sum = sum.recip();
    for weight in &mut output.weights[..output.len] {
        *weight *= inverse_sum;
    }
    Ok(())
}

/// Group one canonical route slot by logical expert into caller-owned scratch.
///
/// `token_indices[offsets[e]..offsets[e + 1]]` contains the token rows assigned
/// to expert `e`. All slices are reused by the caller, so successful steady
/// state dispatch performs no heap allocation.
pub fn group_route_slot_by_expert(
    routes: &[MoeRoutingRow],
    route_slot: usize,
    expert_count: usize,
    counts: &mut [usize],
    offsets: &mut [usize],
    cursors: &mut [usize],
    token_indices: &mut [usize],
) -> Result<()> {
    if expert_count == 0
        || counts.len() < expert_count
        || offsets.len() < expert_count.saturating_add(1)
        || cursors.len() < expert_count
        || token_indices.len() < routes.len()
    {
        return Err(XrtError::InvalidTensor(
            "MoE grouping scratch does not match route geometry".to_string(),
        ));
    }

    counts[..expert_count].fill(0);
    for route in routes {
        let expert_id = route
            .logical_ids()
            .get(route_slot)
            .copied()
            .ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "MoE route has {} selections but route slot {route_slot} was requested",
                    route.len()
                ))
            })? as usize;
        let count = counts.get_mut(expert_id).ok_or_else(|| {
            XrtError::InvalidTensor(format!(
                "MoE route selected logical expert {expert_id} outside expert count {expert_count}"
            ))
        })?;
        *count = count
            .checked_add(1)
            .ok_or_else(|| XrtError::Runtime("MoE expert token count overflowed".to_string()))?;
    }

    offsets[0] = 0;
    for expert in 0..expert_count {
        offsets[expert + 1] = offsets[expert]
            .checked_add(counts[expert])
            .ok_or_else(|| XrtError::Runtime("MoE grouping offset overflowed".to_string()))?;
    }
    cursors[..expert_count].copy_from_slice(&offsets[..expert_count]);
    for (token, route) in routes.iter().enumerate() {
        let expert_id = route.logical_ids()[route_slot] as usize;
        let cursor = &mut cursors[expert_id];
        token_indices[*cursor] = token;
        *cursor += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "moe-route-trace")]
    use super::MoeTelemetry;
    use super::{
        group_route_slot_by_expert, route_top_k, MoeLayerDescriptor, MoeRoutingRow,
        MAX_SELECTED_EXPERTS,
    };

    #[test]
    fn canonical_router_orders_by_logit_then_logical_id() {
        let mut row = MoeRoutingRow::default();
        route_top_k(&[0.5, 2.0, 2.0, -1.0, 1.5], 3, &mut row).unwrap();
        assert_eq!(row.logical_ids(), &[1, 2, 4]);
        assert!((row.weights().iter().sum::<f32>() - 1.0).abs() <= 1e-6);
        assert_eq!(row.weights()[0], row.weights()[1]);
        assert!(row.weights()[1] > row.weights()[2]);
    }

    #[test]
    #[cfg(not(feature = "moe-router-exact-reference"))]
    fn canonical_router_resolves_boundary_near_ties_by_logical_id() {
        let mut row = MoeRoutingRow::default();
        route_top_k(&[2.0, 1.0, 0.500_006, 0.500_000, -1.0], 3, &mut row).unwrap();
        assert_eq!(row.logical_ids(), &[0, 1, 3]);
        assert_eq!(row.len(), 3);
        assert!((row.weights().iter().sum::<f32>() - 1.0).abs() <= 1e-6);

        route_top_k(&[2.0, 1.0, 0.500_02, 0.5, -1.0], 3, &mut row).unwrap();
        assert_eq!(row.logical_ids(), &[0, 1, 2]);
    }

    #[test]
    #[cfg(feature = "moe-router-exact-reference")]
    fn exact_reference_router_preserves_finite_logit_order() {
        let mut row = MoeRoutingRow::default();
        route_top_k(&[2.0, 1.0, 0.500_006, 0.500_000, -1.0], 3, &mut row).unwrap();
        assert_eq!(row.logical_ids(), &[0, 1, 2]);
    }

    #[cfg(feature = "moe-route-trace")]
    #[test]
    fn bounded_route_trace_records_layer_and_logical_ids() {
        let telemetry = MoeTelemetry::new(8);
        telemetry.start_route_trace(1).unwrap();
        let mut first = MoeRoutingRow::default();
        route_top_k(&[1.5, 0.5, 2.0, -1.0], 2, &mut first).unwrap();
        assert_eq!(first.logical_ids(), &[2, 0]);
        telemetry.record_route_trace(7, &first);
        telemetry.record_route_trace(8, &first);

        let trace = telemetry.take_route_trace().unwrap();
        assert_eq!(trace.max_entries(), 1);
        assert!(trace.overflowed());
        assert_eq!(trace.entries().len(), 1);
        assert_eq!(trace.entries()[0].layer_index(), 7);
        assert_eq!(trace.entries()[0].logical_ids(), &[2, 0]);
        assert!(telemetry.take_route_trace().is_none());
    }

    #[test]
    fn canonical_router_rejects_insufficient_finite_candidates() {
        let mut row = MoeRoutingRow::default();
        assert!(route_top_k(&[f32::NAN, f32::INFINITY], 1, &mut row).is_err());
        assert!(row.is_empty());
        assert!(route_top_k(&[1.0, f32::NEG_INFINITY], 2, &mut row).is_err());
    }

    #[test]
    fn fixed_insertion_router_matches_full_sort_oracle() {
        let mut seed = 0xC0FF_EE12_3456_789Au64;
        for expert_count in [2usize, 4, 8, 16, 64] {
            for top_k in 1..=expert_count.min(8) {
                for _ in 0..64 {
                    let mut logits = Vec::with_capacity(expert_count);
                    for expert_id in 0..expert_count {
                        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        logits.push(
                            ((seed >> 40) as i32 % 23) as f32 * 0.125 + expert_id as f32 * 0.001,
                        );
                    }
                    let mut oracle = logits.iter().copied().enumerate().collect::<Vec<_>>();
                    oracle.sort_by(|(left_id, left), (right_id, right)| {
                        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
                    });
                    oracle.truncate(top_k);

                    let mut row = MoeRoutingRow::default();
                    route_top_k(&logits, top_k, &mut row).unwrap();
                    assert_eq!(
                        row.logical_ids(),
                        oracle.iter().map(|(id, _)| *id as u32).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn descriptor_rejects_invalid_top_k_and_fixed_capacity_overflow() {
        assert!(MoeLayerDescriptor::new(0, 4, 5, 8, 16).is_err());
        assert!(MoeLayerDescriptor::new(
            0,
            MAX_SELECTED_EXPERTS + 1,
            MAX_SELECTED_EXPERTS + 1,
            8,
            16
        )
        .is_err());
    }

    #[test]
    fn grouping_is_stable_by_token_within_each_expert() {
        let mut routes = [MoeRoutingRow::default(); 4];
        for (route, logits) in routes.iter_mut().zip([
            [4.0, 1.0, 3.0],
            [1.0, 5.0, 4.0],
            [3.0, 2.0, 4.0],
            [2.0, 5.0, 1.0],
        ]) {
            route_top_k(&logits, 2, route).unwrap();
        }
        let mut counts = [0usize; 3];
        let mut offsets = [0usize; 4];
        let mut cursors = [0usize; 3];
        let mut tokens = [usize::MAX; 4];
        group_route_slot_by_expert(
            &routes,
            0,
            3,
            &mut counts,
            &mut offsets,
            &mut cursors,
            &mut tokens,
        )
        .unwrap();
        assert_eq!(counts, [1, 2, 1]);
        assert_eq!(offsets, [0, 1, 3, 4]);
        assert_eq!(tokens, [0, 1, 3, 2]);
    }
}
