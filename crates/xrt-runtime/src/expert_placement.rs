use std::sync::Arc;

use parking_lot::RwLock;
use xrt_core::{Result, XrtError};

/// Immutable logical-expert to physical-GPU-slot mapping for one MoE layer.
///
/// Logical expert IDs are the model-visible identity. GPU slots are an
/// execution detail and are deliberately exposed through a distinct type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertPlacementSnapshot {
    layer_index: usize,
    generation: u64,
    logical_to_gpu_slot: Box<[Option<u16>]>,
    gpu_slot_to_logical: Box<[u16]>,
}

impl ExpertPlacementSnapshot {
    /// Build a validated snapshot. The order of `gpu_logical_experts` defines
    /// physical slot order: element zero occupies slot zero, and so on.
    pub fn from_gpu_experts(
        layer_index: usize,
        expert_count: usize,
        generation: u64,
        gpu_logical_experts: &[usize],
    ) -> Result<Self> {
        if generation == 0 {
            return Err(XrtError::Runtime(
                "MoE placement generation zero is reserved for no active placement".to_string(),
            ));
        }
        if expert_count == 0 {
            return Err(XrtError::InvalidMetadata(format!(
                "MoE layer {layer_index} has no logical experts"
            )));
        }
        if expert_count > usize::from(u16::MAX) + 1 {
            return Err(XrtError::Unsupported(format!(
                "MoE layer {layer_index} has {expert_count} experts, exceeding the checked u16 placement-map width"
            )));
        }
        if gpu_logical_experts.len() > expert_count {
            return Err(XrtError::InvalidMetadata(format!(
                "MoE layer {layer_index} assigns {} GPU slots for only {expert_count} logical experts",
                gpu_logical_experts.len()
            )));
        }

        let mut logical_to_gpu_slot = vec![None; expert_count];
        let mut gpu_slot_to_logical = Vec::with_capacity(gpu_logical_experts.len());
        for (slot, &logical_expert) in gpu_logical_experts.iter().enumerate() {
            if logical_expert >= expert_count {
                return Err(XrtError::InvalidMetadata(format!(
                    "MoE layer {layer_index} places logical expert {logical_expert}, outside 0..{expert_count}"
                )));
            }
            if logical_to_gpu_slot[logical_expert].is_some() {
                return Err(XrtError::InvalidMetadata(format!(
                    "MoE layer {layer_index} places logical expert {logical_expert} more than once"
                )));
            }
            let slot = u16::try_from(slot).map_err(|_| {
                XrtError::Unsupported(format!(
                    "MoE layer {layer_index} GPU slot index exceeds u16"
                ))
            })?;
            let logical_expert_u16 = u16::try_from(logical_expert).map_err(|_| {
                XrtError::Unsupported(format!(
                    "MoE layer {layer_index} logical expert ID exceeds u16"
                ))
            })?;
            logical_to_gpu_slot[logical_expert] = Some(slot);
            gpu_slot_to_logical.push(logical_expert_u16);
        }

        Ok(Self {
            layer_index,
            generation,
            logical_to_gpu_slot: logical_to_gpu_slot.into_boxed_slice(),
            gpu_slot_to_logical: gpu_slot_to_logical.into_boxed_slice(),
        })
    }

    /// Select evenly spaced logical experts for a deterministic static layout.
    pub fn uniform(
        layer_index: usize,
        expert_count: usize,
        gpu_slot_count: usize,
        generation: u64,
    ) -> Result<Self> {
        if gpu_slot_count > expert_count {
            return Err(XrtError::InvalidMetadata(format!(
                "MoE layer {layer_index} requests {gpu_slot_count} uniform GPU slots for only {expert_count} experts"
            )));
        }
        let mut gpu_logical_experts = Vec::with_capacity(gpu_slot_count);
        if gpu_slot_count > 0 {
            for slot in 0..gpu_slot_count {
                gpu_logical_experts.push(slot * expert_count / gpu_slot_count);
            }
        }
        Self::from_gpu_experts(layer_index, expert_count, generation, &gpu_logical_experts)
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn expert_count(&self) -> usize {
        self.logical_to_gpu_slot.len()
    }

    pub fn gpu_slot_count(&self) -> usize {
        self.gpu_slot_to_logical.len()
    }

    pub fn gpu_slot_for(&self, logical_expert: usize) -> Option<u16> {
        self.logical_to_gpu_slot
            .get(logical_expert)
            .copied()
            .flatten()
    }

    pub fn logical_expert_for(&self, gpu_slot: u16) -> Option<u16> {
        self.gpu_slot_to_logical.get(usize::from(gpu_slot)).copied()
    }

    pub fn logical_to_gpu_slots(&self) -> &[Option<u16>] {
        &self.logical_to_gpu_slot
    }

    pub fn gpu_slots_to_logical(&self) -> &[u16] {
        &self.gpu_slot_to_logical
    }
}

/// Publishes complete placement generations atomically at the snapshot level.
///
/// Readers retain an `Arc` to the generation they observed. A publication
/// never mutates an old map, so in-flight execution cannot see a partial swap.
#[derive(Debug)]
pub struct ExpertPlacementManager {
    current: RwLock<Arc<ExpertPlacementSnapshot>>,
}

impl ExpertPlacementManager {
    pub fn new(initial: ExpertPlacementSnapshot) -> Self {
        Self {
            current: RwLock::new(Arc::new(initial)),
        }
    }

    pub fn new_all_cpu(layer_index: usize, expert_count: usize) -> Result<Self> {
        Ok(Self::new(ExpertPlacementSnapshot::from_gpu_experts(
            layer_index,
            expert_count,
            1,
            &[],
        )?))
    }

    pub fn snapshot(&self) -> Arc<ExpertPlacementSnapshot> {
        Arc::clone(&self.current.read())
    }

    /// Validate a complete replacement before publishing it as one generation.
    pub fn publish(&self, gpu_logical_experts: &[usize]) -> Result<Arc<ExpertPlacementSnapshot>> {
        let mut current = self.current.write();
        let generation = current.generation().checked_add(1).ok_or_else(|| {
            XrtError::Runtime(format!(
                "MoE placement generation overflowed for layer {}",
                current.layer_index()
            ))
        })?;
        let replacement = Arc::new(ExpertPlacementSnapshot::from_gpu_experts(
            current.layer_index(),
            current.expert_count(),
            generation,
            gpu_logical_experts,
        )?);
        *current = Arc::clone(&replacement);
        Ok(replacement)
    }
}

/// One bounded logical-expert replacement proposed for a safe placement epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptivePlacementMove {
    layer_index: usize,
    gpu_slot: usize,
    outgoing_expert: usize,
    incoming_expert: usize,
}

impl AdaptivePlacementMove {
    pub fn layer_index(self) -> usize {
        self.layer_index
    }

    pub fn gpu_slot(self) -> usize {
        self.gpu_slot
    }

    pub fn outgoing_expert(self) -> usize {
        self.outgoing_expert
    }

    pub fn incoming_expert(self) -> usize {
        self.incoming_expert
    }
}

/// Immutable two-phase adaptive-placement proposal.
///
/// Callers upload every incoming expert first. Only after all uploads complete
/// may they publish `target_gpu_experts` and call `commit_evaluation`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptivePlacementDecision {
    evaluation_epoch: u64,
    placement_generation: u64,
    observed_tokens: u64,
    moves: Vec<AdaptivePlacementMove>,
    target_gpu_experts: Vec<Vec<usize>>,
}

impl AdaptivePlacementDecision {
    pub fn evaluation_epoch(&self) -> u64 {
        self.evaluation_epoch
    }

    pub fn placement_generation(&self) -> u64 {
        self.placement_generation
    }

    pub fn observed_tokens(&self) -> u64 {
        self.observed_tokens
    }

    pub fn moves(&self) -> &[AdaptivePlacementMove] {
        &self.moves
    }

    pub fn target_gpu_experts(&self) -> &[Vec<usize>] {
        &self.target_gpu_experts
    }
}

/// Aggregate-only adaptive placement tracker.
///
/// It stores no prompts, token IDs, logits, or activations. Counts are reset
/// after each committed evaluation so the policy responds to workload changes
/// without retaining a request history.
#[derive(Debug)]
pub struct AdaptivePlacementTracker {
    expert_counts: Vec<Vec<u64>>,
    last_moved_epoch: Vec<Vec<u64>>,
    observed_tokens: u64,
    evaluation_epoch: u64,
    update_tokens: u64,
    max_moves_per_update: usize,
    min_residency_epochs: u64,
    hysteresis_percent: u64,
}

impl AdaptivePlacementTracker {
    pub fn new(
        layer_count: usize,
        expert_count: usize,
        update_tokens: u64,
        max_moves_per_update: usize,
        min_residency_epochs: u64,
        hysteresis_percent: u64,
    ) -> Result<Self> {
        if layer_count == 0 || expert_count < 2 {
            return Err(XrtError::Runtime(
                "adaptive MoE placement requires at least one layer and two experts".to_string(),
            ));
        }
        if update_tokens == 0 || max_moves_per_update == 0 || min_residency_epochs == 0 {
            return Err(XrtError::Runtime(
                "adaptive MoE placement intervals and move bounds must be positive".to_string(),
            ));
        }
        if hysteresis_percent > 100 {
            return Err(XrtError::Runtime(
                "adaptive MoE placement hysteresis must be in 0..=100 percent".to_string(),
            ));
        }
        Ok(Self {
            expert_counts: vec![vec![0; expert_count]; layer_count],
            last_moved_epoch: vec![vec![0; expert_count]; layer_count],
            observed_tokens: 0,
            evaluation_epoch: 1,
            update_tokens,
            max_moves_per_update,
            min_residency_epochs,
            hysteresis_percent,
        })
    }

    pub fn record_route(
        &mut self,
        layer_index: usize,
        logical_experts: &[u32],
        completes_token: bool,
    ) -> Result<()> {
        let layer_count = self.expert_counts.len();
        let layer = self.expert_counts.get_mut(layer_index).ok_or_else(|| {
            XrtError::Runtime(format!(
                "adaptive MoE route references layer {layer_index}, outside 0..{}",
                layer_count
            ))
        })?;
        let expert_count = layer.len();
        for &logical_expert in logical_experts {
            let logical_expert = usize::try_from(logical_expert).map_err(|_| {
                XrtError::Runtime("adaptive MoE expert ID does not fit usize".to_string())
            })?;
            let count = layer.get_mut(logical_expert).ok_or_else(|| {
                XrtError::Runtime(format!(
                    "adaptive MoE route references expert {logical_expert}, outside 0..{}",
                    expert_count
                ))
            })?;
            *count = count.saturating_add(1);
        }
        if completes_token {
            self.observed_tokens = self.observed_tokens.saturating_add(1);
        }
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.observed_tokens >= self.update_tokens
    }

    pub fn propose(
        &self,
        current: &[Arc<ExpertPlacementSnapshot>],
    ) -> Result<Option<AdaptivePlacementDecision>> {
        if !self.is_ready() {
            return Ok(None);
        }
        if current.len() != self.expert_counts.len() {
            return Err(XrtError::Runtime(format!(
                "adaptive MoE placement has {} current layers for {} count layers",
                current.len(),
                self.expert_counts.len()
            )));
        }
        let expert_count = self.expert_counts.first().map(Vec::len).unwrap_or_default();
        let placement_generation = current
            .iter()
            .map(|placement| placement.generation())
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| {
                XrtError::Runtime("adaptive MoE placement generation overflowed".to_string())
            })?;
        let evaluation_epoch = self.evaluation_epoch.checked_add(1).ok_or_else(|| {
            XrtError::Runtime("adaptive MoE evaluation epoch overflowed".to_string())
        })?;
        let mut moves = Vec::new();
        let mut targets = Vec::with_capacity(current.len());

        for (layer_index, placement) in current.iter().enumerate() {
            if placement.layer_index() != layer_index
                || placement.expert_count() != expert_count
                || placement.gpu_slot_count() == 0
                || placement.gpu_slot_count() >= expert_count
            {
                return Err(XrtError::Runtime(format!(
                    "adaptive MoE placement geometry is invalid at layer {layer_index}"
                )));
            }
            let mut target = placement
                .gpu_slots_to_logical()
                .iter()
                .map(|&expert| usize::from(expert))
                .collect::<Vec<_>>();
            if moves.len() < self.max_moves_per_update {
                let resident = |expert: usize| target.contains(&expert);
                let incoming = (0..expert_count)
                    .filter(|&expert| !resident(expert))
                    .max_by(|&left, &right| {
                        self.expert_counts[layer_index][left]
                            .cmp(&self.expert_counts[layer_index][right])
                            .then_with(|| right.cmp(&left))
                    });
                let outgoing_slot = target
                    .iter()
                    .enumerate()
                    .filter(|(_, expert)| {
                        evaluation_epoch
                            .saturating_sub(self.last_moved_epoch[layer_index][**expert])
                            >= self.min_residency_epochs
                    })
                    .min_by(|(left_slot, &left), (right_slot, &right)| {
                        self.expert_counts[layer_index][left]
                            .cmp(&self.expert_counts[layer_index][right])
                            .then_with(|| right.cmp(&left))
                            .then_with(|| left_slot.cmp(right_slot))
                    })
                    .map(|(slot, &expert)| (slot, expert));
                if let (Some(incoming), Some((gpu_slot, outgoing))) = (incoming, outgoing_slot) {
                    let incoming_count = self.expert_counts[layer_index][incoming];
                    let outgoing_count = self.expert_counts[layer_index][outgoing];
                    let hysteresis = outgoing_count
                        .saturating_mul(self.hysteresis_percent)
                        .saturating_add(99)
                        / 100;
                    let required = outgoing_count.saturating_add(hysteresis.max(1));
                    if incoming_count >= required {
                        target[gpu_slot] = incoming;
                        moves.push(AdaptivePlacementMove {
                            layer_index,
                            gpu_slot,
                            outgoing_expert: outgoing,
                            incoming_expert: incoming,
                        });
                    }
                }
            }
            targets.push(target);
        }

        Ok(Some(AdaptivePlacementDecision {
            evaluation_epoch,
            placement_generation,
            observed_tokens: self.observed_tokens,
            moves,
            target_gpu_experts: targets,
        }))
    }

    /// Commit an evaluation after its uploads/publication completed.
    ///
    /// Empty decisions still advance the evaluation epoch and clear bounded
    /// counters, preventing a no-churn workload from being reconsidered at
    /// every request boundary.
    pub fn commit_evaluation(&mut self, decision: &AdaptivePlacementDecision) -> Result<()> {
        if decision.evaluation_epoch != self.evaluation_epoch.saturating_add(1)
            || decision.observed_tokens != self.observed_tokens
        {
            return Err(XrtError::Runtime(
                "adaptive MoE decision no longer matches the observed epoch".to_string(),
            ));
        }
        for movement in &decision.moves {
            for expert in [movement.outgoing_expert, movement.incoming_expert] {
                let epoch = self
                    .last_moved_epoch
                    .get_mut(movement.layer_index)
                    .and_then(|layer| layer.get_mut(expert))
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "adaptive MoE decision contains invalid movement geometry".to_string(),
                        )
                    })?;
                *epoch = decision.evaluation_epoch;
            }
        }
        self.evaluation_epoch = decision.evaluation_epoch;
        self.observed_tokens = 0;
        for layer in &mut self.expert_counts {
            layer.fill(0);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{AdaptivePlacementTracker, ExpertPlacementManager, ExpertPlacementSnapshot};

    #[test]
    fn placement_maps_are_checked_bijections() {
        let snapshot = ExpertPlacementSnapshot::from_gpu_experts(7, 8, 4, &[6, 1, 4]).unwrap();
        assert_eq!(snapshot.layer_index(), 7);
        assert_eq!(snapshot.generation(), 4);
        assert_eq!(snapshot.gpu_slot_for(6), Some(0));
        assert_eq!(snapshot.gpu_slot_for(1), Some(1));
        assert_eq!(snapshot.gpu_slot_for(4), Some(2));
        assert_eq!(snapshot.gpu_slot_for(0), None);
        assert_eq!(snapshot.logical_expert_for(0), Some(6));
        assert_eq!(snapshot.logical_expert_for(2), Some(4));
        assert_eq!(snapshot.logical_expert_for(3), None);

        assert!(ExpertPlacementSnapshot::from_gpu_experts(0, 4, 1, &[1, 1]).is_err());
        assert!(ExpertPlacementSnapshot::from_gpu_experts(0, 4, 1, &[4]).is_err());
        assert!(ExpertPlacementSnapshot::from_gpu_experts(0, 4, 0, &[]).is_err());
        assert!(
            ExpertPlacementSnapshot::from_gpu_experts(0, usize::from(u16::MAX) + 2, 1, &[])
                .is_err()
        );
    }

    #[test]
    fn uniform_placement_is_even_and_deterministic() {
        let snapshot = ExpertPlacementSnapshot::uniform(3, 8, 4, 1).unwrap();
        assert_eq!(snapshot.gpu_slots_to_logical(), &[0, 2, 4, 6]);
        assert_eq!(
            snapshot.logical_to_gpu_slots(),
            &[Some(0), None, Some(1), None, Some(2), None, Some(3), None]
        );
    }

    #[test]
    fn publication_keeps_old_generation_alive_and_never_partially_mutates_it() {
        let manager = ExpertPlacementManager::new_all_cpu(2, 4).unwrap();
        let old = manager.snapshot();
        let new = manager.publish(&[3, 1]).unwrap();

        assert_eq!(old.generation(), 1);
        assert_eq!(old.gpu_slot_count(), 0);
        assert_eq!(new.generation(), 2);
        assert_eq!(new.gpu_slots_to_logical(), &[3, 1]);
        assert_eq!(manager.snapshot().generation(), 2);

        assert!(manager.publish(&[0, 0]).is_err());
        assert_eq!(manager.snapshot().generation(), 2);
        assert_eq!(manager.snapshot().gpu_slots_to_logical(), &[3, 1]);
    }

    #[test]
    fn adaptive_policy_is_bounded_hysteretic_and_two_phase() {
        let current = vec![
            Arc::new(ExpertPlacementSnapshot::from_gpu_experts(0, 4, 1, &[0, 2]).unwrap()),
            Arc::new(ExpertPlacementSnapshot::from_gpu_experts(1, 4, 1, &[0, 2]).unwrap()),
        ];
        let mut tracker = AdaptivePlacementTracker::new(2, 4, 2, 1, 1, 10).unwrap();
        tracker.record_route(0, &[3, 3], false).unwrap();
        tracker.record_route(1, &[1, 1], true).unwrap();
        assert!(!tracker.is_ready());
        tracker.record_route(0, &[3, 3], false).unwrap();
        tracker.record_route(1, &[1, 1], true).unwrap();
        let decision = tracker.propose(&current).unwrap().unwrap();
        assert_eq!(decision.observed_tokens(), 2);
        assert_eq!(decision.placement_generation(), 2);
        assert_eq!(decision.moves().len(), 1);
        assert_eq!(decision.moves()[0].layer_index(), 0);
        assert_eq!(decision.moves()[0].incoming_expert(), 3);
        assert_eq!(decision.target_gpu_experts()[0], vec![0, 3]);
        assert_eq!(decision.target_gpu_experts()[1], vec![0, 2]);

        // Proposals are side-effect free until upload/publication commits.
        assert_eq!(tracker.propose(&current).unwrap(), Some(decision.clone()));
        tracker.commit_evaluation(&decision).unwrap();
        assert!(!tracker.is_ready());
        assert!(tracker.commit_evaluation(&decision).is_err());
    }

    #[test]
    fn adaptive_policy_does_not_churn_on_ties() {
        let current = vec![Arc::new(
            ExpertPlacementSnapshot::from_gpu_experts(0, 4, 3, &[0, 2]).unwrap(),
        )];
        let mut tracker = AdaptivePlacementTracker::new(1, 4, 1, 2, 1, 10).unwrap();
        tracker.record_route(0, &[0, 1, 2, 3], true).unwrap();
        let decision = tracker.propose(&current).unwrap().unwrap();
        assert!(decision.moves().is_empty());
        assert_eq!(decision.target_gpu_experts()[0], vec![0, 2]);
        tracker.commit_evaluation(&decision).unwrap();
    }
}
