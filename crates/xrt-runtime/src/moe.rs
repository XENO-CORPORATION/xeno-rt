use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
};

use xrt_core::{Result, XrtError};
use xrt_models::{MoeLayerDescriptor, MoeRoutingRow};

use crate::expert_placement::ExpertPlacementSnapshot;

const HETEROGENEOUS_COORDINATOR_QUEUE_CAPACITY: usize = 8;

enum CoordinatorMessage {
    Run(Box<dyn FnOnce() + Send + 'static>),
    Shutdown,
}

/// One persistent, bounded coordinator used to overlap CPU expert work with
/// CUDA work without spawning a thread for each token or layer.
pub(crate) struct HeterogeneousMoeCoordinator {
    sender: SyncSender<CoordinatorMessage>,
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for HeterogeneousMoeCoordinator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HeterogeneousMoeCoordinator")
            .field("queue_capacity", &HETEROGENEOUS_COORDINATOR_QUEUE_CAPACITY)
            .finish_non_exhaustive()
    }
}

impl HeterogeneousMoeCoordinator {
    pub(crate) fn new() -> Result<Self> {
        let (sender, receiver) =
            mpsc::sync_channel::<CoordinatorMessage>(HETEROGENEOUS_COORDINATOR_QUEUE_CAPACITY);
        let worker = thread::Builder::new()
            .name("xrt-moe-coordinator".to_string())
            .spawn(move || {
                while let Ok(message) = receiver.recv() {
                    match message {
                        CoordinatorMessage::Run(job) => job(),
                        CoordinatorMessage::Shutdown => break,
                    }
                }
            })
            .map_err(|error| {
                XrtError::Runtime(format!("failed to start MoE coordinator thread: {error}"))
            })?;
        Ok(Self {
            sender,
            worker: Some(worker),
        })
    }

    pub(crate) fn submit<T, F>(&self, work: F) -> Result<HeterogeneousMoeJoin<T>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let (result_sender, result_receiver) = mpsc::sync_channel(1);
        self.sender
            .send(CoordinatorMessage::Run(Box::new(move || {
                let result = match catch_unwind(AssertUnwindSafe(work)) {
                    Ok(result) => result,
                    Err(_) => Err(XrtError::Runtime(
                        "heterogeneous MoE coordinator job panicked".to_string(),
                    )),
                };
                let _ = result_sender.send(result);
            })))
            .map_err(|_| {
                XrtError::Runtime("heterogeneous MoE coordinator is unavailable".to_string())
            })?;
        Ok(HeterogeneousMoeJoin { result_receiver })
    }
}

impl Drop for HeterogeneousMoeCoordinator {
    fn drop(&mut self) {
        let _ = self.sender.send(CoordinatorMessage::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(crate) struct HeterogeneousMoeJoin<T> {
    result_receiver: Receiver<Result<T>>,
}

impl<T> HeterogeneousMoeJoin<T> {
    pub(crate) fn join(self) -> Result<T> {
        self.result_receiver.recv().map_err(|_| {
            XrtError::Runtime(
                "heterogeneous MoE coordinator stopped before returning a result".to_string(),
            )
        })?
    }
}

/// One selected logical expert in canonical token/top-k order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoeWorkItem {
    token_index: u32,
    route_slot: u8,
    canonical_index: u32,
    logical_expert: u32,
    routing_weight: f32,
    gpu_slot: Option<u16>,
}

impl MoeWorkItem {
    pub fn token_index(self) -> usize {
        self.token_index as usize
    }

    pub fn route_slot(self) -> usize {
        self.route_slot as usize
    }

    pub fn canonical_index(self) -> usize {
        self.canonical_index as usize
    }

    pub fn logical_expert(self) -> usize {
        self.logical_expert as usize
    }

    pub fn routing_weight(self) -> f32 {
        self.routing_weight
    }

    pub fn gpu_slot(self) -> Option<u16> {
        self.gpu_slot
    }
}

#[derive(Debug)]
pub struct MoeExecutionPlan<'a> {
    layer: &'a MoeLayerDescriptor,
    placement: &'a ExpertPlacementSnapshot,
    routes: &'a [MoeRoutingRow],
    cpu_work: &'a [MoeWorkItem],
    gpu_work: &'a [MoeWorkItem],
}

impl<'a> MoeExecutionPlan<'a> {
    pub fn layer(&self) -> &'a MoeLayerDescriptor {
        self.layer
    }

    pub fn placement(&self) -> &'a ExpertPlacementSnapshot {
        self.placement
    }

    pub fn placement_generation(&self) -> u64 {
        self.placement.generation()
    }

    pub fn routes(&self) -> &'a [MoeRoutingRow] {
        self.routes
    }

    pub fn cpu_work(&self) -> &'a [MoeWorkItem] {
        self.cpu_work
    }

    pub fn gpu_work(&self) -> &'a [MoeWorkItem] {
        self.gpu_work
    }

    pub fn selected_work_len(&self) -> usize {
        self.cpu_work.len() + self.gpu_work.len()
    }
}

/// Build a tier-partitioned exact plan using caller-owned, reusable storage.
///
/// The output work items retain their canonical reduction index, so a
/// heterogeneous executor can finish in any order but must merge in that
/// original logical order.
pub fn build_moe_execution_plan<'a>(
    layer: &'a MoeLayerDescriptor,
    placement: &'a ExpertPlacementSnapshot,
    routes: &'a [MoeRoutingRow],
    cpu_scratch: &'a mut [MoeWorkItem],
    gpu_scratch: &'a mut [MoeWorkItem],
) -> Result<MoeExecutionPlan<'a>> {
    if layer.layer_index() != placement.layer_index() {
        return Err(XrtError::Runtime(format!(
            "MoE placement layer {} cannot plan model layer {}",
            placement.layer_index(),
            layer.layer_index()
        )));
    }
    if layer.expert_count() != placement.expert_count() {
        return Err(XrtError::Runtime(format!(
            "MoE placement for layer {} has {} logical experts, expected {}",
            layer.layer_index(),
            placement.expert_count(),
            layer.expert_count()
        )));
    }
    let selected_work_len = routes
        .len()
        .checked_mul(layer.selected_per_token())
        .ok_or_else(|| XrtError::Runtime("MoE selected work size overflowed".to_string()))?;
    if selected_work_len > u32::MAX as usize {
        return Err(XrtError::Unsupported(format!(
            "MoE plan has {selected_work_len} selections, exceeding the checked work-index width"
        )));
    }
    if routes.len() > u32::MAX as usize {
        return Err(XrtError::Unsupported(format!(
            "MoE plan has {} token rows, exceeding the checked token-index width",
            routes.len()
        )));
    }
    if cpu_scratch.len() < selected_work_len || gpu_scratch.len() < selected_work_len {
        return Err(XrtError::InvalidTensor(format!(
            "MoE planning scratch requires {selected_work_len} CPU and GPU entries, received {} and {}",
            cpu_scratch.len(),
            gpu_scratch.len()
        )));
    }

    // Validate the complete input before mutating either output span.
    for (token_index, route) in routes.iter().enumerate() {
        if route.len() != layer.selected_per_token() {
            return Err(XrtError::InvalidTensor(format!(
                "MoE layer {} token row {token_index} has {} selections, expected {}",
                layer.layer_index(),
                route.len(),
                layer.selected_per_token()
            )));
        }
        for logical_expert in route.logical_ids() {
            if *logical_expert as usize >= layer.expert_count() {
                return Err(XrtError::InvalidTensor(format!(
                    "MoE layer {} token row {token_index} selects logical expert {}, outside 0..{}",
                    layer.layer_index(),
                    logical_expert,
                    layer.expert_count()
                )));
            }
        }
    }

    let mut cpu_len = 0usize;
    let mut gpu_len = 0usize;
    for (token_index, route) in routes.iter().enumerate() {
        for (route_slot, (logical_expert, routing_weight)) in route.iter().enumerate() {
            let canonical_index = token_index
                .checked_mul(layer.selected_per_token())
                .and_then(|base| base.checked_add(route_slot))
                .expect("selected work length was checked above");
            let gpu_slot = placement.gpu_slot_for(logical_expert);
            let item = MoeWorkItem {
                token_index: token_index as u32,
                route_slot: route_slot as u8,
                canonical_index: canonical_index as u32,
                logical_expert: logical_expert as u32,
                routing_weight,
                gpu_slot,
            };
            if gpu_slot.is_some() {
                gpu_scratch[gpu_len] = item;
                gpu_len += 1;
            } else {
                cpu_scratch[cpu_len] = item;
                cpu_len += 1;
            }
        }
    }

    Ok(MoeExecutionPlan {
        layer,
        placement,
        routes,
        cpu_work: &cpu_scratch[..cpu_len],
        gpu_work: &gpu_scratch[..gpu_len],
    })
}

#[cfg(test)]
mod tests {
    use super::{build_moe_execution_plan, HeterogeneousMoeCoordinator, MoeWorkItem};
    use crate::expert_placement::ExpertPlacementSnapshot;
    use xrt_models::{route_top_k, MoeLayerDescriptor, MoeRoutingRow};

    fn scratch_item() -> MoeWorkItem {
        MoeWorkItem {
            token_index: 0,
            route_slot: 0,
            canonical_index: 0,
            logical_expert: 0,
            routing_weight: 0.0,
            gpu_slot: None,
        }
    }

    #[test]
    fn execution_plan_partitions_tiers_without_losing_canonical_order() {
        let layer = MoeLayerDescriptor::new(3, 4, 2, 8, 16).unwrap();
        let placement = ExpertPlacementSnapshot::from_gpu_experts(3, 4, 7, &[1, 3]).unwrap();
        let mut routes = [MoeRoutingRow::default(); 2];
        route_top_k(&[0.0, 4.0, 3.0, 2.0], 2, &mut routes[0]).unwrap();
        route_top_k(&[4.0, 0.0, 1.0, 3.0], 2, &mut routes[1]).unwrap();
        let mut cpu = [scratch_item(); 4];
        let mut gpu = [scratch_item(); 4];

        let plan =
            build_moe_execution_plan(&layer, &placement, &routes, &mut cpu, &mut gpu).unwrap();
        assert_eq!(plan.placement_generation(), 7);
        assert_eq!(plan.selected_work_len(), 4);
        assert_eq!(
            plan.gpu_work()
                .iter()
                .map(|item| (
                    item.canonical_index(),
                    item.logical_expert(),
                    item.gpu_slot()
                ))
                .collect::<Vec<_>>(),
            vec![(0, 1, Some(0)), (3, 3, Some(1))]
        );
        assert_eq!(
            plan.cpu_work()
                .iter()
                .map(|item| (item.canonical_index(), item.logical_expert()))
                .collect::<Vec<_>>(),
            vec![(1, 2), (2, 0)]
        );
    }

    #[test]
    fn execution_plan_rejects_mismatched_generation_geometry_and_scratch() {
        let layer = MoeLayerDescriptor::new(1, 2, 1, 4, 8).unwrap();
        let wrong_layer = ExpertPlacementSnapshot::uniform(2, 2, 1, 1).unwrap();
        let mut route = MoeRoutingRow::default();
        route_top_k(&[1.0, 0.0], 1, &mut route).unwrap();
        let mut cpu = [scratch_item(); 1];
        let mut gpu = [scratch_item(); 1];
        assert!(
            build_moe_execution_plan(&layer, &wrong_layer, &[route], &mut cpu, &mut gpu).is_err()
        );

        let placement = ExpertPlacementSnapshot::uniform(1, 2, 1, 1).unwrap();
        assert!(build_moe_execution_plan(&layer, &placement, &[route], &mut [], &mut gpu).is_err());
    }

    #[test]
    fn coordinator_is_bounded_reusable_and_converts_panics_to_errors() {
        let coordinator = HeterogeneousMoeCoordinator::new().unwrap();
        assert_eq!(
            coordinator.submit(|| Ok(42usize)).unwrap().join().unwrap(),
            42
        );
        let panic_result = coordinator
            .submit::<(), _>(|| panic!("synthetic coordinator panic"))
            .unwrap()
            .join();
        assert!(panic_result.is_err());
        assert_eq!(
            coordinator.submit(|| Ok(7usize)).unwrap().join().unwrap(),
            7
        );
    }
}
