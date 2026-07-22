use std::fmt;
use std::num::NonZeroUsize;
use std::str::FromStr;
use std::thread;

use xrt_core::{Result, XrtError};

/// NUMA discovery and affinity policy for CPU execution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NumaPolicy {
    /// Discover topology when supported and fall back to one portable node.
    #[default]
    Auto,
    /// Do not inspect or apply NUMA topology.
    Off,
    /// Require usable NUMA discovery and affinity support.
    Strict,
}

impl FromStr for NumaPolicy {
    type Err = XrtError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            "strict" => Ok(Self::Strict),
            other => Err(XrtError::Runtime(format!(
                "invalid XRT_MOE_NUMA value {other:?}; expected auto, off, or strict"
            ))),
        }
    }
}

/// One logical CPU locality node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuNode {
    id: usize,
    logical_cpus: Box<[usize]>,
}

impl CpuNode {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn logical_cpus(&self) -> &[usize] {
        &self.logical_cpus
    }
}

/// Portable CPU topology description used by dense and expert execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuTopology {
    nodes: Box<[CpuNode]>,
    logical_cpus: usize,
    affinity_supported: bool,
    source: TopologySource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopologySource {
    Portable,
    LinuxSysfs,
}

impl CpuTopology {
    /// Discover the host topology according to `policy`.
    ///
    /// `Auto` is deliberately fail-open: an unreadable platform topology becomes
    /// one portable node. `Strict` is intended for benchmark diagnosis and fails
    /// when discovery or affinity cannot be guaranteed.
    pub fn discover(policy: NumaPolicy) -> Result<Self> {
        if policy == NumaPolicy::Off {
            return Ok(Self::portable());
        }

        #[cfg(target_os = "linux")]
        {
            match discover_linux() {
                Ok(topology) => return Ok(topology),
                Err(error) if policy == NumaPolicy::Strict => return Err(error),
                Err(_) => return Ok(Self::portable()),
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            if policy == NumaPolicy::Strict {
                return Err(XrtError::Unsupported(
                    "strict NUMA mode is currently supported only on Linux".to_string(),
                ));
            }
            Ok(Self::portable())
        }
    }

    pub fn portable() -> Self {
        let logical_cpus = thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1);
        Self {
            nodes: vec![CpuNode {
                id: 0,
                logical_cpus: (0..logical_cpus).collect::<Vec<_>>().into_boxed_slice(),
            }]
            .into_boxed_slice(),
            logical_cpus,
            affinity_supported: false,
            source: TopologySource::Portable,
        }
    }

    pub fn nodes(&self) -> &[CpuNode] {
        &self.nodes
    }

    pub fn logical_cpus(&self) -> usize {
        self.logical_cpus
    }

    pub fn affinity_supported(&self) -> bool {
        self.affinity_supported
    }

    pub fn source(&self) -> TopologySource {
        self.source
    }

    /// Bind the current thread to the CPUs in `node_index`.
    ///
    /// This is a best-effort optimization in `auto` mode. Callers decide whether
    /// an error is diagnostic-only or fatal according to their resolved policy.
    pub fn bind_current_thread_to_node(&self, node_index: usize) -> Result<()> {
        let node = self.nodes.get(node_index).ok_or_else(|| {
            XrtError::Runtime(format!(
                "CPU topology node index {node_index} is out of range for {} nodes",
                self.nodes.len()
            ))
        })?;
        if !self.affinity_supported {
            return Err(XrtError::Unsupported(
                "CPU affinity is not supported by the discovered topology".to_string(),
            ));
        }

        #[cfg(target_os = "linux")]
        {
            // SAFETY: cpu_set_t is initialized before use, the CPU ids were
            // validated against CPU_SETSIZE during discovery, and the pointer is
            // valid for exactly size_of::<cpu_set_t>() bytes for this call.
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut set);
                for &cpu in node.logical_cpus() {
                    libc::CPU_SET(cpu, &mut set);
                }
                let result = libc::pthread_setaffinity_np(
                    libc::pthread_self(),
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &set,
                );
                if result != 0 {
                    return Err(XrtError::Runtime(format!(
                        "pthread_setaffinity_np failed for NUMA node {} with errno {result}",
                        node.id()
                    )));
                }
                return Ok(());
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = node;
            Err(XrtError::Unsupported(
                "CPU affinity is unavailable on this platform".to_string(),
            ))
        }
    }
}

/// Origin of the runtime-wide CPU thread budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadBudgetSource {
    XrtEnvironment,
    RayonCompatibility,
    HostDefault,
}

impl fmt::Display for ThreadBudgetSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::XrtEnvironment => "XRT_CPU_THREADS",
            Self::RayonCompatibility => "RAYON_NUM_THREADS",
            Self::HostDefault => "host-default",
        })
    }
}

/// One resolved budget shared by dense kernels and MoE dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuThreadBudget {
    total_threads: NonZeroUsize,
    logical_cpus: NonZeroUsize,
    source: ThreadBudgetSource,
}

impl CpuThreadBudget {
    pub fn resolve_from_environment() -> Result<Self> {
        let logical_cpus = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);

        for (name, source) in [
            ("XRT_CPU_THREADS", ThreadBudgetSource::XrtEnvironment),
            ("RAYON_NUM_THREADS", ThreadBudgetSource::RayonCompatibility),
        ] {
            if let Ok(raw) = std::env::var(name) {
                let total_threads = raw.parse::<NonZeroUsize>().map_err(|_| {
                    XrtError::Runtime(format!(
                        "{name} must be a positive integer, received {raw:?}"
                    ))
                })?;
                return Ok(Self {
                    total_threads,
                    logical_cpus,
                    source,
                });
            }
        }

        Ok(Self::host_default_for(logical_cpus))
    }

    pub fn host_default() -> Self {
        let logical_cpus = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
        Self::host_default_for(logical_cpus)
    }

    fn host_default_for(logical_cpus: NonZeroUsize) -> Self {
        // Approximate physical cores on SMT hosts and cap the spin pool where
        // memory bandwidth normally saturates. Never exceed available CPUs.
        let default_threads = logical_cpus
            .get()
            .div_ceil(2)
            .clamp(1, 16)
            .min(logical_cpus.get());
        Self {
            total_threads: NonZeroUsize::new(default_threads).unwrap_or(NonZeroUsize::MIN),
            logical_cpus,
            source: ThreadBudgetSource::HostDefault,
        }
    }

    pub fn total_threads(self) -> usize {
        self.total_threads.get()
    }

    pub fn worker_threads(self) -> usize {
        self.total_threads.get().saturating_sub(1)
    }

    pub fn logical_cpus(self) -> usize {
        self.logical_cpus.get()
    }

    pub fn source(self) -> ThreadBudgetSource {
        self.source
    }
}

#[cfg(target_os = "linux")]
fn discover_linux() -> Result<CpuTopology> {
    use std::fs;
    use std::path::Path;

    let online = fs::read_to_string("/sys/devices/system/node/online").map_err(|error| {
        XrtError::Runtime(format!(
            "failed to read Linux NUMA node list from sysfs: {error}"
        ))
    })?;
    let node_ids = parse_linux_cpu_list(&online)?;
    if node_ids.is_empty() {
        return Err(XrtError::Runtime(
            "Linux sysfs reported no online NUMA nodes".to_string(),
        ));
    }

    let mut nodes = Vec::with_capacity(node_ids.len());
    let mut maximum_cpu = None;
    for id in node_ids {
        let path = format!("/sys/devices/system/node/node{id}/cpulist");
        if !Path::new(&path).is_file() {
            return Err(XrtError::Runtime(format!(
                "Linux NUMA node {id} has no cpulist in sysfs"
            )));
        }
        let raw = fs::read_to_string(&path).map_err(|error| {
            XrtError::Runtime(format!(
                "failed to read Linux NUMA node {id} cpulist: {error}"
            ))
        })?;
        let logical_cpus = parse_linux_cpu_list(&raw)?;
        if logical_cpus.is_empty() {
            return Err(XrtError::Runtime(format!(
                "Linux NUMA node {id} contains no logical CPUs"
            )));
        }
        maximum_cpu = maximum_cpu.max(logical_cpus.iter().copied().max());
        nodes.push(CpuNode {
            id,
            logical_cpus: logical_cpus.into_boxed_slice(),
        });
    }

    let logical_cpus = nodes.iter().map(|node| node.logical_cpus.len()).sum();
    let affinity_supported = maximum_cpu
        .map(|cpu| cpu < libc::CPU_SETSIZE as usize)
        .unwrap_or(false);
    if !affinity_supported {
        return Err(XrtError::Unsupported(format!(
            "Linux topology contains a CPU id outside cpu_set_t capacity {}",
            libc::CPU_SETSIZE
        )));
    }

    Ok(CpuTopology {
        nodes: nodes.into_boxed_slice(),
        logical_cpus,
        affinity_supported,
        source: TopologySource::LinuxSysfs,
    })
}

#[cfg(target_os = "linux")]
fn parse_linux_cpu_list(raw: &str) -> Result<Vec<usize>> {
    let mut values = Vec::new();
    for part in raw.trim().split(',').filter(|part| !part.is_empty()) {
        let mut bounds = part.split('-');
        let start = bounds
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| XrtError::Runtime(format!("invalid Linux CPU list segment {part:?}")))?;
        let end = bounds
            .next()
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|_| XrtError::Runtime(format!("invalid Linux CPU list segment {part:?}")))?
            .unwrap_or(start);
        if bounds.next().is_some() || end < start {
            return Err(XrtError::Runtime(format!(
                "invalid Linux CPU list segment {part:?}"
            )));
        }
        values.extend(start..=end);
    }
    values.sort_unstable();
    values.dedup();
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_topology_always_has_one_non_empty_node() {
        let topology = CpuTopology::portable();
        assert_eq!(topology.nodes().len(), 1);
        assert!(!topology.nodes()[0].logical_cpus().is_empty());
        assert_eq!(
            topology.logical_cpus(),
            topology.nodes()[0].logical_cpus().len()
        );
        assert!(!topology.affinity_supported());
    }

    #[test]
    fn numa_policy_parser_is_explicit() {
        assert_eq!("auto".parse::<NumaPolicy>().unwrap(), NumaPolicy::Auto);
        assert_eq!("OFF".parse::<NumaPolicy>().unwrap(), NumaPolicy::Off);
        assert_eq!("strict".parse::<NumaPolicy>().unwrap(), NumaPolicy::Strict);
        assert!("best-effort".parse::<NumaPolicy>().is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_cpu_list_parser_handles_ranges_and_rejects_reverse_ranges() {
        assert_eq!(
            parse_linux_cpu_list("0-2,5,7-8\n").unwrap(),
            vec![0, 1, 2, 5, 7, 8]
        );
        assert!(parse_linux_cpu_list("3-1").is_err());
        assert!(parse_linux_cpu_list("1-2-3").is_err());
    }
}
