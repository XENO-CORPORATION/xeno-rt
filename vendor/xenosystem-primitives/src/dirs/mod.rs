//! `dirs`: where every kind of XENO data lives on a person's machine (XENO LOCAL HOME - SPEC.md).
//! Zero runtime dependencies. The registry and vectors in the repository's `contract/` directory are authoritative.

pub mod atomic;
pub mod codec;
pub mod doctor;
pub mod error;
pub mod json;
pub mod layout;
pub mod lock;
pub mod migrate;
pub mod procinfo;
pub mod registry;
pub mod roots;
pub mod segments;
pub mod xdirs;

pub use atomic::atomic_write_file;
pub use error::{DirsError, ErrorCode, Result};
pub use layout::{parse_layout, supported_layout, LayoutState};
pub use lock::{acquire_lock_file, judge_lock, AcquireOptions, LockHandle, LockRecord, LockVerdict, Probe};
pub use doctor::{doctor, DoctorReport, Unregistered};
pub use migrate::{builtin_rules, journal_path, plan_id, plan_migration, read_journal, run_migration, verify_migration, EntryResult, EntryStatus, MigrationPlan, MigrationResult, MigrationRule, PlanEntry, Refusal, RuleAction, RunOptions, SecretSink, VerifyResult};
pub use registry::{find_area, inputs, lock_relative_path, registered_relative_path, AreaEntry, Inputs};
pub use roots::{is_absolute, join_path, normalize_base, resolve_roots, Kind, Platform, ResolveInput, ResolvedRoots, KINDS};
pub use segments::{assert_segment, segment_problem};
pub use xdirs::{create_dirs, create_dirs_at, CreateDirsOptions, XenoDirs};
