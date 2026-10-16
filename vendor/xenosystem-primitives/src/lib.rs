//! XENO back-end rung 1 primitives. One module and cargo feature per family:
//! `dirs` (the XENO local home) and `secrets` (envelope-encrypted credentials).
//! Authority: `XENO LOCAL HOME - SPEC.md`; the shared contract lives in the repository's `contract/` directory.

#[cfg(feature = "dirs")]
pub mod dirs;
#[cfg(feature = "secrets")]
pub mod secrets;
