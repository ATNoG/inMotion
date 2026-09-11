//! Inference for the inMotion WiFi-RSSI models, outside Python.
//!
//! The Python side exports an artifact (weights plus a JSON description) and this
//! crate executes it. Two things are deliberately true of this port:
//!
//! 1. **It does not depend on PyTorch, ONNX Runtime or `ndarray`.** The target
//!    includes an ESP32-S3, and the graph is small and fixed-shape.
//! 2. **It is verified against the reference, not assumed correct.** The Python
//!    exporter writes golden input/output pairs from the real test split and
//!    `tests/parity.rs` asserts this port reproduces them. A port that is not
//!    checked against the original is worthless.

pub mod artifact;
pub mod lejepa;
pub mod ops;
pub mod quant;
pub mod tensor;

pub use artifact::Artifact;
pub use lejepa::LeJepa;
pub use tensor::Tensor;
