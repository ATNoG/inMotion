//! Inference for the inMotion WiFi-RSSI models, outside Python.
//!
//! The Python side exports an artifact (weights plus a JSON description) and this
//! crate executes it. Two things are deliberately true of this port:
//!
//! 1. **It does not depend on `PyTorch`, ONNX Runtime or `ndarray`.** The target
//!    includes an ESP32-S3, and the graph is small and fixed-shape.
//! 2. **It is verified against the reference, not assumed correct.** The Python
//!    exporter writes golden input/output pairs from the real test split and the
//!    CLI replays them. A port that is not checked against the original is
//!    worthless.

pub mod artifact;
pub mod model;
pub mod ops;
pub mod quant;
pub mod sigreg;
pub mod tensor;
pub mod world_model;

pub use artifact::Artifact;
pub use model::{Family, Model};
pub use tensor::Tensor;
