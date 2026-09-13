//! Inference for the inMotion WiFi-RSSI models, outside Python.
//!
//! The Python side exports an artifact (weights plus a JSON description) and this
//! crate executes it. Two things are deliberately true of this port:
//!
//! 1. **It does not depend on `PyTorch`, ONNX Runtime or `ndarray`.** The
//!    supported targets are baseline `x86_64-unknown-linux-gnu` and
//!    `x86_64-unknown-linux-musl`, and the graphs are small and fixed-shape. (The
//!    ESP32 demo is separate C firmware under `firmware/`, not this crate.)
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
