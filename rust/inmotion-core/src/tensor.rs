//! A minimal dense f32 tensor.
//!
//! Deliberately not `ndarray`: the shapes here are fixed and small, the target
//! includes microcontrollers, and a 60-line type with no dependencies is easier
//! to audit than a general-purpose array library.

use std::fmt;

/// Row-major dense tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Build from row-major data, checking the shape is consistent.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length {} does not match shape {:?} ({} elements)",
            data.len(),
            shape,
            expected
        );
        Self { data, shape }
    }

    /// All-zero tensor.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(vec![0.0; shape.iter().product()], shape.to_vec())
    }

    /// Last dimension, or 1 for a scalar.
    pub fn last_dim(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    /// Number of leading elements before the last axis.
    pub fn rows(&self) -> usize {
        self.data.len() / self.last_dim().max(1)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 2-D view as (rows, cols), panicking on higher rank.
    pub fn as_2d(&self) -> (usize, usize) {
        assert_eq!(self.shape.len(), 2, "expected a 2-D tensor, got {:?}", self.shape);
        (self.shape[0], self.shape[1])
    }

    /// Maximum absolute difference against another tensor of equal length.
    pub fn max_abs_diff(&self, other: &Tensor) -> f32 {
        assert_eq!(self.len(), other.len(), "length mismatch");
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?}", self.shape)
    }
}
