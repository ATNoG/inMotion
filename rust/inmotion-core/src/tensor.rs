//! A minimal dense f32 tensor.
//!
//! Deliberately not `ndarray`: the shapes here are fixed and small, the targets
//! are baseline x86-64 (gnu and musl), and a 60-line type with no dependencies is
//! easier to audit than a general-purpose array library. The ESP32 path is C
//! firmware under `firmware/`, not this crate.

use std::fmt;

/// Row-major dense tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Row-major flat storage, `shape.iter().product()` elements long.
    pub data: Vec<f32>,
    /// Extent of each axis, outermost first.
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Build from row-major data, checking the shape is consistent.
    #[must_use]
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
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(vec![0.0; shape.iter().product()], shape.to_vec())
    }

    /// Last dimension, or 1 for a scalar.
    #[must_use]
    pub fn last_dim(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    /// Number of leading elements before the last axis.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.data.len() / self.last_dim().max(1)
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 2-D view as (rows, cols), panicking on higher rank.
    #[must_use]
    pub fn as_2d(&self) -> (usize, usize) {
        assert_eq!(
            self.shape.len(),
            2,
            "expected a 2-D tensor, got {:?}",
            self.shape
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_must_match_the_data() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.len(), 4);
        assert_eq!(t.last_dim(), 2);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.as_2d(), (2, 2));
    }

    #[test]
    #[should_panic(expected = "does not match shape")]
    fn mismatched_shape_is_rejected() {
        // The constructor must reject the mismatch before anything else happens.
        let _ = Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
    }

    #[test]
    fn zeros_has_the_requested_extent() {
        let t = Tensor::zeros(&[3, 5]);
        assert_eq!(t.shape, vec![3, 5]);
        assert!(t.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn max_abs_diff_reports_the_largest_elementwise_gap() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![1.0, 2.5, 2.0], vec![3]);
        assert!((a.max_abs_diff(&b) - 1.0).abs() < 1e-6);
    }
}
