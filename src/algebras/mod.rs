/// Core operations.
pub mod core;

/// Pointwise analytic functions (cos, sin, log, exp, pow, sqrt, ..)
pub mod analytic;

/// Pointwise arithmetic operations.
pub mod arithmetic;

/// Pointwise arithmetic operations with a constant value.
pub mod const_arith;

/// Pointwise comparison operations.
pub mod compare;

/// Operation to propagate gradients in the case of high-order differentials.
pub mod linked;

/// Array operations.
pub mod array;

/// Array operations with comparisons.
pub mod array_compare;

/// Operations on matrix.
pub mod matrix;

/// Additional definitions for Arrayfire.
#[cfg(feature = "arrayfire")]
pub mod arrayfire;
