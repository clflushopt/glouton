//! Glouton IR optimization passes.
//!
//! This module implements multiple transforms on the glouton IR
//! mostly focused on scalar optimizations.

/// `Transform` trait is used to encapsulate the behavior of independant
/// optimizations executed on individual functions.
trait Transform {}
