//! Glouton IR optimization passes.
//!
//! This module implements multiple transforms on the glouton IR
//! mostly focused on scalar optimizations.
use crate::ir::{Function, Instruction};

/// So what is a pass ?
trait Transform {}
