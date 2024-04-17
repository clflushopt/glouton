//! Glouton IR optimization passes.
//!
//! This module implements multiple transforms on the glouton IR
//! mostly focused on scalar optimizations.

use crate::ir;

/// `Transform` trait is used to encapsulate the behavior of independant
/// optimizations executed on individual functions.
trait Transform {
    fn run(function: &mut ir::Function) {}
}

/// Instruction combination pass executes over basic blocks and tries to
/// combine instructions that can be combined into one instruction.
struct InstCombine {}

impl Transform for InstCombine {}

/// Constant folding and propagation pass targets expressions that can be
/// evaluated at compile time and replaces them with the evaluation, once
/// constants are folded a second sub-pass executes to propagate constants
/// to their usage locations.
struct ConstantFold {}

impl Transform for ConstantFold {}

/// Common subexpression elimination pass replaces common subexpressions in
/// a basic block by their previously computed values. The pass will in most
/// cases introduce a new temporary storage location for the subexpression
/// before replacing its uses with the new variable.
struct CSE {}

impl Transform for CSE {}

/// Dead code elimination pass eliminates unused and unreachable instructions.
///
/// Because most optimizations can cause dead instructions this pass is run
/// after some optimizations multiple times until it converges i.e blocks
/// remain unchanged after a pass.
struct DCE {}

impl Transform for DCE {}

/// Strength reduction pass replaces some computations with cheaper and more
/// efficient equivalent alternatives.
struct StrengthReduce {}

impl Transform for StrengthReduce {}

/// Loop invariant code motion pass tries to remove as much code as possible
/// from the loop body.
struct LoopInvariantCodeMotion {}

impl Transform for LoopInvariantCodeMotion {}
