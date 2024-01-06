//! Global intermediate representation definition and implementation.
//!
//! Glouton Intermediate Representation is a small SSA based intermediate
//! representation. The intermediate representation describes three core
//! instruction types, constant operations which produce constant values
//! value operations which take operands and produce values and effect based
//! operations which take operands and produce no values.
//!
//! When generating a control flow graph or a sea of nodes representation from
//! the AST, basic blocks and nodes are composed of GIR instructions.
//!
//! In a way GIR is a hybrid intermediate representation that uses a linear IR
//! for the individual operations and a graph IR for the program representation
//! in our case we use both a CFG and a Sea of Nodes approach for the program
//! representation.

/// Instruction describes the supported instructed in GIR, instructions are
/// grouped by their semantics, constant operations produce constant values
/// to a destination, value operations produce dynamic values to a destination
/// from a group of operands and effect operations produce a behavior from a
/// given group of operands.
///
/// `%a: int = const 5` for example is a constant operation that produces an
/// integer value (5) to a target variable `%a`.
///
/// `br cond .label_1 .label_2` is an effect operation that produces no value
/// but describes the behavior of a branch condition to either one of the two
/// provided labels.
///
/// `%res = add %var1 %var2` is a value operaiton that produces the sum of two
/// variables.
pub enum Instruction {}

/// `BasicBlock` is used to represent a node in the control flow graph
/// each basic block represent a single block of sequences executed as
/// straight code and doesn't include any branches or jumps.
///
/// Edges between basic blocks describe transfer of control flow such as
/// conditional branches or direct jumps.
pub struct BasicBlock {
    insts: Vec<Instruction>,
}
