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

/// We follow the same approach in the AST when building the intermediate
/// representation. The control flow graph is flattened and doesn't use
/// pointers.
///
/// The first handle or reference we expose is a `BlockRef` which is used
/// to reference basic blocks (the nodes in the graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockRef(u32);
/// We also expose a handle to edges in the graph, since our edges are enums
/// that hold data (instruction, labels...) they need to have their own storage
/// in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeRef(u32);

/// Instruction are the atomic operations of the linear IR part in GIR
/// they compose the building blocks of basic blocks.
///
/// Instructions in GIR are split into three groups, each group describe a set
/// of behaviors that the instructions implement.
///
/// Each behavior set describe certain semantics, enumerated below :
///
/// * Constant Operations: produce constant values to a destination and have
///   no side effects.
///
/// * Value Operations: produce dynamic values to a destination, consequently
///   they have no side effects.
///
/// * Effect Operations: produce behavior such as redirecting control flow and
///   thus have side effects.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    // The constant instruction produces a single constant integer value.
    Const,
}

/// `BasicBlock` is the atomic unit of graph IR and represent a node in
/// the control flow graph. Each basic block represent a single block of
/// instructions assumed to execute sequentially, branches, indirect jumps
/// are edges between basic blocks.
///
/// Each basic block has a set of input edges and a set of output edges
/// each edge can describe either an unconditional jump with a target label
/// a conditional jump with two target labels, a functional call or a function
/// return.
///
/// Edges hold a single instruction that describes which control flow operation
/// is executed on that particular edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    // Instructions that constitute the basic block.
    insts: Vec<Instruction>,
    // Set of possible entry edges.
    entry_points: Vec<EdgeRef>,
    // Set of possible exit edges.
    exit_points: Vec<EdgeRef>,
}

/// Edges in the control flow graph connect basic blocks and hold control
/// flow instructions and a target label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edge {
    Cond {
        target_if_true: BlockRef,
        target_if_false: BlockRef,
        cond: Instruction,
    },
    Jump {
        target: BlockRef,
    },
    Ret {
        target: BlockRef,
    },
}

/// IRGenerator implements the visitor pattern for the AST.
///
/// When generating IR from an AST there are two always present, specially
/// designated blocks: the entry block, used to indicate a program's entry
/// point and the exit block, used to designate the exit point of a program.
pub struct IRGenerator {
    // Nodes in the control flow graph constructed from the AST.
    blocks: Vec<BasicBlock>, // Edges in the control flow graph
    // Edges in the control flow graph.
    edges: Vec<Edge>,
}
