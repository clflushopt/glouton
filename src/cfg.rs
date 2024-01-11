//! Implementation of a control flow graph for representing GIR programs.
use crate::ir;

/// We follow the same approach in the AST when building the graph, it's
/// flattened and doesn't use pointers.
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

/// `BasicBlock` is the atomic unit of the graph IR and represent a node in
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
    insts: Vec<ir::Instruction>,
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
        cond: ir::Instruction,
    },
    Jump {
        target: BlockRef,
    },
    Ret {
        target: BlockRef,
    },
}

/// `ControlFlowGraph` is the control flow graph built from the `AbstractProgram`
/// representation and used for analysis and optimization passes.
pub struct ControlFlowGraph {
    // Nodes in the control flow graph constructed from the AST.
    blocks: Vec<BasicBlock>, // Edges in the control flow graph
    // Edges in the control flow graph.
    edges: Vec<Edge>,
}
