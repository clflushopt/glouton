//! Implementation of a control flow graph for representing GIR programs.
use crate::ir::{self};
use core::fmt;
use std::collections::HashMap;

/// We follow the same approach in the AST when building the graph, it's
/// flattened and doesn't use pointers.
///
/// The first handle or reference we expose is a `BlockRef` which is used
/// to reference basic blocks (the nodes in the graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockRef(usize);

/// We also expose a handle to edges in the graph, since our edges are enums
/// that hold data (instruction, labels...) they need to have their own storage
/// in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeRef(usize);

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
    instrs: Vec<ir::Instruction>,
    // Set of possible entry edges.
    entry_points: Vec<EdgeRef>,
    // Set of possible exit edges.
    exit_points: Vec<EdgeRef>,
}

impl BasicBlock {
    /// Create a new `BasicBlock` instance.
    fn new() -> Self {
        Self {
            instrs: Vec::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
        }
    }

    /// Append an instruction to the basic block.
    fn append(&mut self, inst: &ir::Instruction) {
        self.instrs.push(inst.clone())
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for inst in &self.instrs {
            writeln!(f, "{inst}");
        }
        Ok(())
    }
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
pub struct Graph {
    // Nodes in the control flow graph constructed from the AST.
    blocks: Vec<BasicBlock>,
    // Mapping from labels to blocks, this constructed during basic block
    // formation and then updated when we build a control flow graph assigning
    // one label for each block.
    labels: HashMap<String, BlockRef>,
    // Edges in the control flow graph.
    edges: Vec<Edge>,
}

impl Graph {
    /// Build a new control flow graph from an abstract program representation.
    ///
    /// The control flow graph construction proceeds by first constructing a
    /// list of basic blocks.
    ///
    /// Given the list of basic blocks and labels mappings we construct
    /// the control flow graph by building a dictionary of labels to successors
    /// where each successor is one of many possible control flow targets.
    fn new(program: &Vec<ir::Function>) -> Self {
        let mut blocks = Vec::new();
        for func in program {
            let mut fun_blocks = Self::form_basic_blocks(func.instructions());
            blocks.append(&mut fun_blocks)
        }
        Self {
            blocks,
            labels: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Construct basic blocks from a list of instructions.
    fn form_basic_blocks(instrs: &[ir::Instruction]) -> Vec<BasicBlock> {
        let mut blocks = Vec::new();
        let mut block = BasicBlock::new();
        // Map labels to `BlockRef`.
        let mut labels = HashMap::<String, BlockRef>::new();
        for inst in instrs {
            if !inst.is_label() {
                block.append(inst);
                if inst.is_terminator() {
                    blocks.push(block);
                    block = BasicBlock::new();
                }
            }
            if inst.is_label() {
                match inst {
                    &ir::Instruction::Label { ref name } => {
                        block.append(inst);
                        labels.insert(name.clone(), BlockRef(blocks.len()));
                    }
                    _ => unreachable!(),
                }
            }
        }
        blocks
    }
}

#[cfg(test)]
mod tests {
    use crate::cfg::Graph;
    use crate::ir::IRGenerator;
    use crate::parser::Parser;
    use crate::scanner::Scanner;
    use crate::sema::analyze;

    // Macro to generate test cases.
    macro_rules! test_form_basic_blocks {
        ($name:ident, $source:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let source = $source;
                let mut scanner = Scanner::new(source);
                let tokens = scanner
                    .scan()
                    .expect("expected test case source to be valid");
                let mut parser = Parser::new(&tokens);
                parser.parse();
                let symbol_table = analyze(parser.ast());
                println!("Symbol table : {symbol_table}");

                let mut irgen = IRGenerator::new(parser.ast(), &symbol_table);
                irgen.gen();

                println!("Instructions: ");
                for func in irgen.program() {
                    println!("{func}");
                }
                println!("Basic Blocks: ");
                for func in irgen.program() {
                    let blocks = Graph::form_basic_blocks(func.instructions());
                    for block in blocks {
                        println!("Start");
                        println!("{block}");
                        println!("End");
                    }
                }
            }
        };
    }

    test_form_basic_blocks!(
        can_generate_nested_scopes,
        r#"
            int main() {
                int i = 0;
                int x = 1;
                int y = 2;
                int z = x + y;
                return x;
            }
        "#,
        &vec![]
    );

    test_form_basic_blocks!(
        can_generate_if_block_without_else_branch,
        r#"
        int main() {
            int a = 42;
            int b = 17;
            if (a > b) {
                return a - b;
            } else {
                return a + b;
            }
            return 0;
        }
        "#,
        &vec![]
    );
}
