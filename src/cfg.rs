//! Implementation of a control flow graph for representing GIR programs.
use crate::ir::{self};
use core::fmt;
use std::collections::HashMap;

/// We follow the same approach in the AST when building the graph, it's
/// flattened and doesn't use pointers.
///
/// The first handle or reference we expose is a `BlockRef` which is used
/// to reference basic blocks (the nodes in the graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockRef(usize);

/// We also expose a handle to edges in the graph, since our edges are enums
/// that hold data (instruction, labels...) they need to have their own storage
/// in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    // Successors map.
    successors: HashMap<String, Vec<String>>,
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "digraph {} {{", "Example");
        for label in self.labels.keys() {
            writeln!(f, "      {};", label);
        }

        for (label, succs) in &self.successors {
            for succ in succs {
                writeln!(f, "       {} -> {};", label, succ);
            }
        }

        writeln!(f, "}}")
    }
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
    pub fn new(program: &Vec<ir::Function>) -> Self {
        let mut blocks = Vec::new();
        for func in program {
            let mut bbs = Self::form_basic_blocks(func.instructions());
            blocks.append(&mut bbs)
        }
        Self {
            blocks,
            labels: HashMap::new(),
            successors: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Construct basic blocks from a list of instructions.
    fn form_basic_blocks(instrs: &[ir::Instruction]) -> Vec<BasicBlock> {
        let mut blocks = Vec::new();
        let mut block = BasicBlock::new();
        for inst in instrs {
            if !inst.is_label() {
                block.append(inst);
                if inst.is_terminator() {
                    blocks.push(block);
                    block = BasicBlock::new();
                }
            }
            if inst.is_label() {
                if block.instrs.len() > 0 {
                    blocks.push(block)
                }
                match inst {
                    &ir::Instruction::Label { ref name } => {
                        block = BasicBlock::new();
                        block.append(inst);
                    }
                    _ => unreachable!(),
                }
            }
        }
        blocks
    }
    /// Build a map from labels to blocks.
    fn label_blocks(&mut self) {
        for (index, block) in self.blocks.iter().enumerate() {
            if block.instrs[0].is_label() {
                match block.instrs[0] {
                    ir::Instruction::Label { ref name } => {
                        self.labels.insert(name.clone(), BlockRef(index));
                    }
                    _ => (),
                }
            } else {
                let label = format!("_LABEL_AUTO_{}", index);
                self.labels.insert(label, BlockRef(index));
            }
        }
    }

    /// Produce a list of succesors for each basic block.
    fn successors(&mut self) {
        let mut successors = HashMap::new();

        for (label, block_ref) in &self.labels {
            let mut succs = vec![];
            let last = self.blocks[block_ref.0]
                .instrs
                .last()
                .expect("Expected instruction found empty basic block");

            match last {
                ir::Instruction::Effect { args, op } => match last.opcode() {
                    ir::OPCode::Jump | ir::OPCode::Branch => {
                        for arg in args {
                            if let Some(next) = self.labels.get(arg) {
                                succs.push(arg.clone())
                            }
                        }
                    }
                    ir::OPCode::Return => succs = vec![],
                    _ => (),
                },
                _ => {
                    if block_ref.0 == self.labels.len() - 1 {
                        succs = vec![]
                    } else {
                        let block_name = self.labels.iter().find_map(|(k, &val)| {
                            if val == BlockRef(block_ref.0 + 1) {
                                Some(k.clone())
                            } else {
                                unreachable!("No label found for block {}", block_ref.0 + 1)
                            }
                        });
                        succs = vec![block_name.unwrap()]
                    }
                }
            }
            successors.insert(label.clone(), succs);
        }

        self.successors = successors
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

                // println!("Instructions: ");
                // for func in irgen.program() {
                //     println!("{func}");
                // }
                let mut graph = Graph::new(irgen.program());
                graph.label_blocks();
                // Compuyte successors.
                graph.successors();
                for (label, succs) in &graph.successors {
                    println!("Label {} | Successors {:?}", label, succs)
                }

                println!("Graph :");
                println!("{}", graph);

                for (name, block_ref) in &graph.labels {
                    let block = &graph.blocks[block_ref.0];
                    println!("Block {}", name);
                    for inst in &block.instrs {
                        println!("{}", inst);
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
            } 
            return a + b;
        }
        "#,
        &vec![]
    );
}
