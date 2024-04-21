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
    // BlockID, assigned during initialization.
    id: BlockRef,
    // Instructions that constitute the basic block.
    instrs: Vec<ir::Instruction>,
    // Set of possible entry edges.
    entry_points: Vec<BlockRef>,
    // Set of possible exit edges.
    exit_points: Vec<BlockRef>,
}

impl BasicBlock {
    /// Create a new `BasicBlock` instance.
    fn new() -> Self {
        Self {
            id: BlockRef(0),
            instrs: Vec::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
        }
    }

    /// Returns the number of instructions in the block.
    pub fn len(&self) -> usize {
        self.instrs.len()
    }

    /// Returns a non-mutable reference to the block leader.
    pub fn leader(&self) -> Option<&ir::Instruction> {
        self.instrs.first()
    }

    /// Returns a non-mutable reference to the block terminator.
    pub fn terminator(&self) -> Option<&ir::Instruction> {
        self.instrs.last()
    }

    /// Returns a non-mutable reference to the block instructions.
    pub fn instructions(&self) -> &Vec<ir::Instruction> {
        &self.instrs
    }

    /// Drop the instruction at the given index and return it, has the same
    /// semantics as `Vec::remove`.
    pub fn remove(&mut self, index: usize) -> ir::Instruction {
        self.instrs.remove(index)
    }

    /// Append an instruction to the basic block.
    fn append(&mut self, inst: &ir::Instruction) {
        self.instrs.push(inst.clone())
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for inst in &self.instrs {
            writeln!(f, "{inst}")?;
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

/// A control flow graph constructed from a linear representation.
pub struct Graph {
    /// Nodes in the CFG.
    blocks: Vec<BasicBlock>,
    /// Edges in the CFG.
    edges: Vec<Edge>,
    // Mapping from labels to blocks built during the initial basic blocks
    // construction pass and then updated during the control flow graph pass
    // where blocks that don't have a label originally (non-target blocks) are
    // assigned a label.
    // TODO: Since we already have the notion of `BlockRef` refactor this
    // to get rid of `String` in both the keys and values.
    labels: HashMap<String, BlockRef>,
    // Successors of each basic block.
    // TODO: Since we already have the notion of `BlockRef` refactor this
    // to get rid of `String` in both the keys and values.
    successors: HashMap<String, Vec<String>>,
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CFG:  {{")?;
        for (label, _) in &self.labels {
            writeln!(f, "      {};", label)?;
        }

        for (label, succs) in &self.successors {
            for succ in succs {
                writeln!(f, "      {} -> {};", label, succ)?;
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
    ///
    /// Algorithm to form control flow graph, proceeds by connecting the basic
    /// blocks with each other using either :
    ///
    /// 1. Branch targets if the last instruction is a conditional branch.
    /// 2. Jump target if the last instruction is a conditional branch.
    /// 3. The next block in the list if the next instruction is neither of
    ///    the previous two.
    ///
    /// for block in blocks:
    ///     if block.last().is_branch():
    ///         add_edge(block, block.last().then_target())
    ///         add_edge(block, block.last().else_target())
    ///     if block.last().is_jump():
    ///         add_edge(block, block.last().target())
    ///     else:
    ///         add_edge(block, blocks.next())?
    pub fn new(program: &Vec<ir::Function>) -> Self {
        let mut blocks: Vec<BasicBlock> = Vec::new();

        for function in program {
            let mut bbs = Self::form_basic_blocks(function.instructions());
            blocks.append(&mut bbs)
        }

        Self {
            blocks,
            labels: HashMap::new(),
            successors: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Construct basic blocks from a list of instructions, the algorithm
    /// proceeds by iterating over the list of instructions and collecting
    /// leaders. Leaders are the first instructions in a basic block and
    /// we assign them based on whether the previous instruction terminates.
    pub fn form_basic_blocks(instrs: &[ir::Instruction]) -> Vec<BasicBlock> {
        let mut blocks = Vec::new();
        let mut block = BasicBlock::new();
        for inst in instrs {
            if inst.is_label() {
                if !block.instrs.is_empty() {
                    blocks.push(block)
                }
                match inst {
                    ir::Instruction::Label { .. } => {
                        block = BasicBlock::new();
                        block.append(inst);
                    }
                    _ => unreachable!(),
                }
            } else {
                block.append(inst);
                if inst.is_terminator() {
                    blocks.push(block);
                    block = BasicBlock::new();
                }
            }
        }
        blocks
    }
    /// Iterate over all the constructed basic blocks and assign them a label
    /// if they don't have one. Blocks that are control flow target will always
    /// have a label instruction as a leader.
    fn assign_labels_to_blocks(&mut self) {
        for (index, block) in self.blocks.iter().enumerate() {
            if block.instrs[0].is_label() {
                if let ir::Instruction::Label { ref name } = block.instrs[0] {
                    self.labels.insert(name.clone(), BlockRef(index));
                }
            } else {
                let label = format!("_LABEL_AUTO_{}", index);
                self.labels.insert(label, BlockRef(index));
            }
        }
    }

    /// Compute a list of succesors for each basic block in the graph.
    fn successors(&mut self) {
        let mut successors = HashMap::new();

        for (label, block_ref) in &self.labels {
            let mut succs = vec![];
            let last = self.blocks[block_ref.0]
                .terminator()
                .expect("Expected instruction found empty basic block");

            match last {
                ir::Instruction::Effect { args, .. } => match last.opcode() {
                    ir::OPCode::Jump | ir::OPCode::Branch => {
                        for arg in args {
                            if self.labels.get(arg).is_some() {
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
                        let block_name = self
                            .labels
                            .iter()
                            .map(|(k, &v)| {
                                if v == BlockRef(block_ref.0 + 1) {
                                    k.clone()
                                } else {
                                    unreachable!("No label found for block {}", block_ref.0 + 1)
                                }
                            })
                            .next()
                            .take();
                        if let Some(succ) = block_name {
                            succs = vec![succ.clone()];
                        }
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
    use crate::ir::IRBuilder;
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

                let mut irgen = IRBuilder::new(parser.ast(), &symbol_table);
                irgen.gen();
                println!("======== FUNCTIONS ========");
                println!("{:?}", irgen.functions());
                println!("======== GLOBALS ========");
                println!("{:?}", irgen.globals());
                println!("========  END  ========");

                let mut graph = Graph::new(irgen.functions());
                graph.assign_labels_to_blocks();
                // Compute successors.
                graph.successors();
                for (label, succs) in &graph.successors {
                    println!("Label {} | Successors {:?}", label, succs)
                }

                for (name, block_ref) in &graph.labels {
                    let block = &graph.blocks[block_ref.0];
                    for inst in &block.instrs {
                        println!("{}", inst);
                    }
                }
                println!("======== CFG ========");
                println!("{}", graph);
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
