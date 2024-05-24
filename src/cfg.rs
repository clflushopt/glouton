//! Implementation of a control flow graph for representing GIR programs.
use crate::ir;
use crate::ir::{BasicBlock, BlockRef};
use core::fmt;
use std::collections::HashMap;

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
            let mut bbs = function.form_basic_blocks();
            blocks.append(&mut bbs)
        }

        Self {
            blocks,
            labels: HashMap::new(),
            successors: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Iterate over all the constructed basic blocks and assign them a label
    /// if they don't have one. Blocks that are control flow target will always
    /// have a label instruction as a leader.
    fn assign_labels_to_blocks(&mut self) {
        for (index, block) in self.blocks.iter().enumerate() {
            if block.leader().is_some_and(|inst| inst.is_label()) {
                if let ir::Instruction::Label { ref name } = block.leader().unwrap() {
                    self.labels.insert(name.clone(), ir::BlockRef(index));
                }
            } else {
                let label = format!("_LABEL_AUTO_{}", index);
                self.labels.insert(label, ir::BlockRef(index));
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
                    for inst in block.instructions() {
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
