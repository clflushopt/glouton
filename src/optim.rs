//! This module implements multiple transforms on the glouton IR
//! mostly focused on scalar optimizations.

use std::collections::HashSet;

use crate::{
    cfg::Graph,
    ir::{self, Function, Instruction},
};

struct FunctionRewriter {}

impl FunctionRewriter {
    fn rewrite(f: &mut ir::Function, transform: &impl Transform) {
        transform.run(f)
    }
}

/// `Transform` trait is used to encapsulate the behavior of independant
/// optimizations executed on individual functions.
pub trait Transform {
    fn run(&self, function: &mut ir::Function) {}
}

/// Identity transform implements the `Transform` interface but actually
/// does nothing.
#[derive(Default, Debug)]
struct Identity {}

impl Transform for Identity {
    fn run(&self, function: &mut ir::Function) {
        // Get a list of basic blocks.
        let bbs = Graph::form_basic_blocks(function.instructions());

        for bb in bbs {
            println!("{}", bb)
        }
    }
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

impl DCE {
    /// Trivial DCE pass on a function returns `true` if any instructions are
    /// eliminated.
    pub fn tdce(function: &mut ir::Function) -> bool {
        let orig = function.instructions().len();
        let mut elim = false;
        println!("Starting tdce: eliminate {}", elim);
        // Get a list of basic blocks.
        let mut bbs = Graph::form_basic_blocks(function.instructions());
        // List of defs which are just variable definitions in a block.
        let mut use_defs: HashSet<String> = HashSet::new();

        // Mark all the instruction args as used.
        for bb in &bbs {
            for inst in bb.instructions() {
                // Check for instruction uses, if an instruction is uses defs
                // we remove them from the `defs` set.
                match inst.args() {
                    Some(args) => args.iter().for_each(|arg| {
                        use_defs.insert(arg.clone());
                    }),
                    _ => (),
                }
            }
        }

        for bb in &mut bbs {
            // Number of candidate instructions for elimination is initially set
            // to the number of instructions in the block since we will at most
            // eliminate all instructions in the block.
            let mut n_elim_candidates = bb.len();
            let mut droppable = vec![];
            // Create a local copy of the basic block we are processing.
            for (index, inst) in bb.instructions().iter().enumerate() {
                if inst.dst().is_some_and(|dst| !use_defs.contains(dst)) {
                    println!("Dropping {}", inst);
                    droppable.push(index);
                }
            }

            for idx in droppable {
                bb.remove(idx);
                // Decrement the number of elimnated candidates.
                n_elim_candidates -= 1;
            }

            println!("Post basic block:\n{}", bb);
        }

        // "Package" back the basic blocks into the parent function.
        // TODO: This is quite possibly the worst way to do this ideally
        // each `instruction` should hold a reference to its `parent`
        // basic block and a location into the parent function.
        // But that's a bigger change so for now, re-write the function.
        let mut dced = vec![];

        for bb in &bbs {
            for inst in bb.instructions() {
                dced.push(inst.clone())
            }
        }
        elim = dced.len() < orig;

        println!("Finished tdce: eliminate {}", elim);

        function.body = dced;
        elim
    }
}

impl Transform for DCE {
    /// Run dead code elimination over a function repeatedly until all
    /// convergence. The pass convergences when the number of candidates
    /// for elimination reaches 0.
    fn run(&self, function: &mut ir::Function) {
        while Self::tdce(function) {}
    }
}

/// Strength reduction pass replaces some computations with cheaper and more
/// efficient equivalent alternatives.
struct StrengthReduce {}

impl Transform for StrengthReduce {}

/// Loop invariant code motion pass tries to remove as much code as possible
/// from the loop body.
struct LoopInvariantCodeMotion {}

impl Transform for LoopInvariantCodeMotion {}

#[cfg(test)]
mod tests {
    use crate::ir::IRBuilder;
    use crate::optim::{Identity, Transform, DCE};
    use crate::parser::Parser;
    use crate::scanner::Scanner;
    use crate::sema::analyze;
    // Macro to generate test cases.
    macro_rules! test_optimization_pass {
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
                println!("{symbol_table}");

                let mut irgen = IRBuilder::new(parser.ast(), &symbol_table);
                irgen.gen();

                let ident = Identity {};
                let dce = DCE {};

                for mut func in irgen.functions_mut() {
                    ident.run(&mut func);
                    dce.run(&mut func);
                }
            }
        };
    }

    test_optimization_pass!(
        can_do_nothing_on_input_program,
        r#"
            int main() {
                return 42;
            }
        "#,
        &vec![]
    );

    test_optimization_pass!(
        can_trivially_dce_dead_store,
        r#"
            int main() {
                int a = 4;
                int b = 2;
                int c = 1;
                int d = a + b;
                return d;
            }
        "#,
        &vec![]
    );
}
