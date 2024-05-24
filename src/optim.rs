//! This module implements multiple transforms on the glouton IR
//! mostly focused on scalar optimizations.

use std::collections::HashSet;

use crate::ir::{self, Instruction, OPCode};

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
        let bbs = function.form_basic_blocks();

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

impl Transform for ConstantFold {
    fn run(&self, function: &mut ir::Function) {}
}

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
    /// Trivial Global DCE pass on a function returns `true` if any instructions
    /// are eliminated.
    pub fn tdce(function: &mut ir::Function) -> bool {
        let mut worklist = function.instructions_mut_vec().clone();
        let candidates = worklist.len();
        let mut use_defs = HashSet::new();

        for inst in &worklist {
            // Check for instruction uses, if an instruction is uses defs
            // we remove them from the `defs` set.
            match inst.args() {
                Some(args) => args.iter().for_each(|arg| {
                    use_defs.insert(arg.clone());
                }),
                _ => (),
            }
        }

        for inst in &mut worklist {
            if inst.dst().is_some_and(|dst| !use_defs.contains(dst)) {
                println!("Dropping {}", inst);
                let _ = std::mem::replace(inst, Instruction::nop());
            }
        }

        // Filter the worklist keeping only non-noop instructions, the usage
        // of `into_iter()` is necessary otherwise we end up with a container
        // `Vec<&Instruction>` which we can't "clone" back into the function
        // body.
        worklist
            .into_iter()
            .filter(|inst| inst.opcode() != OPCode::Nop)
            .collect::<Vec<_>>()
            .clone_into(&mut function.body);

        candidates != function.instructions().len()
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
                    println!("Pre-Pass function: {}", func);
                }
                for func in irgen.functions_mut() {
                    ident.run(func);
                    dce.run(func);
                }
                for mut func in irgen.functions_mut() {
                    println!("Post-Pass function: {}", func);
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

    test_optimization_pass!(
        can_trivially_dce_dead_blocks,
        r#"
            int main() {
                int a = 42;
                if (a > 43) {
                    int b = 313;
                    int c = 212;
                    int d = 111;
                    int e = 414;
                    int f = 515;
                    int g = 616;
                }
                return a;
            }
        "#,
        &vec![]
    );
}
