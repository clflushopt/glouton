//! This module implements multiple transforms on the glouton IR
//! mostly focused on scalar optimizations.

use std::collections::HashSet;

use crate::ir::{self};

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
        // let bbs = function.form_basic_blocks();

        // for bb in bbs {
        //     println!("{}", bb)
        // }
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

impl ConstantFold {}

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
    /// Trivial Global DCE pass on a function returns `true` if any instructions
    /// are eliminated.
    pub fn tdce(function: &mut ir::Function) -> bool {
        let worklist = function.instructions_mut();
        let candidates = worklist.len();
        let mut use_defs = HashSet::new();

        for inst in &mut *worklist {
            // Check for instruction uses, if an instruction is uses defs
            // we remove them from the `defs` set.
            match inst.operands() {
                (Some(lhs), Some(rhs)) => {
                    match (lhs, rhs) {
                        (
                            ir::Value::StorageLocation(lhs),
                            ir::Value::StorageLocation(rhs),
                        ) => {
                            use_defs.insert(lhs.clone());
                            use_defs.insert(rhs.clone());
                        }
                        // The only instructions that receive a constant literal
                        // as a value as a literal is `const` and it only has
                        // one operand.
                        _ => (),
                    }
                }
                (Some(operand), None) => match operand {
                    ir::Value::StorageLocation(operand) => {
                        use_defs.insert(operand.clone());
                    }
                    _ => (),
                },
                _ => (),
            }
        }

        for inst in &mut *worklist {
            if inst
                .destination()
                .is_some_and(|dst| !use_defs.contains(dst))
            {
                let _ = std::mem::replace(inst, ir::Instruction::Nop);
            }
        }
        // Remove all instructions marked as dead i.e replaced with `Nop`.
        function.remove_dead_instructions();

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

                for func in irgen.functions_mut() {
                    println!("Pre-Pass function: {}", func);
                }
                for func in irgen.functions_mut() {
                    ident.run(func);
                    dce.run(func);
                }
                for func in irgen.functions_mut() {
                    println!("Post-Pass function: {}", func);
                }

                let mut actual = "".to_string();
                for func in irgen.functions() {
                    // println!("{func}");
                    actual.push_str(format!("{func}").as_str());
                }
                // For readability trim the newlines at the start and end
                // of our IR text fixture.
                let expected = $expected
                    .strip_suffix("\n")
                    .and($expected.strip_prefix("\n"));
                assert_eq!(actual, expected.unwrap())
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
        r#"
@main: int {
   %v0: int = const 42
   ret %v0
}
"#
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
        r#"
@main: int {
   %v0: int = const 4
   a: int = id %v0
   %v1: int = const 2
   b: int = id %v1
   %v3: int = add a b
   d: int = id %v3
   ret d
}
"#
    );

    test_optimization_pass!(
        can_trivially_dce_dead_blocks,
        r#"
            int main() {
                int a = 42;
                int b = 313;
                int c = 212;
                int d = 111;
                int e = 414;
                int f = 515;
                int g = 616;
                return a;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 42
   a: int = id %v0
   ret a
}
"#
    );
}
