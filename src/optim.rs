//! Implementation of several scalar optimization passes such as local value
//! numbering, dead code elimination...
use std::{
    collections::{HashMap, HashSet},
    env::var,
    ops::Deref,
};

use crate::{
    cfg::Graph,
    ir::{self, Instruction, Literal, OPCode, Symbol, Value},
};

/// `Transform` trait is used to encapsulate the behavior of independant
/// optimizations executed on individual functions.
pub trait Transform {
    fn run(&self, _: &mut ir::Function) {}
}

/// Identity transform implements the identity transformation which is a noop.
#[derive(Default, Debug)]
struct Identity {}

impl Transform for Identity {
    fn run(&self, _: &mut ir::Function) {}
}

/// Instruction combination pass executes over basic blocks and tries to
/// combine instructions that can be combined into one instruction.
///
/// Unlike local value numbering with `InstCombine` each pass is effectively
/// a re-write of an existing instruction or multiple instructions.
///
/// This implementation is mainly inspired by the way LLVM does it and contains
/// a strength reduction pass for some popular algebraic simplification.
struct InstCombine {}

impl Transform for InstCombine {}

/// Local Value Numbering pass builds a value numbering table that is then
/// re-used in several local optimizations such as dead code elimination
/// copy propagation, constant folding and common subexpression elimination.
struct LVN {}

/// Generic data structure that maps entries to monotonically increasing
/// numbers that act as row identifiers.
use std::hash::Hash;
#[derive(Debug, Clone)]
struct NumberTable<T: Clone + Hash + Eq>(HashMap<T, usize>, usize);

impl<T: Clone + Hash + Eq> NumberTable<T> {
    fn new() -> Self {
        Self {
            0: HashMap::new(),
            1: 0,
        }
    }

    fn get_or_insert(&mut self, key: &T) -> usize {
        if let Some(value) = self.0.get(key) {
            return *value;
        }
        let value = self.next_number();
        self.0.insert(key.clone(), value);
        value
    }

    fn next_number(&mut self) -> usize {
        let n = self.1;
        self.1 += 1;
        n
    }
}

impl LVN {
    /// Run the local value numbering pass to build the value numbering table
    /// then iteratively run peephole optimizations on using the table.
    fn run(&self, function: &mut ir::Function) {
        // First step when constructing the LVN is to form basic blocks
        // for the input function.
        let worklist = Graph::form_basic_blocks(function);

        // The second step involves iterating over the individual basic blocks
        // and building the value table for each storage location.
        //
        // We need two hashmaps to represent our table, one to map variable names
        // to value numbers and one to map expressions to their original variable
        // names. Since we use a 3-address code IR every assignments has an lhs
        // that is a variable name and every instruction we care about has variable
        // names as arguments.

        // ValueNumber act as row numbers in our value numbering table.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        struct ValueNumber(usize);

        // Environment maps variable names to value numbers.
        //
        // var2num
        // let environment: HashMap<Symbol, ValueNumber> = HashMap::new();
        let mut var2num: NumberTable<Symbol> = NumberTable::new();
        // value2number
        // let value_table: HashMap<Value, ValueNumber> = HashMap::new();
        let value2number: HashMap<Value, ValueNumber> = HashMap::new();
        // num2vars
        // let variables: HashMap<ValueNumber, Symbol> = HashMap::new();
        let mut num2vars: HashMap<ValueNumber, Symbol> = HashMap::new();
        // num2const
        // let constants: HashMap<ValueNumber, Literal> = HashMap::new();
        let num2constants: HashMap<ValueNumber, Literal> = HashMap::new();

        for block in worklist {
            // Compute the list of variable names that are read ebfore they are
            // written.
            let read_before_written =
                Self::read_before_writes(block.instructions());
            for variable in &read_before_written {
                let num = var2num.get_or_insert(&variable);
                num2vars.insert(ValueNumber(num), variable.clone());
            }
            println!("Var2Num : {:?}", var2num);
            println!("Num2Vars: {:?}", num2vars);

            for (inst, last_write) in
                block.instructions().iter().zip(&read_before_written)
            {
                // Get the list of operands for the instruction.
                let (lhs, rhs) = inst.operands();
                // Lookup the value number for each argument if they have one, otherwise
                // create new ones.
                let argnums = match (lhs, rhs) {
                    (
                        Some(Value::StorageLocation(lhs)),
                        Some(Value::StorageLocation(rhs)),
                    ) => {
                        let argnums = vec![
                            var2num.get_or_insert(lhs),
                            var2num.get_or_insert(rhs),
                        ];
                        argnums
                    }
                    _ => vec![],
                };
                // If the instruction does have operands use the constructed argnums
                // to build a new instruction.
                for argnum in argnums {
                    if let (Some(lhs), Some(rhs)) = inst.operands() {
                        println!(
                            "New instruction: {:?} {:?}",
                            inst.opcode(),
                            num2vars.get(&ValueNumber(argnum))
                        )
                    }
                }
            }
        }
    }

    fn last_writes(instrs: &[Instruction]) -> Vec<bool> {
        let mut last_written_to = vec![false; instrs.len()];
        let mut seen = HashSet::new();

        for (index, instr) in instrs.iter().enumerate() {
            match instr {
                Instruction::Id(dst, ..)
                | Instruction::Neg(dst, ..)
                | Instruction::Not(dst, ..)
                | Instruction::Const(dst, ..)
                | Instruction::Or(dst, ..)
                | Instruction::And(dst, ..)
                | Instruction::Add(dst, ..)
                | Instruction::Sub(dst, ..)
                | Instruction::Mul(dst, ..)
                | Instruction::Div(dst, ..)
                | Instruction::Eq(dst, ..)
                | Instruction::Neq(dst, ..)
                | Instruction::Gte(dst, ..)
                | Instruction::Gt(dst, ..)
                | Instruction::Lte(dst, ..)
                | Instruction::Lt(dst, ..) => {
                    if !seen.contains(&dst) {
                        last_written_to[index] = true;
                        seen.insert(dst);
                    }
                }
                _ => (),
            }
        }
        last_written_to
    }

    fn read_before_writes(instrs: &[Instruction]) -> Vec<Symbol> {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();

        for inst in instrs {
            let (lhs, rhs) = inst.operands();
            let dst = inst.destination();

            match (lhs, rhs) {
                (
                    Some(Value::StorageLocation(lhs)),
                    Some(Value::StorageLocation(rhs)),
                ) => {
                    let local_reads = HashSet::from([lhs, rhs]);
                    let local_reads = local_reads.difference(&writes);
                    for sym in local_reads {
                        reads.insert(sym.clone());
                    }
                }
                (Some(Value::StorageLocation(operand)), None)
                | (None, Some(Value::StorageLocation(operand))) => {
                    let local_reads = HashSet::from([operand]);
                    let local_reads = local_reads.difference(&writes);
                    for sym in local_reads {
                        reads.insert(sym.clone());
                    }
                }
                _ => (),
            }

            if let Some(dst) = dst {
                writes.insert(dst);
            }
        }

        println!("Writes: {:?}", writes);
        println!("Reads: {:?}", reads);
        reads.iter().cloned().cloned().collect::<_>()
    }

    /// Common subexpression elimination pass replaces common subexpressions in
    /// a basic block by their previously computed values. The pass will in most
    /// cases introduce a new temporary storage location for the subexpression
    /// before replacing its uses with the new variable.
    fn cse(&self) {}

    /// Constant folding and propagation pass targets expressions that can be
    /// evaluated at compile time and replaces them with the evaluation, once
    /// constants are folded a second sub-pass executes to propagate constants
    /// to their usage locations.
    fn fold(&self) {}
}

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

        candidates != function.len()
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
    use crate::optim::{Identity, Transform, DCE, LVN};
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

                let mut irgen = IRBuilder::new(parser.ast(), &symbol_table);
                irgen.build();

                let ident = Identity {};
                let dce = DCE {};
                let lvn = LVN {};

                for func in irgen.functions_mut() {
                    ident.run(func);
                    dce.run(func);
                    lvn.run(func);
                }

                let mut actual = "".to_string();
                for func in irgen.functions() {
                    actual.push_str(format!("{func}").as_str());
                }
                println!("{}", actual);
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
        can_trivially_dce_single_dead_store,
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
        can_trivially_dce_multiple_dead_stores,
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

    test_optimization_pass!(
        can_trivially_dce_add_unused_result,
        r#"
            int main() {
                int a = 1;
                int b = 2;
                int c = a + b;
                int d = a + b;
                return d;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 1
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
        can_trivially_dce_dead_store_and_unused_add_result,
        r#"
            int main() {
                int a = 4;
                int b = 2;
                int c = 1;
                int d = a + b;
                int e = c + d;
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
        can_trivially_dce_dead_store_in_different_blocks,
        r#"
            int main() {
                int a = 4;
                int b = 2;
                int c = 0;
                if (a < b) {
                    int c = a + b;
                } else {
                    int d = a - b;
                }
                return c;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 4
   a: int = id %v0
   %v1: int = const 2
   b: int = id %v1
   %v2: int = const 0
   c: int = id %v2
   %v3: bool = lt a b
   br %v3 .LABEL_0 .LABEL_1
   .LABEL_0
   %v4: int = add a b
   c: int = id %v4
   jmp .LABEL_2
   .LABEL_1
   jmp .LABEL_2
   .LABEL_2
   ret c
}
"#
    );

    test_optimization_pass!(
        can_lvn,
        r#"
            int main() {
                int a = 4;
                int b = 2;
                int c = a + b;
                int d = a - b;
                int e = c - d;
                return e;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 4
   a: int = id %v0
   %v1: int = const 2
   b: int = id %v1
   %v2: int = add a b
   c: int = id %v2
   %v3: int = sub a b
   d: int = id %v3
   %v4: int = sub c d
   e: int = id %v4
   ret e
}
"#
    );

    test_optimization_pass!(
        can_build_valid_read_write_dependency_vectors,
        r#"
            int main() {
                int a = 4;
                int b = 2;
                int sum1 = a + b;
                int sum2 = a + b;
                int prod1 = sum1 * sum2;
                
                sum1 = 0;
                sum2 = 0;

                int sum3 = a + b;
                int prod2 = sum3 * sum3;

                return prod2;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 4
   a: int = id %v0
   %v1: int = const 2
   b: int = id %v1
   %v7: int = add a b
   sum3: int = id %v7
   %v8: int = mul sum3 sum3
   prod2: int = id %v8
   ret prod2
}
"#
    );
}
