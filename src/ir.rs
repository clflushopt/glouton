//! Global intermediate representation definition and implementation.
//!
//! Glouton Intermediate Representation is a linear SSA based intermediate
//! representation. The intermediate representation has three core
//! instruction types, constant operations which produce constant values
//! value operations which take operands and produce values and effect based
//! operations which take operands and produce no values.
use core::fmt;

use crate::ast::{self, Visitor};

/// Types used in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    // Integers, defaults to i32.
    Int,
    // Booleans.
    Bool,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Bool => write!(f, "bool"),
        }
    }
}

/// Literal values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Literal {
    /// Integers
    Int(i32),
    /// Booleans.
    Bool(bool),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(value) => write!(f, "{value}"),
            Self::Bool(value) => write!(f, "{value}"),
        }
    }
}

/// ConstOp are opcodes for constant operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstOp {
    /// `const` operation.
    Const,
}

impl fmt::Display for ConstOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const => write!(f, "const"),
        }
    }
}

/// EffectOp are opcodes for effect operations such as control flow operations
/// (indirect jumps, conditional branches, calls...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectOp {
    // Indirect jumps.
    Jump,
    // Condtional branches.
    Branch,
    // Function calls that don't produce values.
    Call,
    // Return statements.
    Return,
    // No operation.
    Nop,
}

impl fmt::Display for EffectOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jump => write!(f, "jmp"),
            Self::Branch => write!(f, "br"),
            Self::Call => write!(f, "call"),
            Self::Return => write!(f, "ret"),
            Self::Nop => write!(f, "nop"),
        }
    }
}

/// ValueOp are opcodes for value operations, a value operation is any value
/// producing operation. Such as arithemtic operations, comparison operations
/// and loads or stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueOp {
    // Arithmetic operators.
    Add,
    Sub,
    Mul,
    Div,
    // Comparison operators.
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    // Boolean operators.
    Not,
    And,
    Or,
    // Function calls that produce values.
    Call,
    // Identity operator.
    Id,
}

impl fmt::Display for ValueOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::Mul => write!(f, "mul"),
            Self::Div => write!(f, "div"),
            Self::Eq => write!(f, "eq"),
            Self::Neq => write!(f, "neq"),
            Self::Lt => write!(f, "lt"),
            Self::Gt => write!(f, "gt"),
            Self::Lte => write!(f, "lte"),
            Self::Gte => write!(f, "gte"),
            Self::Not => write!(f, "not"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Call => write!(f, "call"),
            Self::Id => write!(f, "id"),
        }
    }
}

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
    // Constant instructions are instructions that produce a single constant
    // typed value to a destination variable.
    Constant {
        // Destination variable name.
        dst: String,
        // Opcode for the instruction, currently only `const`.
        op: ConstOp,
        // Type of the variable.
        const_type: Type,
        // Literal value stored.
        value: Literal,
    },

    // Value instructions that produce a value from arguments where each opcode
    // enforces the rules for the arguments.
    Value {
        // List of argument names (variable, function or label).
        args: Vec<String>,
        // Destination variable name.
        dst: String,
        // Opcode for the instruction.
        op: ValueOp,
        // Type of the variable.
        op_type: Type,
    },
    // Effect instructions that incur a side effect.
    Effect {
        // List of argument names (variable, function or label).
        args: Vec<String>,
        // Opcode for the instruction.
        op: EffectOp,
    },
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant {
                dst,
                op,
                const_type,
                value,
            } => {
                write!(f, "{dst}: {const_type} = {op} {value};")
            }
            Self::Value {
                args,
                dst,
                op,
                op_type,
            } => {
                write!(f, "{dst}: {op_type} = {op}")?;
                for arg in args {
                    match op {
                        ValueOp::Call => write!(f, "@{arg}")?,
                        _ => write!(f, "{arg}")?,
                    }
                }
                write!(f, ";")
            }
            Self::Effect { args, op } => {
                write!(f, "{op}")?;
                for arg in args {
                    match op {
                        EffectOp::Jump => write!(f, ".{arg}")?,
                        EffectOp::Call => write!(f, "@{arg}")?,
                        EffectOp::Branch => write!(f, ".{arg}")?,
                        EffectOp::Return => write!(f, ".{arg}")?,
                        EffectOp::Nop => write!(f, ".")?,
                    }
                }
                write!(f, ";")
            }
        }
    }
}

impl Instruction {
    /// Emit a constant instruction.
    fn emit_const(dst: String, const_type: Type, value: Literal) -> Instruction {
        Instruction::Constant {
            dst,
            op: ConstOp::Const,
            const_type,
            value,
        }
    }

    /// Emit a jump instruction.
    fn emit_jmp(target: String) -> Instruction {
        Instruction::Effect {
            args: vec![target],
            op: EffectOp::Jump,
        }
    }

    /// Emit a branch instruction.
    fn emit_branch(cond: String, then_target: String, else_target: String) -> Instruction {
        Instruction::Effect {
            args: vec![cond, then_target, else_target],
            op: EffectOp::Branch,
        }
    }

    /// Emit a call instruction.
    fn emit_call(func: String, args: Vec<String>) -> Instruction {
        let mut call_args = vec![func];
        call_args.extend(args);

        Instruction::Effect {
            args: call_args,
            op: EffectOp::Call,
        }
    }

    /// Emit a return instruction.
    fn emit_ret(value: String) -> Instruction {
        Instruction::Effect {
            args: vec![value],
            op: EffectOp::Return,
        }
    }
}

/// `Argument` is a wrapper around a name and type it represents a single
/// function argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Argument {
    // Argument name.
    name: String,
    // Argument type.
    kind: Type,
}

impl fmt::Display for Argument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.kind)
    }
}

/// `Function` represents a function declaration in the AST, a `Function`
/// is composed as a linear sequence of GIR instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    // Function name.
    name: String,
    // List of arguments the function accepts.
    args: Vec<Argument>,
    // Body of the function as GIR instructions.
    body: Vec<Instruction>,
    // Return type of the function if any.
    return_type: Option<Type>,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.name)?;

        if !self.args.is_empty() {
            write!(f, "(")?;

            for (i, arg) in self.args.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{arg}")?;
            }
            write!(f, ")")?;
        }

        if let Some(return_type) = self.return_type {
            write!(f, ": {return_type}")?;
        }

        writeln!(f, " {{")?;

        for inst in &self.body {
            writeln!(f, "{inst}")?;
        }
        write!(f, "}}")
    }
}

/// `IRGenerator` generates an intermediate representation by walking the AST
/// and building a intermediate representation of the original program.
///
/// The result of the process is a `Vec` of IR generated functions and a global
/// scope used to hold global variable declarations.
pub struct IRGenerator<'a> {
    program: Vec<Function>,
    ast: &'a ast::AST,
    var_count: u32,
}

impl<'a> IRGenerator<'a> {
    #[must_use]
    pub const fn new(ast: &'a ast::AST) -> Self {
        Self {
            program: Vec::new(),
            ast,
            var_count: 0u32,
        }
    }

    /// Returns a non-mutable reference to the generated program.
    const fn program(&self) -> &Vec<Function> {
        &self.program
    }

    /// Emit a constant operation.
    fn emit_const_op() {}

    pub fn gen(&mut self) {
        for stmt in self.ast.declarations() {
            match stmt {
                ast::Stmt::Return(expr_ref) => self.ast.get_expr(*expr_ref).map_or_else(
                    || unreachable!("Return statement is missing expression ref"),
                    |expr| {
                        let _ = self.visit_expr(expr);
                    },
                ),
                ast::Stmt::VarDecl {
                    decl_type: _,
                    name: _,
                    value,
                } => self.ast.get_expr(*value).map_or_else(
                    || unreachable!("Variable declaration is missing assignmenet"),
                    |expr| {
                        let _ = self.visit_expr(expr);
                    },
                ),
                ast::Stmt::Expr(expr_ref) => self.ast.get_expr(*expr_ref).map_or_else(
                    || unreachable!("Expr statement is missing expression ref"),
                    |expr| {
                        let _ = self.visit_expr(expr);
                    },
                ),
                _ => todo!("unimplemented ir gen phase for stmt {:?}", stmt),
            };
        }
    }
}

impl<'a> ast::Visitor<u32> for IRGenerator<'a> {
    fn visit_stmt(&mut self, stmt: &ast::Stmt) -> u32 {
        match *stmt {
            _ => todo!("Unimplemented visitor for Node {:?}", stmt),
        }
    }
    fn visit_expr(&mut self, expr: &ast::Expr) -> u32 {
        match *expr {
            _ => todo!("Unimplemented visitor for Node {:?}", expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::IRGenerator;
    use crate::parser::Parser;
    use crate::scanner::Scanner;

    // Macro to generate test cases.
    macro_rules! test_ir_gen {
        ($name:ident, $source:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let source = $source;
                let mut scanner = Scanner::new(source);
                let tokens = scanner.scan().unwrap();
                let mut parser = Parser::new(&tokens);
                parser.parse();
                let mut irgen = IRGenerator::new(parser.ast());
                irgen.gen();
                let program = irgen.program();

                for inst in program {
                    println!("{inst}");
                }
            }
        };
    }

    test_ir_gen!(
        can_parse_return_statements,
        "return 1 + 2 + 3 + 4;",
        &vec![Instruction::Const { 0, 1 }]
    );
}
