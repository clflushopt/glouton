//! Global intermediate representation definition and implementation.
//!
//! Glouton Intermediate Representation is a linear SSA based intermediate
//! representation. The intermediate representation has three core
//! instruction types, constant operations which produce constant values
//! value operations which take operands and produce values and effect based
//! operations which take operands and produce no values.
use core::fmt;
use std::fmt::Display;

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
    // The constant instruction produces a single constant integer value
    // to a destination variable, in case of literal values the destination
    // is either the variable defined in the AST in the case of assignment
    // expressions or a one created by the generator.
    Const {
        dst: u32,
        const_type: Type,
        const_value: Literal,
    },
    // Value operations.
    //
    // Arithmetic instructions.
    Add {
        lhs: u32,
        rhs: u32,
        dst: u32,
    },
    Sub {
        lhs: u32,
        rhs: u32,
        dst: u32,
    },
    Mul {
        lhs: u32,
        rhs: u32,
        dst: u32,
    },
    Div {
        lhs: u32,
        rhs: u32,
        dst: u32,
    },
    // Return instruction is an effect operation that transfers execution
    // back to the caller.
    Ret,
    // Label is a special instruction that acts as a marker.
    Label {
        // Label name.
        label: String,
    },
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Const {
                dst,
                const_type: _,
                const_value,
            } => write!(f, "var_{dst}: int = const {const_value}"),
            Self::Add { lhs, rhs, dst } => write!(f, "var_{dst} = var_{lhs} + var_{rhs}"),
            _ => todo!("Unimplemented display for instruction {}", self),
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
