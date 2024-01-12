//! Global intermediate representation definition and implementation.
//!
//! Glouton Intermediate Representation is a linear SSA based intermediate
//! representation. The intermediate representation has three core
//! instruction types, constant operations which produce constant values
//! value operations which take operands and produce values and effect based
//! operations which take operands and produce no values.
use core::fmt;

use crate::ast::{self, Visitor};

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
    Const { value: i32, dst: u32 },
    Add { lhs: u32, rhs: u32, dst: u32 },
    Sub { lhs: u32, rhs: u32, dst: u32 },
    Mul { lhs: u32, rhs: u32, dst: u32 },
    Div { lhs: u32, rhs: u32, dst: u32 },
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            &Self::Const { value, dst } => write!(f, "var_{dst}: const {value}"),
            &Self::Add { lhs, rhs, dst } => write!(f, "var_{dst} = var_{lhs} + var_{rhs}"),
            _ => todo!("Unimplemented display for instruction {}", self),
        }
    }
}

/// `IRGenerator` generates an intermediate representation by walking the AST
/// and building an `AbstractProgram` which is an abstract representation of
/// the entire program.
pub struct IRGenerator<'a> {
    abstract_program: Vec<Instruction>,
    ast: &'a ast::AST,
    var_count: u32,
}

impl<'a> IRGenerator<'a> {
    pub fn new(ast: &'a ast::AST) -> Self {
        Self {
            abstract_program: Vec::new(),
            ast,
            var_count: 0u32,
        }
    }

    fn program(&self) -> &Vec<Instruction> {
        &self.abstract_program
    }

    pub fn gen(&mut self) {
        for stmt in self.ast.statements() {
            let _ = match stmt {
                ast::Stmt::Return(expr_ref) => {
                    if let Some(expr) = self.ast.get_expr(*expr_ref) {
                        let _ = self.visit_expr(expr);
                    } else {
                        unreachable!("Return statement is missing expression ref")
                    }
                }
                ast::Stmt::Expr(expr_ref) => {
                    if let Some(expr) = self.ast.get_expr(*expr_ref) {
                        let _ = self.visit_expr(expr);
                    } else {
                        unreachable!("Expr statement is missing expression ref")
                    }
                }
            };
        }
        println!("IR dump {:?}", self.abstract_program);
    }
}

impl<'a> ast::Visitor<u32> for IRGenerator<'a> {
    fn visit_expr(&mut self, expr: &ast::Expr) -> u32 {
        match expr {
            &ast::Expr::IntLiteral(value) => {
                self.var_count += 1;
                self.abstract_program.push(Instruction::Const {
                    value,
                    dst: self.var_count,
                });
                self.var_count
            }
            &ast::Expr::UnaryOp { operator, operand } => {
                if let Some(operand) = self.ast.get_expr(operand) {
                    self.var_count += 1;
                    let dst = self.visit_expr(operand);
                    match operator {
                        ast::UnaryOperator::Neg => self.var_count,
                        ast::UnaryOperator::Not => self.var_count,
                    }
                } else {
                    unreachable!("unary node is missing operand")
                }
            }
            &ast::Expr::BinOp {
                left,
                operator,
                right,
            } => {
                if let (Some(left), Some(right)) =
                    (self.ast.get_expr(left), self.ast.get_expr(right))
                {
                    match operator {
                        ast::BinaryOperator::Add => {
                            self.var_count += 1;
                            let lhs = self.visit_expr(left);
                            self.var_count += 1;
                            let rhs = self.visit_expr(right);
                            let dst = self.var_count;
                            self.abstract_program
                                .push(Instruction::Add { lhs, rhs, dst });
                            dst
                        }
                        _ => todo!("Unimplemented IR generation for operator {operator}"),
                    }
                } else {
                    unreachable!("binary node is missing operand")
                }
            }
            &ast::Expr::Grouping(expr_ref) => {
                if let Some(expr) = self.ast.get_expr(expr_ref) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("unary node is missing operand")
                }
            }
            _ => todo!("Unimplemented display for Node {:?}", expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{IRGenerator, Instruction};
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
                println!("AST: {}", parser.ast());
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
