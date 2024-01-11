//! Global intermediate representation definition and implementation.
//!
//! Glouton Intermediate Representation is a linear SSA based intermediate
//! representation. The intermediate representation has three core
//! instruction types, constant operations which produce constant values
//! value operations which take operands and produce values and effect based
//! operations which take operands and produce no values.
use crate::ast;

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
}

/// `IRGenerator` generates an intermediate representation by walking the AST
/// and building an `AbstractProgram` which is an abstract representation of
/// the entire program.
pub struct IRGenerator<'a> {
    abstract_program: Vec<Instruction>,
    ast: &'a ast::AST,
}

impl<'a> IRGenerator<'a> {
    pub fn new(ast: &'a ast::AST) -> Self {
        Self {
            abstract_program: Vec::new(),
            ast,
        }
    }
}

impl<'a> ast::Visitor<()> for IRGenerator<'a> {
    fn visit_expr(&mut self, expr: &ast::Expr) -> () {
        match expr {
            &ast::Expr::IntLiteral(value) => self
                .abstract_program
                .push(Instruction::Const { value, dst: 0 }),
            &ast::Expr::UnaryOp { operator, operand } => {
                if let Some(operand) = self.ast.get_expr(operand) {
                    match operator {
                        ast::UnaryOperator::Neg => (),
                        ast::UnaryOperator::Not => (),
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
                            self.visit_expr(left);
                            self.visit_expr(right);
                            ()
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
