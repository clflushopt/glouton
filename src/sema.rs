//! Glouton semantic analyzhhr implementation.
//!
//! The semantic analyzer implements several passes on the AST to ensure type
//! correctness, reference correctness and overall soundness.

use crate::ast::{self, DeclType, Expr, Stmt};

/// TypeChecker implements an AST visitor responsible for ensuring the type
/// correctness of C0 programs.
pub struct TypeChecker<'a> {
    ast: &'a ast::AST,
}

impl TypeChecker<'_> {
    /// Returns the type of an expression.
    fn get_expr_type(&self, expr: &Expr) -> ast::DeclType {
        match expr {
            Expr::IntLiteral(_) => ast::DeclType::Int,
            Expr::BoolLiteral(_) => ast::DeclType::Bool,
            Expr::CharLiteral(_) => ast::DeclType::Char,
            _ => todo!("unimplemented type check for {:?}", expr),
        }
    }

    /// Returns the type of a declaration.
    fn get_decl_type(&self, stmt: &Stmt) -> ast::DeclType {
        ast::DeclType::Int
    }
}

macro_rules! typecheck {
    () => {};
}

impl<'a> ast::Visitor<()> for TypeChecker<'a> {
    fn visit_expr(&mut self, expr: &ast::Expr) -> () {
        match expr {
            ast::Expr::UnaryOp { operator, operand } => match operator {
                ast::UnaryOperator::Neg => {}
                ast::UnaryOperator::Not => {}
            },
            _ => todo!("Unimplemented type checking pass for Expr {:?}", expr),
        }
    }

    fn visit_stmt(&mut self, stmt: &ast::Stmt) -> () {
        match stmt {
            _ => todo!("Unimplemented type checking pass for statement {:?}", stmt),
        }
    }
}
