//! Glouton semantic analyzhhr implementation.
//!
//! The semantic analyzer implements several passes on the AST to ensure type
//! correctness, reference correctness and overall soundness.

use crate::ast;

/// TypeChecker implements an AST visitor responsible for ensuring the type
/// correctness of C0 programs.
pub struct TypeChecker<'a> {
    ast: &'a ast::AST,
}

impl<'a> ast::Visitor<()> for TypeChecker<'a> {
    fn visit_expr(&mut self, expr: &ast::Expr) -> () {
        match expr {
            _ => todo!("Unimplemented type checking pass for Expr {:?}", expr),
        }
    }

    fn visit_stmt(&mut self, stmt: &ast::Stmt) -> () {
        match stmt {
            _ => todo!("Unimplemented type checking pass for statement {:?}", stmt),
        }
    }
}
