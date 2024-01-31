//! Glouton semantic analyzhhr implementation.
//!
//! The semantic analyzer implements several passes on the AST to ensure type
//! correctness, reference correctness and overall soundness.

use core::fmt;
use std::collections::HashMap;

use crate::ast::{self, DeclType, Expr, Stmt};

/// Scope of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Scope {
    Local,
    Global,
    Argument,
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local => write!(f, "LOCAL"),
            Self::Global => write!(f, "GLOBAL"),
            Self::Argument => write!(f, "ARGUMENT"),
        }
    }
}

/// Symbol kind, function or variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Variable,
    Function,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function => write!(f, "FUNCTION"),
            Self::Variable => write!(f, "VARIABLE"),
        }
    }
}

/// Symbol in the AST represented as a tuple of `Name`, a `Scope`, `Kind`
/// and `Type`.  Each symbol is assigned a positional value which is used
/// to derive its ordinal position in an activation record.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Symbol {
    name: String,
    t: DeclType,
    scope: Scope,
    kind: Kind,
    ord: usize,
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Symbol({}, {}, {}, {})",
            self.name, self.t, self.scope, self.kind
        )
    }
}

/// The symbol table is responsible for tracking all declarations in effect
/// when a reference to a symbol is encountered and responsible for recording
/// all declared symbols.
///
/// `SymbolTable` is implemented as a stack of hash maps, a stack pointer is
/// always pointing to a `HashMap` that holds all declarations within a scope
/// when entering or leaving a scope the stack pointer is changed.
///
/// In order to simplify most operations we use three stack pointers, a root
/// stack pointer which is immutable and points to the global scope, a pointer
/// to the current scope and a pointer to the parent of the current scope.
///
/// This always us to walk the stack backwards when resolving symbols acting
/// like a linked list.
#[derive(Debug)]
struct SymbolTable {
    // Stack pointer to the global scope.
    root: usize,
    // Stack pointer to the current scope.
    current: usize,
    // Stack pointer to the parent of the current scope.
    parent: usize,
    // Symbol tables.
    tables: Vec<HashMap<String, Symbol>>,
}

/// Analyzer implements an AST visitor responsible for analysing the AST and
/// ensuring its semantic correctness.
pub struct Analyzer<'a> {
    ast: &'a ast::AST,
}

impl<'a> ast::Visitor<()> for Analyzer<'a> {
    fn visit_expr(&mut self, expr: &Expr) -> () {
        match expr {
            Expr::Named(identifier) => todo!("validate that identifier exists in the symbol table"),
            _ => todo!("unimplemented semantic analysis for expr {:?}", expr),
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) -> () {}
}

/// TypeChecker implements an AST visitor responsible for ensuring the type
/// correctness of C0 programs.
pub struct TypeChecker<'a> {
    ast: &'a ast::AST,
}

impl TypeChecker<'_> {
    /// Returns the type of an expression.
    fn get_expr_type(expr: &Expr) -> ast::DeclType {
        match expr {
            Expr::IntLiteral(_) => ast::DeclType::Int,
            Expr::BoolLiteral(_) => ast::DeclType::Bool,
            Expr::CharLiteral(_) => ast::DeclType::Char,
            _ => todo!("unimplemented type check for {:?}", expr),
        }
    }

    /// Returns the type of a declaration.
    fn get_decl_type(stmt: &Stmt) -> ast::DeclType {
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
