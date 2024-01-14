//! Implementation of glouton's abstract syntax tree.
//!
//! The glouton AST uses the apporach of a flat data structure where instead
//! of using traditional tree like structure where each node holds a pointer
//! in this case `Box<Node>`, nodes hold handles or reference indices to nodes
//! stored in an arena represented by a `Vec<Node>`.
//!
//! This data oriented approach has several pros, the first being that arenas
//! are borrow checker friendly. This is especially important in Rust where
//! an initial design might be invalidated because it doesn't have a borrow
//! checker friendly representation.
//!
//! The second argument for this is that complex self references and circular
//! references are not an issue anymore because there is no lifetime associated
//! with the reference you use, the lifetime is now associated with the entire
//! arena and the individual entries hold indices into the arena which are just
//! `u32`.
//!
//! Another argument although less impressive at this scale is speed, because
//! fetches aren't done via pointers and because AST nodes have nice locality
//! if your arena fits in cache then walking it becomes much faster than going
//! through the pointer fetch road.
//!
//! One point that must need to be though of before using the approach is how
//! the ownership of references is "oriented", i.e is your lifetime represented
//! as a tree of resources or a graph. This is important because the direction
//! or path of ownership could hinder the design.
//!
//! In our case the AST represents a program, the root node is an entry point
//! and the program itself is a sequence of *statements*. Where each statement
//! either represents control flow or expressions. Since expressions *will not*
//! reference *statements*, the AST can be represented as a tuple of `StmtPool`
//! and `ExprPool`.
//!
//! Where `StmtPool` holds the statement nodes, each node holds an `ExprRef`
//! that can reference an expression or a `StmtRef` that references statements.
//!
//! This approach is not new and has been used in LuaJIT, Zig, Sorbet, ECS
//! game engines and more, see[1] for more details.
//!
//! [1]: https://www.cs.cornell.edu/~asampson/blog/flattening.html

use core::fmt;

/// Node references are represented as `usize` handles to the AST arena
/// this avoides type casting everytime we want to access a node and down
/// casting when building references from indices.
///
/// `StmtRef` is used to reference statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StmtRef(usize);
/// `ExprRef` is used to reference expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExprRef(usize);

/// `ExprPool` represents an arena of AST expression nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprPool {
    nodes: Vec<Expr>,
}

impl ExprPool {
    /// Create a new node pool with a pre-allocated capacity.
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(4096),
        }
    }

    /// Return a reference to a node given its `NodeRef`.
    pub fn get(&self, node_ref: ExprRef) -> Option<&Expr> {
        self.nodes.get(node_ref.0)
    }

    /// Push a new expression into the pool.
    fn add(&mut self, expr: Expr) -> ExprRef {
        let node_ref = self.nodes.len();
        self.nodes.push(expr);
        ExprRef(node_ref)
    }
}

/// `StmtPool` represents an arena of AST statement nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StmtPool {
    nodes: Vec<Stmt>,
}

impl StmtPool {
    /// Create a new node pool with a pre-allocated capacity.
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(4096),
        }
    }

    /// Return a reference to a node given its `NodeRef`.
    pub fn get(&self, node_ref: StmtRef) -> Option<&Stmt> {
        self.nodes.get(node_ref.0)
    }

    /// Push a new expression into the pool.
    fn add(&mut self, stmt: Stmt) -> StmtRef {
        let node_ref = self.nodes.len();
        self.nodes.push(stmt);
        StmtRef(node_ref)
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "ADD"),
            BinaryOperator::Sub => write!(f, "SUB"),
            BinaryOperator::Mul => write!(f, "MUL"),
            BinaryOperator::Div => write!(f, "DIV"),
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Neg,
    Not,
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOperator::Neg => write!(f, "NEG"),
            UnaryOperator::Not => write!(f, "NOT"),
        }
    }
}

/// Declaration types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclType {
    Int,
    Char,
    Bool,
}

impl DeclType {
    /// Returns the default value for a declaration type.
    pub fn default_value(&self) -> Expr {
        match self {
            Self::Int => Expr::IntLiteral(0),
            Self::Char => Expr::CharLiteral('\0'),
            Self::Bool => Expr::BoolLiteral(false),
        }
    }
}

impl fmt::Display for DeclType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeclType::Int => write!(f, "INT_TYPE"),
            DeclType::Char => write!(f, "CHAR_TYPE"),
            DeclType::Bool => write!(f, "BOOL_TYPE"),
        }
    }
}

/// Function arguments

/// Expression nodes are used to represent expressions.
/// TODO make Expr homogenous by storing `LiteralRef`, `StringRef` and so on
/// in a separate storage array stored in the AST.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    // Named values (variables),
    Named(String),
    // Integer literal values.
    IntLiteral(i32),
    // Boolean literal values.
    BoolLiteral(bool),
    // Char literal values.
    CharLiteral(char),
    // Grouping expressions (parenthesised expressions).
    Grouping(ExprRef),
    // Binary operations (arithmetic, boolean, bitwise).
    BinOp {
        left: ExprRef,
        operator: BinaryOperator,
        right: ExprRef,
    },
    // Unary operations (boolean not and arithmetic negation).
    UnaryOp {
        operator: UnaryOperator,
        operand: ExprRef,
    },
}

/// Statement nodes are used to represent statements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    // Return statements.
    Return(ExprRef),
    // Variable declarations.
    VarDecl {
        decl_type: DeclType,
        name: String,
        value: ExprRef,
    },
    // Function arguments.
    FuncArg {
        decl_type: DeclType,
        name: String,
    },
    // Function declarations.
    FuncDecl {
        name: String,
        return_type: DeclType,
        args: Vec<StmtRef>,
        body: Vec<StmtRef>,
    },
    // Expression statements.
    Expr(ExprRef),
    // Empty statement.
    Empty,
}

/// `AST` represents the AST generated by the parser when processing a list
/// of tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AST {
    statements: StmtPool,
    expressions: ExprPool,
}

/// AST visitor trait exposes the set of behaviors to be implemented by AST
/// consumers (semantic analyzer, type checker, IR generator...).
pub trait Visitor<T> {
    /// Visit an expression.
    fn visit_expr(&mut self, expr: &Expr) -> T;
}

/// Display implementation uses the `ASTDisplayer` to display the AST.
impl fmt::Display for AST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut displayer = ASTDisplayer::new(self);
        for stmt in &self.statements.nodes {
            let _ = match stmt {
                Stmt::Return(expr_ref) => {
                    if let Some(expr) = self.get_expr(*expr_ref) {
                        write!(f, "Return({})", displayer.visit_expr(&expr))
                    } else {
                        unreachable!("Return statement is missing expression ref")
                    }
                }
                Stmt::VarDecl {
                    decl_type,
                    name,
                    value,
                } => {
                    write!(f, "VAR({}, {}", decl_type, name)?;
                    if let Some(expr) = self.get_expr(*value) {
                        write!(f, ", {})", displayer.visit_expr(&expr))
                    } else {
                        unreachable!("Missing expression in assignment")
                    }
                }
                Stmt::Expr(expr_ref) => {
                    if let Some(expr) = self.get_expr(*expr_ref) {
                        write!(f, "Expr({})", displayer.visit_expr(&expr))
                    } else {
                        unreachable!("Expr statement is missing expression ref")
                    }
                }
                Stmt::FuncArg { decl_type, name } => {
                    write!(f, "ARG({}, {})", decl_type, name)
                }
                Stmt::FuncDecl {
                    name,
                    return_type,
                    args,
                    body,
                } => {
                    todo!("Unimplemented display trait for function declaration")
                }
                Stmt::Empty => unreachable!(
                    "empty statement is a temporary placeholder and should not be in the ast"
                ),
            };
        }
        Ok(())
    }
}

impl AST {
    /// Create a new empty AST.
    pub fn new() -> Self {
        Self {
            statements: StmtPool::new(),
            expressions: ExprPool::new(),
        }
    }

    /// Return a non-mutable reference to the statement pool.
    pub fn statements(&self) -> &Vec<Stmt> {
        &self.statements.nodes
    }

    /// Return a non-mutable reference to the expression pool.
    pub fn expressions(&self) -> &Vec<Expr> {
        &self.expressions.nodes
    }

    /// Push a new statement node to the AST returning a reference to it.
    pub fn push_stmt(&mut self, stmt: Stmt) -> StmtRef {
        self.statements.add(stmt)
    }

    /// Push a new expression node to the AST returning a reference to it.
    pub fn push_expr(&mut self, expr: Expr) -> ExprRef {
        self.expressions.add(expr)
    }

    /// Fetches an expression node by its reference, returning `None`
    /// if the expression doesn't exist.
    pub fn get_expr(&self, expr_ref: ExprRef) -> Option<&Expr> {
        self.expressions.get(expr_ref)
    }

    /// Fetches a statement node by its reference, returning `None`
    /// if the statement node deosn't exist.
    pub fn get_stmt(&self, stmt_ref: StmtRef) -> Option<&Stmt> {
        self.statements.get(stmt_ref)
    }
}

/// `ASTDisplayer` walks the AST nodes and displays the individual expressions.
struct ASTDisplayer<'a> {
    ast: &'a AST,
}

impl<'a> ASTDisplayer<'a> {
    pub fn new(ast: &'a AST) -> Self {
        Self { ast }
    }
}

impl<'a> Visitor<String> for ASTDisplayer<'a> {
    fn visit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            &Expr::IntLiteral(value) => value.to_string(),
            &Expr::UnaryOp { operator, operand } => {
                if let Some(operand) = self.ast.get_expr(operand) {
                    match operator {
                        UnaryOperator::Neg => format!("Neg({})", self.visit_expr(operand)),
                        UnaryOperator::Not => format!("Not({})", self.visit_expr(operand)),
                    }
                } else {
                    unreachable!("unary node is missing operand")
                }
            }
            &Expr::BinOp {
                left,
                operator,
                right,
            } => {
                if let (Some(left), Some(right)) =
                    (self.ast.get_expr(left), self.ast.get_expr(right))
                {
                    match operator {
                        BinaryOperator::Add => {
                            format!("Add({}, {})", self.visit_expr(left), self.visit_expr(right))
                        }
                        BinaryOperator::Sub => {
                            format!("Sub({}, {})", self.visit_expr(left), self.visit_expr(right))
                        }
                        BinaryOperator::Mul => {
                            format!("Mul({}, {})", self.visit_expr(left), self.visit_expr(right))
                        }
                        BinaryOperator::Div => {
                            format!("Div({}, {})", self.visit_expr(left), self.visit_expr(right))
                        }
                    }
                } else {
                    unreachable!("binary node is missing operand")
                }
            }
            &Expr::Grouping(expr_ref) => {
                if let Some(expr) = self.ast.get_expr(expr_ref) {
                    format!("Grouping({})", self.visit_expr(expr))
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
    use crate::ast::{ExprPool, StmtPool};

    use super::{Expr, Stmt};

    #[test]
    fn can_create_and_use_node_pool() {
        let mut expr_pool = ExprPool::new();
        let mut stmt_pool = StmtPool::new();

        for _ in 0..100 {
            let expr_ref = expr_pool.add(Expr::IntLiteral(42));
            let node_ref = stmt_pool.add(Stmt::Return(expr_ref));

            assert_eq!(expr_pool.get(expr_ref), Some(&Expr::IntLiteral(42)));
            assert_eq!(stmt_pool.get(node_ref), Some(&Stmt::Return(expr_ref)));
        }
    }
}
