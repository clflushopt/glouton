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
//! `usize`.
//!
//! Another argument although less impressive at this scale is speed, because
//! fetches aren't done via pointers and because AST nodes have nice locality
//! if your arena fits in cache then walking it becomes much faster than going
//! through the pointer fetch road.
//!
//! One downside of this representation is that it requires being careful when
//! implementing consumers of the AST. This comes from the fact that the AST
//! is constructed bottom-up and so all child leafs will be located before
//! their parents in the flat representation.
//!
//! For expressions this works out and doesn't cause an issue since all node
//! references will be backwards, for example :
//!
//! a * b => [Named(a), Named(b), BinExpr(Mul, Ref(a), Ref(b))]
//!
//! For statements this isn't an issue while building the tree but consuming it
//! becomes less ergonomic, consider the following example :
//!
//! {
//!     int a;
//!     int b;
//! }
//!
//! When parsing blocks we encounter declarations first, if our node in the AST
//! that represents a block looks like `Block(Vec<Ref>)` then consuming this
//! naively will end up encountering the declarations *before* the actual block.
//!
//! To remedy this issue our AST is built on three layered vectors, the first
//! layer represents the top level declaration and statements, these are what
//! our program is composed of. Scoped statements and declarations sit in the
//! second layer and finally expressions are at the last layer.
//!
//! When walking the AST we just need to consume the declarations vector left
//! to right, since all references in it *flow downwards* and thus we don't
//! meet child nodes before their parents.
//!
//! One point that must need to be though of before using the approach is how
//! the ownership of references is "oriented", i.e is your lifetime represented
//! as a tree of resources or a graph. This is important because the direction
//! or path of ownership could hinder the design.
//!
//! In our case the AST represents a program, the root node is an entry point
//! and the program itself is a sequence of *declarations*. Where each statement
//! either represents a variable or function declaration followed by statements
//!
//!
//! This approach is not new and has been used in LuaJIT, Zig, Sorbet, ECS
//! game engines and more, see[1] for more details.
//!
//! [1]: https://www.cs.cornell.edu/~asampson/blog/flattening.html

use core::fmt;
use std::fmt::format;

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
    // Arithmetic operators.
    Add,
    Sub,
    Mul,
    Div,
    // Comparison operators.
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "ADD"),
            BinaryOperator::Sub => write!(f, "SUB"),
            BinaryOperator::Mul => write!(f, "MUL"),
            BinaryOperator::Div => write!(f, "DIV"),
            BinaryOperator::Eq => write!(f, "EQU"),
            BinaryOperator::Neq => write!(f, "NEQ"),
            BinaryOperator::Gt => write!(f, "GT"),
            BinaryOperator::Gte => write!(f, "GTE"),
            BinaryOperator::Lt => write!(f, "LT"),
            BinaryOperator::Lte => write!(f, "LTE"),
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
    // Function calls.
    Call {
        name: ExprRef,
        args: Vec<ExprRef>,
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
        body: StmtRef,
    },
    // Expression statements.
    Expr(ExprRef),
    // Blocks are sequence of statements.
    Block(Vec<StmtRef>),
    // If statements.
    If(ExprRef, StmtRef, Option<Vec<Stmt>>),
    // Loops are represented as While loops and For loops are de-sugarized into `While` loops.
    While(ExprRef, StmtRef),
    // Empty statement.
    Empty,
}

/// `AST` represents the AST generated by the parser when processing a list
/// of tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AST {
    declarations: StmtPool,
    statements: StmtPool,
    expressions: ExprPool,
}

/// AST visitor trait exposes the set of behaviors to be implemented by AST
/// consumers (semantic analyzer, type checker, IR generator...).
pub trait Visitor<T> {
    /// Visit an expression.
    fn visit_expr(&mut self, expr: &Expr) -> T;
    /// Visit a statement.
    fn visit_stmt(&mut self, stmt: &Stmt) -> T;
}

/// Display implementation uses the `ASTDisplayer` to display the AST.
impl fmt::Display for AST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut displayer = ASTDisplayer::new(self);
        for stmt in &self.declarations.nodes {
            write!(f, "{}", displayer.visit_stmt(&stmt))?;
        }
        Ok(())
    }
}

impl AST {
    /// Create a new empty AST.
    pub fn new() -> Self {
        Self {
            declarations: StmtPool::new(),
            statements: StmtPool::new(),
            expressions: ExprPool::new(),
        }
    }

    /// Return a non-mutable reference to the declaration pool.
    pub fn declaration(&self) -> &Vec<Stmt> {
        &self.declarations.nodes
    }

    /// Return a non-mutable reference to the statement pool.
    pub fn statements(&self) -> &Vec<Stmt> {
        &self.statements.nodes
    }

    /// Return a non-mutable reference to the expression pool.
    pub fn expressions(&self) -> &Vec<Expr> {
        &self.expressions.nodes
    }

    /// Push a new declaration node to the AST returning a reference to it.
    pub fn push_decl(&mut self, decl: Stmt) -> StmtRef {
        self.declarations.add(decl)
    }

    /// Push a new statement node to the AST returning a reference to it.
    pub fn push_stmt(&mut self, stmt: Stmt) -> StmtRef {
        self.statements.add(stmt)
    }

    /// Push a new expression node to the AST returning a reference to it.
    pub fn push_expr(&mut self, expr: Expr) -> ExprRef {
        self.expressions.add(expr)
    }

    /// Fetches a declaration node by its reference, returning `None`
    /// if the declaration node deosn't exist.
    pub fn get_decl(&self, decl_ref: StmtRef) -> Option<&Stmt> {
        self.declarations.get(decl_ref)
    }

    /// Fetches a statement node by its reference, returning `None`
    /// if the statement node deosn't exist.
    pub fn get_stmt(&self, stmt_ref: StmtRef) -> Option<&Stmt> {
        self.statements.get(stmt_ref)
    }

    /// Fetches an expression node by its reference, returning `None`
    /// if the expression doesn't exist.
    pub fn get_expr(&self, expr_ref: ExprRef) -> Option<&Expr> {
        self.expressions.get(expr_ref)
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
    /// Visit an expression and return its textual representation.
    fn visit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            &Expr::IntLiteral(value) => value.to_string(),
            &Expr::BoolLiteral(value) => value.to_string(),
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
                        BinaryOperator::Eq => {
                            format!(
                                "Equal({}, {})",
                                self.visit_expr(left),
                                self.visit_expr(right)
                            )
                        }
                        BinaryOperator::Neq => {
                            format!(
                                "NotEqual({}, {})",
                                self.visit_expr(left),
                                self.visit_expr(right)
                            )
                        }
                        BinaryOperator::Gt => {
                            format!(
                                "Greater({}, {})",
                                self.visit_expr(left),
                                self.visit_expr(right)
                            )
                        }
                        BinaryOperator::Gte => {
                            format!(
                                "GreaterEqual({}, {})",
                                self.visit_expr(left),
                                self.visit_expr(right)
                            )
                        }
                        BinaryOperator::Lt => {
                            format!(
                                "Lesser({}, {})",
                                self.visit_expr(left),
                                self.visit_expr(right)
                            )
                        }
                        BinaryOperator::Lte => {
                            format!(
                                "LesserEqual({}, {})",
                                self.visit_expr(left),
                                self.visit_expr(right)
                            )
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
            Expr::Named(name) => {
                let name = name.clone();
                format!("Named({name})")
            }
            Expr::Call {
                name: name_ref,
                args,
            } => {
                if let Some(name) = self.ast.get_expr(*name_ref) {
                    let mut call_str = format!("Call({}, Args(", self.visit_expr(name));
                    for arg in args {
                        match self.ast.get_expr(*arg) {
                            Some(arg) => {
                                let formatted_arg = self.visit_expr(arg);
                                call_str += &format!("{}, ", formatted_arg);
                            }
                            _ => (),
                        }
                    }
                    call_str = call_str.trim_end_matches(", ").to_string();
                    call_str += &format!("))");
                    call_str
                } else {
                    unreachable!("expected call expression to have at least one named value")
                }
            }
            _ => todo!("Unimplemented display for Node {:?}", expr),
        }
    }
    /// Visit a statement and return its textual representation.
    fn visit_stmt(&mut self, stmt: &Stmt) -> String {
        match stmt {
            Stmt::Return(expr_ref) => {
                if let Some(expr) = self.ast.get_expr(*expr_ref) {
                    format!("Return({})", self.visit_expr(&expr))
                } else {
                    unreachable!("Return statement is missing expression ref")
                }
            }
            Stmt::VarDecl {
                decl_type,
                name,
                value,
            } => {
                let mut s = format!("VAR({}, {}", decl_type, name);
                if let Some(expr) = self.ast.get_expr(*value) {
                    s += &format!(", {})", self.visit_expr(&expr)).to_string();
                } else {
                    unreachable!("Missing expression in assignment")
                }
                s
            }
            Stmt::Expr(expr_ref) => {
                if let Some(expr) = self.ast.get_expr(*expr_ref) {
                    format!("Expr({})", self.visit_expr(&expr))
                } else {
                    unreachable!("Expr statement is missing expression ref")
                }
            }
            Stmt::Block(stmts) => {
                let mut s = "Block {\n".to_string();
                for stmt_ref in stmts {
                    if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                        s += &format!("Stmt({}),\n", self.visit_stmt(&stmt)).to_string();
                    } else {
                        unreachable!("Block is missing statement ref")
                    }
                }
                s += "}";
                s
            }
            Stmt::FuncArg { decl_type, name } => {
                format!("ARG({}, {})", decl_type, name)
            }
            Stmt::FuncDecl {
                name,
                return_type,
                args,
                body,
            } => {
                let mut args_str = "".to_owned();
                for arg in args {
                    if let Some(arg) = self.ast.get_stmt(*arg) {
                        let arg_str = self.visit_stmt(arg);
                        args_str += &arg_str;
                        args_str += ", ";
                    }
                }

                args_str = args_str.trim_end_matches(", ").to_string();

                let body = match self.ast.get_stmt(*body) {
                    Some(body) => body,
                    None => unreachable!("function is missing body"),
                };
                format!(
                    "FUNCTION({}, {}, ARGS({}), {}",
                    name,
                    return_type,
                    args_str,
                    self.visit_stmt(body)
                )
            }
            Stmt::If(condition_ref, then_ref, else_ref) => {
                let cond = if let Some(cond_expr) = self.ast.get_expr(*condition_ref) {
                    format!("{}", self.visit_expr(cond_expr),)
                } else {
                    unreachable!("missing condition for if statement")
                };

                let then = if let Some(body) = self.ast.get_stmt(*then_ref) {
                    format!("{}", self.visit_stmt(body))
                } else {
                    "".to_string()
                };

                format!("IF({}, {})", cond, then)
            }
            Stmt::Empty => unreachable!(
                "empty statement is a temporary placeholder and should not be in the ast"
            ),
            _ => todo!("Unimplemented display trait for function declaration"),
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
