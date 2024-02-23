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
//! For expressions this is fine and doesn't cause an issue since all node
//! references will be backwards consuming it in order (left to right) or
//! out of order works the same.
//!
//!
//! a * b => [Named(a), Named(b), BinExpr(Mul, Ref(0), Ref(1))]
//!
//!
//! For statements this makes traversal less ergonomic as you will encounter
//! child nodes before their parent, which can be problematic when dealing
//! with scopes.
//!
//! Consider the following
//!
//!
//! {
//!     int a;
//!     int b;
//! }
//!
//!
//! The equivalent naive construction will yield the following :
//!
//!
//! [VarDecl(int, a), VarDecl(int, b), Block([Ref(0), Ref(1)])
//!
//!
//!
//! Processing the above left to right means seeing declarations out of context
//! i.e their scope, as such consumers will generate wrong output be it code
//! or textual representation.
//!
//! To remedy this issue our AST is built on three layered vectors, the first
//! layer represents the top level declaration and statements, these are what
//! our program is composed of. Scoped statements and declarations sit in the
//! second layer and finally expressions are at the last layer.
//!
//! The above example is laid out as such in memory :
//!
//!
//! declarations: [Block([RefStmt(0), RefStmt(1)])]
//! statements: [VarDecl(int, a), VarDecl(int, b)]
//!
//!
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
//! This approach is not new and has been used in `LuaJIT`, Zig, Sorbet, ECS
//! game engines and more, see [1] for more details.
//!
//! [1]: https://www.cs.cornell.edu/~asampson/blog/flattening.html

use core::fmt;

/// Node references are represented as `usize` handles to the AST arena entries
/// if space is a concern smaller handles can be used `u32` for example if you
/// assume at most 4 billion nodes per node kind.
///
/// `Ref` trait allows us to restrict which types can be used as handles
/// or node references.
pub trait Ref: Sized {
    fn new(reference: usize) -> Self;
    fn get(&self) -> usize;
}

/// `NodeRef` is a reference to an AST node, its a generic handle that wraps
/// a `usize` which in itself is an index into a `Vec` of AST nodes.
///
/// Since our pool is generic over either `Expr` or `Stmt` and we want to have
/// one reference kind of each, `NodeRef` acts as a phantom type that will be
/// later marked to reference either expressions or statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeRef<T> {
    inner: usize,
    _marker: std::marker::PhantomData<T>,
}

/// Implement the `Ref` trait for our `NodeRef` type.
impl<T> Ref for NodeRef<T> {
    fn new(inner: usize) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }

    fn get(&self) -> usize {
        self.inner
    }
}

/// `NodePool` represents a pool of AST nodes, each node pool is restricted
/// to an implementation of the `Ref` trait. This allows us to have compile
/// time type safe references for either expressions or statements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodePool<T, R: Ref> {
    nodes: Vec<T>,
    _marker: std::marker::PhantomData<R>,
}

/// Implementation of the `NodePool` exposes a `get` function to fetch an AST
/// node by its reference and an `add` function that appends a new AST node
/// to the pool.
impl<T, R: Ref> NodePool<T, R> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(4096),
            _marker: std::marker::PhantomData,
        }
    }

    /// Return a reference to a node given its `NodeRef`.
    #[must_use]
    pub fn get(&self, node_ref: R) -> Option<&T> {
        self.nodes.get(node_ref.get())
    }

    /// Push a new expression into the pool.
    fn put(&mut self, expr: T) -> R {
        let node_ref = self.nodes.len();
        self.nodes.push(expr);
        R::new(node_ref)
    }
}

impl<T, R: Ref> Default for NodePool<T, R> {
    fn default() -> Self {
        Self::new()
    }
}

/// `ExprRefMarker` is a phantom type marker for node references that reference
/// expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExprRefMarker;
/// `StmtRefMarker` is a phantom type marker for node references that reference
/// statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StmtRefMarker;
/// `DeclRefMarker` is a phantom type marker for node references that reference
/// declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeclRefMarker;

/// `ExprRef` is a reference to an `Expr` node in the expression pool.
pub type ExprRef = NodeRef<ExprRefMarker>;
/// `StmtRef` is a reference to a `Stmt` node in the statements pool.
pub type StmtRef = NodeRef<StmtRefMarker>;
/// `DeclRef` is a reference to `Decl` node in the declarations pool.
pub type DeclRef = NodeRef<DeclRefMarker>;

/// `ExprPool` is a node pool that holds `Expr` nodes.
type ExprPool = NodePool<Expr, ExprRef>;
/// `StmtPool` is a node pool that holds `Stmt` nodes.
type StmtPool = NodePool<Stmt, StmtRef>;
/// `DeclPool` is a node pool for `Decl` nodes.
type DeclPool = NodePool<Decl, DeclRef>;

// `SymbolPool` is a pool of source code symbols i.e identifiers used for
// string interning. The lifecycle of the `SymbolPool` is similar to that
// of the `ast` and since we repeatedly deal with strings in compilers e.g
// parsing, semantic analysis... this optimization allows us to reduce the
// number of allocations, make comparisons O(1) and only build a the string
// index once (during parsing).
//
// Inspired by LLVM IdentifierTable, matklad's interner implementation, Sorbet.
// TODO: Implement symbol pool with string interning for identifiers.
#[derive(Default)]
pub struct SymbolPool {}

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
            Self::Add => write!(f, "ADD"),
            Self::Sub => write!(f, "SUB"),
            Self::Mul => write!(f, "MUL"),
            Self::Div => write!(f, "DIV"),
            Self::Eq => write!(f, "EQU"),
            Self::Neq => write!(f, "NEQ"),
            Self::Gt => write!(f, "GT"),
            Self::Gte => write!(f, "GTE"),
            Self::Lt => write!(f, "LT"),
            Self::Lte => write!(f, "LTE"),
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
            Self::Neg => write!(f, "NEG"),
            Self::Not => write!(f, "NOT"),
        }
    }
}

/// Declaration types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeclType {
    Int,
    Char,
    Bool,
}

impl DeclType {
    /// Returns the default value for a declaration type.
    #[must_use]
    pub const fn default_value(&self) -> Expr {
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
            Self::Int => write!(f, "INT_TYPE"),
            Self::Char => write!(f, "CHAR_TYPE"),
            Self::Bool => write!(f, "BOOL_TYPE"),
        }
    }
}

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
    // Assignment expressions.
    Assignment {
        name: ExprRef,
        value: ExprRef,
    },
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
    LocalVar {
        decl_type: DeclType,
        name: String,
        value: ExprRef,
    },
    // Function arguments.
    FuncArg {
        decl_type: DeclType,
        name: String,
    },
    // Expression statements.
    Expr(ExprRef),
    // Blocks are sequence of statements.
    Block(Vec<StmtRef>),
    // If statements.
    If(ExprRef, StmtRef, Option<StmtRef>),
    // For loops.
    For(Option<ExprRef>, Option<ExprRef>, Option<ExprRef>, StmtRef),
    // While loops.
    While(Option<ExprRef>, Option<StmtRef>),
    // Empty statement.
    Empty,
}

/// Declaration nodes are used to represent declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decl {
    // Global variable declarations.
    GlobalVar {
        decl_type: DeclType,
        name: String,
        value: ExprRef,
    },
    // Function declarations.
    Function {
        name: String,
        return_type: DeclType,
        args: Vec<StmtRef>,
        body: StmtRef,
    },
}

/// `AST` represents the AST generated by the parser when processing a list
/// of tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AST {
    declarations: DeclPool,
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
    /// Visit a declaration.
    fn visit_decl(&mut self, decl: &Decl) -> T;
}

/// Accept and run a visitor over an AST.
pub fn visit<T>(ast: &AST, visitor: &'_ mut dyn Visitor<T>) {
    for decl in ast.declarations() {
        visitor.visit_decl(decl);
    }
}

/// Display implementation uses the `ASTDisplayer` to display the AST.
impl fmt::Display for AST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut displayer = ASTDisplayer::new(self);
        self.declarations()
            .iter()
            .try_for_each(|decl| write!(f, "{}", displayer.visit_decl(decl)))
    }
}

impl Default for AST {
    fn default() -> Self {
        Self::new()
    }
}

impl AST {
    /// Create a new empty AST.
    #[must_use]
    pub fn new() -> Self {
        Self {
            declarations: DeclPool::new(),
            statements: StmtPool::new(),
            expressions: ExprPool::new(),
        }
    }

    /// Return a non-mutable reference to the declaration pool.
    #[must_use]
    pub const fn declarations(&self) -> &Vec<Decl> {
        &self.declarations.nodes
    }

    /// Return a non-mutable reference to the statement pool.
    #[must_use]
    pub const fn statements(&self) -> &Vec<Stmt> {
        &self.statements.nodes
    }

    /// Return a non-mutable reference to the expression pool.
    #[must_use]
    pub const fn expressions(&self) -> &Vec<Expr> {
        &self.expressions.nodes
    }

    /// Push a new declaration node to the AST returning a reference to it.
    pub fn push_decl(&mut self, decl: Decl) -> DeclRef {
        self.declarations.put(decl)
    }

    /// Push a new statement node to the AST returning a reference to it.
    pub fn push_stmt(&mut self, stmt: Stmt) -> StmtRef {
        self.statements.put(stmt)
    }

    /// Push a new expression node to the AST returning a reference to it.
    pub fn push_expr(&mut self, expr: Expr) -> ExprRef {
        self.expressions.put(expr)
    }

    /// Fetches a declaration node by its reference, returning `None`
    /// if the declaration node deosn't exist.
    #[must_use]
    pub fn get_decl(&self, decl_ref: DeclRef) -> Option<&Decl> {
        self.declarations.get(decl_ref)
    }

    /// Fetches a statement node by its reference, returning `None`
    /// if the statement node deosn't exist.
    #[must_use]
    pub fn get_stmt(&self, stmt_ref: StmtRef) -> Option<&Stmt> {
        self.statements.get(stmt_ref)
    }

    /// Fetches an expression node by its reference, returning `None`
    /// if the expression doesn't exist.
    #[must_use]
    pub fn get_expr(&self, expr_ref: ExprRef) -> Option<&Expr> {
        self.expressions.get(expr_ref)
    }
}

/// `ASTDisplayer` walks the AST nodes and displays the individual expressions.
pub struct ASTDisplayer<'a> {
    ast: &'a AST,
}

impl<'a> ASTDisplayer<'a> {
    #[must_use]
    pub const fn new(ast: &'a AST) -> Self {
        Self { ast }
    }
}

impl<'a> Visitor<String> for ASTDisplayer<'a> {
    /// Visit an expression and return its textual representation.
    fn visit_expr(&mut self, expr: &Expr) -> String {
        match expr {
            &Expr::Assignment { name, value } => {
                let name = self.ast.get_expr(name).map_or_else(
                    || unreachable!("missing named expression for assignment"),
                    |name| self.visit_expr(name),
                );
                let value = self.ast.get_expr(value).map_or_else(
                    || unreachable!("missing named expression for assignment"),
                    |value| self.visit_expr(value),
                );
                format!("Assign({name}, {value})")
            }
            &Expr::IntLiteral(value) => value.to_string(),
            &Expr::BoolLiteral(value) => value.to_string(),
            &Expr::CharLiteral(value) => value.to_string(),
            &Expr::UnaryOp { operator, operand } => self.ast.get_expr(operand).map_or_else(
                || unreachable!("unary node is missing operand"),
                |operand| match operator {
                    UnaryOperator::Neg => format!("Neg({})", self.visit_expr(operand)),
                    UnaryOperator::Not => format!("Not({})", self.visit_expr(operand)),
                },
            ),
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
            &Expr::Grouping(expr_ref) => self.ast.get_expr(expr_ref).map_or_else(
                || unreachable!("unary node is missing operand"),
                |expr| format!("Grouping({})", self.visit_expr(expr)),
            ),
            Expr::Named(ref name) => {
                format!("Named({name})")
            }
            Expr::Call {
                name: name_ref,
                args,
            } => {
                if let Some(name) = self.ast.get_expr(*name_ref) {
                    let mut call_str = format!("Call({}, Args(", self.visit_expr(name));
                    for arg_ref in args.iter() {
                        if let Some(arg) = self.ast.get_expr(*arg_ref) {
                            let formatted_arg = self.visit_expr(arg);
                            call_str += &format!("{formatted_arg}, ");
                        }
                    }
                    call_str = call_str.trim_end_matches(", ").to_string();
                    call_str += "))";
                    call_str
                } else {
                    unreachable!("expected call expression to have at least one named value")
                }
            }
        }
    }
    /// Visit a statement and return its textual representation.
    fn visit_stmt(&mut self, stmt: &Stmt) -> String {
        match stmt {
            Stmt::Return(expr_ref) => self.ast.get_expr(*expr_ref).map_or_else(
                || unreachable!("Return statement is missing expression ref"),
                |expr| format!("Return({})", self.visit_expr(expr)),
            ),
            Stmt::LocalVar {
                decl_type,
                name,
                value,
            } => {
                let mut s = format!("VAR({decl_type}, {name}");
                self.ast.get_expr(*value).map_or_else(
                    || unreachable!("Missing expression in assignment"),
                    |expr| {
                        s += &format!(", {})", self.visit_expr(expr));
                    },
                );
                s
            }
            Stmt::Expr(expr_ref) => self.ast.get_expr(*expr_ref).map_or_else(
                || unreachable!("Expr statement is missing expression ref"),
                |expr| format!("Expr({})", self.visit_expr(expr)),
            ),
            Stmt::Block(stmts) => {
                let mut s = "Block {\n".to_string();
                for stmt_ref in stmts.iter() {
                    self.ast.get_stmt(*stmt_ref).map_or_else(
                        || unreachable!("Block is missing statement reference"),
                        |stmt| {
                            s += &format!("Stmt({}),\n", self.visit_stmt(stmt));
                        },
                    );
                }
                s += "}";
                s
            }
            Stmt::FuncArg { decl_type, name } => {
                format!("ARG({decl_type}, {name})")
            }
            Stmt::If(condition_ref, then_ref, else_ref) => {
                let cond = self.ast.get_expr(*condition_ref).map_or_else(
                    || unreachable!("missing condition for if statement"),
                    |cond_expr| self.visit_expr(cond_expr),
                );

                let then_block = self
                    .ast
                    .get_stmt(*then_ref)
                    .map_or_else(String::new, |body| self.visit_stmt(body));

                match else_ref {
                    Some(else_ref) => {
                        let else_block = self
                            .ast
                            .get_stmt(*else_ref)
                            .map_or_else(String::new, |else_block| self.visit_stmt(else_block));
                        format!("IF({cond}, {then_block}, {else_block})")
                    }

                    None => format!("IF({cond}, {then_block})"),
                }
            }
            Stmt::For(init_ref, cond_ref, iter_ref, body_ref) => {
                let init = match init_ref {
                    None => String::new(),
                    Some(expr_ref) => self.ast.get_expr(*expr_ref).map_or_else(
                        || unreachable!("missing expression in `for` statement"),
                        |init_expr| self.visit_expr(init_expr),
                    ),
                };

                let cond = match cond_ref {
                    None => String::new(),
                    Some(expr_ref) => self.ast.get_expr(*expr_ref).map_or_else(
                        || unreachable!("missing expression in `for` statement"),
                        |cond_expr| self.visit_expr(cond_expr),
                    ),
                };

                let iter = match iter_ref {
                    None => String::new(),
                    Some(expr_ref) => self.ast.get_expr(*expr_ref).map_or_else(
                        || unreachable!("missing expression in `for` statement"),
                        |iter_expr| self.visit_expr(iter_expr),
                    ),
                };

                let body = self.ast.get_stmt(*body_ref).map_or_else(
                    || unreachable!("expected `for` statement to have body"),
                    |body_stmt| self.visit_stmt(body_stmt),
                );

                format!("FOR({init}, {cond}, {iter}, {body})")
            }
            Stmt::Empty => unreachable!(
                "empty statement is a temporary placeholder and should not be in the ast"
            ),
            _ => todo!("Unimplemented display trait for function declaration"),
        }
    }
    /// Visit a declaration.
    fn visit_decl(&mut self, decl: &Decl) -> String {
        match decl {
            Decl::Function {
                name,
                return_type,
                args,
                body,
            } => {
                let mut args_str = String::new();
                for arg_ref in args.iter() {
                    if let Some(arg) = self.ast.get_stmt(*arg_ref) {
                        let arg = self.visit_stmt(arg);
                        args_str += &arg;
                        args_str += ", ";
                    }
                }

                args_str = args_str.trim_end_matches(", ").to_string();

                let body = self
                    .ast
                    .get_stmt(*body)
                    .map_or_else(|| unreachable!("function is missing body"), |body| body);
                format!(
                    "FUNCTION({}, {}, ARGS({}), {}",
                    name,
                    return_type,
                    args_str,
                    self.visit_stmt(body)
                )
            }
            Decl::GlobalVar {
                decl_type,
                name,
                value,
            } => {
                let mut s = format!("VAR({decl_type}, {name}");
                self.ast.get_expr(*value).map_or_else(
                    || unreachable!("Missing expression in assignment"),
                    |expr| {
                        s += &format!(", {})", self.visit_expr(expr));
                    },
                );
                s
            }
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
            let expr_ref = expr_pool.put(Expr::IntLiteral(42));
            let node_ref = stmt_pool.put(Stmt::Return(expr_ref));

            assert_eq!(expr_pool.get(expr_ref), Some(&Expr::IntLiteral(42)));
            assert_eq!(stmt_pool.get(node_ref), Some(&Stmt::Return(expr_ref)));
        }
    }
}
