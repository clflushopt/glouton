//! Glouton semantic analyzhhr implementation.
//!
//! The semantic analyzer implements several passes on the AST to ensure type
//! correctness, reference correctness and overall soundness.
use std::{collections::HashMap, fmt, hash::Hash};

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

impl Symbol {
    // Create a new symbol.
    fn new(name: &str, t: DeclType, scope: Scope, kind: Kind, ord: usize) -> Self {
        Self {
            name: name.to_string(),
            t,
            scope,
            kind,
            ord,
        }
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

impl SymbolTable {
    /// Create a new symbol table instance.
    fn new() -> Self {
        Self {
            root: 0,
            current: 0,
            parent: 0,
            tables: Vec::new(),
        }
    }

    // Resolve a new symbol.
    #[must_use]
    pub fn resolve(&self, name: &str) -> Option<&Symbol> {
        self.find(name, self.current)
    }

    // Checks if a symbol exists in the table, used to mainly for checking
    // semantic correctness in test cases.
    #[must_use]
    pub fn exists(&self, name: &str) -> Option<&Symbol> {
        self.find(name, self.tables.len() - 1)
    }

    // Find a symbol by starting from the given index, the index should be in
    // the range of symbol tables stack.
    fn find(&self, name: &str, start: usize) -> Option<&Symbol> {
        // Get a reference to the current scope we're processing
        let mut idx = start;
        let mut current_scope_table = &self.tables[idx];
        loop {
            // Try and find the declaration in the current scope.
            for (ident, symbol) in current_scope_table {
                if ident == name {
                    return Some(symbol);
                }
            }
            // If we didn't find the declaration in the current scope
            // we check the parent.
            if idx.checked_sub(1).is_some() {
                idx -= 1;
                current_scope_table = &self.tables[idx];
            } else {
                break;
            }
        }
        None
    }

    // Bind a new symbol to the current scope.
    // # Panics
    // When `name` is already bound, binding fails.
    pub fn bind(&mut self, name: &str, sym: Symbol) {
        let tbl = &mut self.tables[self.current];

        match tbl.get(name) {
            Some(_) => {
                unreachable!("Name {name} is already bound to {sym}")
            }
            None => {
                tbl.insert(name.to_string(), sym);
            }
        }
    }

    // Returns a view of the symbol tables.
    const fn tables(&self) -> &Vec<HashMap<String, Symbol>> {
        &self.tables
    }

    // Return the index of the current scope.
    #[must_use]
    pub const fn scope(&self) -> usize {
        self.current
    }

    // Return how many symbols exist in the current scope.
    #[must_use]
    fn count(&self) -> usize {
        self.tables[self.current].len()
    }

    // Returns the expected position in the stack starting from the current
    // scope.
    #[must_use]
    fn stack_position(&self) -> usize {
        // Walk backwards until the global scope adding up the symbol count.
        let mut sym_count = 0;
        let mut idx = self.current;
        // The global scope is at index 0
        while idx > 0 {
            sym_count += self.tables[idx].len();
            idx -= 1;
        }

        sym_count
    }

    // Entering a new scope pushes a new symbol table into the stack.
    fn enter(&mut self) {
        let table = HashMap::new();
        self.tables.push(table);
        self.parent = self.current;
        self.current += 1;
    }

    // Leave the current scope returning the parent.
    fn exit(&mut self) {
        self.current = self.parent;
        if self.parent > 0 {
            self.parent -= 1;
        }
    }
}

/// Analyzer implements an AST visitor responsible for analysing the AST and
/// ensuring its semantic correctness.
pub struct Analyzer<'a> {
    ast: &'a ast::AST,
    table: SymbolTable,
}

impl<'a> Analyzer<'a> {
    /// Create a new `Analyzer` instance.
    pub fn new(ast: &'a ast::AST) -> Self {
        Self {
            ast,
            table: SymbolTable::new(),
        }
    }

    /// Get the expected scope of a symbol given our index in the symbol
    /// table stack.
    #[must_use]
    const fn scope(&self) -> Scope {
        match self.table.scope() {
            0 => Scope::Global,
            _ => Scope::Local,
        }
    }

    /// Define a new binding given a declaration.
    fn define(&mut self, stmt: &ast::Stmt) {
        match stmt {
            Stmt::VarDecl {
                decl_type, name, ..
            } => {
                let scope = self.scope();
                let position = match scope {
                    Scope::Local => self.table.stack_position(),
                    _ => 0,
                };
                let sym = Symbol::new(name, *decl_type, scope, Kind::Variable, position);
                self.table.bind(name, sym)
                // TODO: Ensure r-value type matches l-value declared type.
            }
            Stmt::FuncDecl {
                name,
                return_type,
                args,
                ..
            } => {
                // Bind the function name.
                let scope = self.scope();
                if self.scope() == Scope::Local {
                    unreachable!("function declarations not allowed in local scope")
                }
                let sym = Symbol::new(name, *return_type, scope, Kind::Function, 0);
                self.table.bind(name, sym);
                // Bind arguments.
                for arg in args {
                    match self.ast.get_stmt(*arg) {
                        Some(Stmt::FuncArg { decl_type, name }) => {
                            let sym = Symbol::new(
                                &name,
                                *decl_type,
                                Scope::Argument,
                                Kind::Variable,
                                self.table.stack_position(),
                            );
                            self.table.bind(name, sym)
                        }
                        arg @ _ => unreachable!(
                            "unxpected statement kind, expected function argument got {:?}",
                            arg
                        ),
                    }
                }
            }
            _ => unreachable!("unexpected `define` for {:?}", stmt),
        }
    }

    /// Lookup an existing binding by name, returns a `Symbol`
    /// if one is found otherwise none.
    fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.table.resolve(name)
    }

    /// Resolve an expression to check if it was properly defined.
    pub fn resolve(&self, expr: &ast::Expr) {
        match expr {
            ast::Expr::UnaryOp { operand, .. } => {
                if let Some(expr) = self.ast.get_expr(*operand) {
                    self.resolve(expr)
                } else {
                    unreachable!("expected unary expression to have operand")
                }
            }
            ast::Expr::BinOp { left, right, .. } => {
                if let (Some(lhs), Some(rhs)) =
                    (self.ast.get_expr(*left), self.ast.get_expr(*right))
                {
                    self.resolve(lhs);
                    self.resolve(rhs)
                }
            }
            ast::Expr::Named(name) => {
                if self.lookup(name).is_none() {
                    panic!("unknown value {name}")
                }
            }
            ast::Expr::Call { name, args } => {
                if let Some(name) = self.ast.get_expr(*name) {
                    self.resolve(name)
                }
                for arg_ref in args {
                    if let Some(arg) = self.ast.get_expr(*arg_ref) {
                        self.resolve(arg)
                    }
                }
            }
            ast::Expr::Assignment { name, .. } => match self.ast.get_expr(*name) {
                Some(ast::Expr::Named(name)) => if let Some(symbol) = self.lookup(&name) {},
                _ => panic!("expected l-value to be defined"),
            },
            ast::Expr::Grouping(group) => {
                if let Some(expr) = self.ast.get_expr(*group) {
                    self.resolve(expr)
                }
            }
            ast::Expr::IntLiteral(_) | ast::Expr::BoolLiteral(_) | ast::Expr::CharLiteral(_) => (),
        }
    }

    /// Core analysis routine, builds the symbol table and collect semantic
    /// errors to display later.
    fn analyze(&mut self, stmt: &ast::Stmt) {
        match stmt {
            Stmt::VarDecl {
                decl_type,
                name,
                value,
            } => {
                // Check if the symbol has been defined before.
                if self.lookup(name).is_some() {
                    panic!("symbol {name} is already defined")
                }
                // Typecheck that `value` has type `decl_type`.
            }
            _ => todo!("unimplemented `analyze`"),
        }
    }

    /// Run semantic analysis pass on the given AST.
    pub fn run(&mut self, ast: &Vec<ast::Stmt>) {
        for stmt in ast {
            self.analyze(stmt);
        }
    }

    /// Resolve the type of an expression, the type system we implement
    /// is very simple and enforces the following rules.
    /// - Values can only be assigned to variables of the same type.
    /// - Function parameters can only accept a value of the same type.
    /// - Return statements bind to the type of the returned values, the type must
    /// match the return type of the function.
    /// - All binary operators must have the same type on the lhs and rhs.
    /// - The equality operators can be applied to any type except `Void`
    /// and `Function` and always return a boolean.
    /// - The comparison operators can only be applied to integer values
    /// and always return boolean.
    /// - The boolean operators (!, ||, &&) can only be applied to boolean
    /// values and always return boolean.
    /// - The arithmetic operators can only be applied to integer values
    /// and always return an integer..
    fn typecheck(&self, expr: &ast::Expr) {}
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
/// correctness of input programs.
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
