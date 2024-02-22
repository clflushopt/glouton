//! Glouton semantic analysis implementation.
//!
//! Semantic analysis follows the C0 rules and walks the AST to build a symbol
//! table which is used to ensure the validity and soundness of the source code.
//!
//! The first pass over the AST builds a symbol table and ensures that all
//! referenced symbols have been declared before being used. Another second pass
//! is used to type check the declarations and assignments.
use std::{collections::HashMap, fmt};

use crate::ast::{self, DeclType, Expr, Stmt};

/// Scope is used to localize the symbol table scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    Local,
    Global,
}

/// Symbols bind names from declarations and identifiers to attributes such as
/// type or ordinal position in the declaration stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Symbol {
    LocalVariable {
        name: String,
        t: DeclType,
        position: usize,
    },
    GlobalVariable {
        name: String,
        t: DeclType,
    },
    FunctionArgument {
        name: String,
        t: DeclType,
    },
    FunctionDefinition {
        name: String,
        t: DeclType,
    },
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Self::LocalVariable { name, t, position } => {
                write!(f, "LOCAL({}): {} @ {position}", name, t)
            }
            Self::GlobalVariable { name, t } => write!(f, "GLOBAL({}): {}", name, t),
            Self::FunctionArgument { name, t } => write!(f, "ARG({}): {}", name, t),
            Self::FunctionDefinition { name, t } => write!(f, "FUNCTION({}): {}", name, t),
        }
    }
}

/// The symbol table is responsible for tracking all declarations in effect
/// when a reference to a symbol is encountered. It's built by walking the AST
/// and recording all variable and function declarations.
///
/// `SymbolTable` is implemented as a stack of hash maps, a stack pointer is
/// always pointing to a `HashMap` that holds all declarations within a scope
/// when entering or leaving a scope the stack pointer is changed.
///
/// In order to simplify most operations we use three stack pointers, a root
/// stack pointer which is immutable and points to the global scope, a pointer
/// to the current scope and a pointer to the parent of the current scope.
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
            tables: vec![HashMap::new()],
        }
    }

    // Resolve a new symbol.
    #[must_use]
    pub fn resolve(&self, name: &str) -> Option<&Symbol> {
        self.find(name, self.current)
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

    // Returns an immutable view of the symbol tables.
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

/// Declaration analyzer is a specialized visitor invoked for processing
/// declarations. It is responsible for building the symbol table information
/// corresponding to variable name, function names, types...
pub struct DeclAnalyzer<'a> {
    // AST to analyze.
    ast: &'a ast::AST,
    // Constructed symbol table.
    table: SymbolTable,
}

impl<'a> DeclAnalyzer<'a> {
    /// Create a new `Analyzer` instance.
    pub fn new(ast: &'a ast::AST) -> Self {
        Self {
            ast,
            table: SymbolTable::new(),
        }
    }

    /// Return whether we are in a local or global scope.
    #[must_use]
    const fn scope(&self) -> Scope {
        match self.table.scope() {
            0 => Scope::Global,
            _ => Scope::Local,
        }
    }

    /// Return an immutable view to the symbol table.
    const fn symbol_table(&self) -> &SymbolTable {
        &self.table
    }

    /// Define a new binding given a variable or function declaration.
    fn define(&mut self, stmt: &ast::Stmt) {
        match stmt {
            Stmt::VarDecl {
                decl_type, name, ..
            } => {
                // TODO: Ensure r-value type matches l-value declared type.
                let symbol = match self.scope() {
                    Scope::Local => {
                        let position = self.table.stack_position();
                        Symbol::LocalVariable {
                            name: name.clone(),
                            t: *decl_type,
                            position,
                        }
                    }
                    Scope::Global => Symbol::GlobalVariable {
                        name: name.clone(),
                        t: *decl_type,
                    },
                };
                self.table.bind(name, symbol)
            }
            Stmt::FuncArg { decl_type, name } => {
                let symbol = match self.scope() {
                    Scope::Local => Symbol::FunctionArgument {
                        name: name.clone(),
                        t: *decl_type,
                    },
                    Scope::Global => unreachable!("Function arguments must be in a local scope"),
                };
                self.table.bind(name, symbol)
            }
            Stmt::FuncDecl {
                name, return_type, ..
            } => {
                match self.scope() {
                    Scope::Local => {
                        unreachable!("function declarations are not allowed in a local scope")
                    }
                    Scope::Global => {
                        // Bind the function definition.
                        let func_symbol = Symbol::FunctionDefinition {
                            name: name.clone(),
                            t: *return_type,
                        };
                        self.table.bind(name, func_symbol);
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

    /// Run semantic analysis pass on the given AST.
    pub fn run(&mut self) {
        ast::visit(self.ast, self)
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

/// `ast::Visitor` implementation for `DeclAnalyzer` processes only declarations
/// by effectively creating bindings and does nothing for the rest of AST nodes.
impl<'a> ast::Visitor<()> for DeclAnalyzer<'a> {
    fn visit_expr(&mut self, expr: &Expr) -> () {
        match expr {
            _ => (),
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) -> () {
        match stmt {
            decl @ (Stmt::VarDecl { .. } | Stmt::FuncArg { .. }) => self.define(decl),
            func_decl @ Stmt::FuncDecl { args, body, .. } => {
                // Process the function declaration.
                self.define(func_decl);
                // Enter the local scope and process the arguments and body.
                self.table.enter();
                for arg_ref in args {
                    if let Some(arg) = self.ast.get_stmt(*arg_ref) {
                        self.define(arg);
                    }
                }
                if let Some(func_body) = self.ast.get_stmt(*body) {
                    match func_body {
                        Stmt::Block(body) => {
                            for stmt_ref in body {
                                if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                                    self.visit_stmt(stmt)
                                }
                            }
                        }
                        _ => unreachable!("function body must be a block statement"),
                    }
                }
                // Exit the function scope.
                self.table.exit();
            }
            Stmt::Block(body) => {
                self.table.enter();
                for stmt_ref in body {
                    if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                        self.visit_stmt(stmt)
                    }
                }
                self.table.exit();
            }
            Stmt::If(.., body_ref, Some(else_ref)) => {
                if let Some(stmt) = self.ast.get_stmt(*body_ref) {
                    self.visit_stmt(stmt)
                }
                if let Some(stmt) = self.ast.get_stmt(*else_ref) {
                    self.visit_stmt(stmt)
                }
            }
            Stmt::For(.., stmt_ref) => {
                if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                    self.visit_stmt(stmt)
                }
            }
            Stmt::While(.., Some(stmt_ref)) => {
                if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                    self.visit_stmt(stmt)
                }
            }
            _ => (),
        }
    }
}

/// Semantics analyzer is a top level visitor for processing declarations
/// and doing smenatic checking on AST nodes.
struct SemanticsAnalyzer<'a> {
    ast: &'a ast::AST,
    symbol_table: &'a SymbolTable,
    current_scope: usize,
}

impl<'a> SemanticsAnalyzer<'a> {
    pub fn new(ast: &'a ast::AST, symbol_table: &'a SymbolTable) -> Self {
        Self {
            ast,
            symbol_table,
            current_scope: 0,
        }
    }

    /// Lookup an existing binding by name, returns a `Symbol`
    /// if one is found otherwise none.
    fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbol_table.resolve(name)
    }
    /// Enter a new scope by increment the current scope pointer.
    fn enter_scope(&mut self) {
        self.current_scope += 1
    }

    /// Exit the current scope, decrementing the current scope pointer.
    fn exit_scope(&mut self) {
        if self.current_scope > 0 {
            self.current_scope -= 1
        }
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

#[cfg(test)]
mod tests {
    use crate::parser::Parser;
    use crate::scanner::Scanner;
    use crate::sema::DeclAnalyzer;

    // Macro to generate test cases.
    macro_rules! test_decl_analyzer {
        ($name:ident, $source:expr ) => {
            #[test]
            fn $name() {
                let source = $source;
                let mut scanner = Scanner::new(source);
                let tokens = scanner
                    .scan()
                    .expect("expected test case source to be valid");
                let mut parser = Parser::new(&tokens);
                parser.parse();

                let mut decl_analyzer = DeclAnalyzer::new(parser.ast());
                decl_analyzer.run();

                for (ii, table) in decl_analyzer.symbol_table().tables().iter().enumerate() {
                    println!("Scope @ {}", ii);
                    for (name, symbol) in table {
                        println!("{} => {}", name, symbol);
                    }
                }
            }
        };
    }

    test_decl_analyzer!(
        can_process_single_declarations,
        "int main() { int a = 0; return a;}"
    );

    test_decl_analyzer!(
        can_process_multiple_declarations_single_scope,
        "int main() { int a = 0; int b = a; int c = b; int d = 1337; }"
    );

    test_decl_analyzer!(
        can_process_multiple_declarations_nested_scopes,
        "int main() { int a = 0; { int b = a; }  int c = b; int d = 1337;  }"
    );

    test_decl_analyzer!(
        can_process_declarations_with_arguments,
        "int main(int a, int b, int c) { int d = b; { int a = c; } }"
    );
}
