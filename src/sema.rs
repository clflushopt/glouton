//! Semantic analysis follows the C0 rules and walks the AST to build a symbol
//! table which is used to ensure the validity and soundness of the source code.
//!
//! The first pass over the AST builds a symbol table and ensures that all
//! referenced symbols have been declared before being used. Another second pass
//! is used to type check the declarations and assignments.
//!
//! This implementation of semantic analysis mostly focuses on type correctness
//! and general soundness. Reachability analysis is currently not implemented.
use std::{collections::HashMap, fmt};

use crate::ast::{self, Decl, DeclType, Expr, Ref, Stmt};

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
        args: Vec<DeclType>,
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
            Self::FunctionDefinition { name, t, .. } => write!(f, "FUNCTION({}): {}", name, t),
        }
    }
}

impl Symbol {
    /// Return the symbol declaration type.
    fn t(&self) -> DeclType {
        match self {
            Self::LocalVariable { t, .. } => *t,
            Self::GlobalVariable { t, .. } => *t,
            Self::FunctionArgument { t, .. } => *t,
            Self::FunctionDefinition { t, .. } => *t,
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
#[derive(Debug, Clone)]
pub struct SymbolTable {
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

    // Find a symbol by starting from the given index, the index should be in
    // the range of symbol tables stack.
    fn find(&self, name: &str, start: usize) -> Option<&Symbol> {
        // Get a reference to the current scope we're processing
        let mut idx = start;
        let mut current_scope_table = &self.tables[idx];
        while idx >= self.root {
            // Try and find the declaration in the current scope.
            for (ident, symbol) in current_scope_table {
                if ident == name {
                    return Some(symbol);
                }
            }
            // If we didn't find the declaration in the current scope
            // we check the parent.
            idx -= 1;
            current_scope_table = &self.tables[idx];
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

    // Return the index of the current scope.
    #[must_use]
    pub const fn scope(&self) -> usize {
        self.current
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

    /// Define a local binding.
    fn define_local_binding(&mut self, stmt: &ast::Stmt) {
        match stmt {
            Stmt::LocalVar {
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
            _ => unreachable!("unexpected binding : {:?}", stmt),
        }
    }

    /// Define a new global binding given a variable or function declaration.
    fn define_global_binding(&mut self, decl: &ast::Decl) {
        match decl {
            Decl::Function {
                name,
                return_type,
                args,
                ..
            } => {
                match self.scope() {
                    Scope::Local => {
                        unreachable!("function declarations are not allowed in a local scope")
                    }
                    Scope::Global => {
                        let args = args
                            .iter()
                            .map(|arg_ref| match self.ast.get_stmt(*arg_ref) {
                                Some(Stmt::FuncArg { decl_type, .. }) => *decl_type,
                                stmt => unreachable!(
                                    "Expected statement of kind `Stmt::FuncArg` got {:?}",
                                    stmt
                                ),
                            })
                            .collect();
                        // Bind the function definition.
                        let func_symbol = Symbol::FunctionDefinition {
                            name: name.clone(),
                            t: *return_type,
                            args,
                        };
                        self.table.bind(name, func_symbol);
                    }
                }
            }
            Decl::GlobalVar {
                decl_type, name, ..
            } => {
                // TODO: Ensure r-value type matches l-value declared type.
                let symbol = match self.scope() {
                    Scope::Global => Symbol::GlobalVariable {
                        name: name.clone(),
                        t: *decl_type,
                    },
                    _ => unreachable!("unexpected state: trying to define global variable in local scope {:?} @ {:?}", decl, self.scope())
                };
                self.table.bind(name, symbol)
            }
        }
    }

    /// Run semantic analysis pass on the given AST.
    pub fn analyze(&mut self) -> &SymbolTable {
        ast::walk(self.ast, self);
        self.symbol_table()
    }
}

/// `ast::Visitor` implementation for `DeclAnalyzer` processes only declarations
/// by effectively creating bindings and does nothing for the rest of AST nodes.
impl<'a> ast::Visitor<()> for DeclAnalyzer<'a> {
    fn visit_decl(&mut self, decl: &ast::Decl) {
        match decl {
            func_decl @ Decl::Function { args, body, .. } => {
                // Process the function declaration.
                self.define_global_binding(func_decl);
                // Enter the local scope and process the arguments and body.
                self.table.enter();
                for arg_ref in args {
                    if let Some(arg) = self.ast.get_stmt(*arg_ref) {
                        self.define_local_binding(arg);
                    }
                }
                // We explicitly want to fetch and unwrap the AST node instead
                // of calling `visit_stmt` directly. Calling `visit_stmt` will
                // naively enter a new scope again even if the block statements
                // are still scoped to the scope we entered when binding args.
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
            global_var @ Decl::GlobalVar { .. } => self.define_global_binding(global_var),
        }
    }
    fn visit_expr(&mut self, _: &Expr) {}

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            decl @ (Stmt::LocalVar { .. } | Stmt::FuncArg { .. }) => {
                self.define_local_binding(decl)
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
            Stmt::If {
                then_block: then_block_ref,
                else_block: Some(else_block_ref),
                ..
            } => {
                if let Some(stmt) = self.ast.get_stmt(*then_block_ref) {
                    self.visit_stmt(stmt)
                }
                if let Some(stmt) = self.ast.get_stmt(*else_block_ref) {
                    self.visit_stmt(stmt)
                }
            }
            Stmt::For { body: body_ref, .. } => {
                if let Some(stmt) = self.ast.get_stmt(*body_ref) {
                    self.visit_stmt(stmt)
                }
            }
            Stmt::While {
                body: Some(stmt_ref),
                ..
            } => {
                if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                    self.visit_stmt(stmt)
                }
            }
            _ => (),
        }
    }
}

/// Semantics analyzer walks the AST and uses the pre-built synmbol table
/// to validate the semantic correctness of the input program.
///
/// It tries to enforce the same semantic correctness that is expected of C0
/// programs, the following is a list of rules it tries to enforce :
///
/// - Identifiers resolve to valid symbols in the AST.
/// - Identifiers can't be re-used within the same block (impacts declarations)
/// - Functions can only be declared in the global scope.
/// - Function calls have the correct number and type of their arguments.
/// - Functions must end with a `return` statement unless they have type `void`
/// - L-values assignments are valid targets of R-values and of the same type.
struct SemanticAnalyzer<'a> {
    ast: &'a ast::AST,
    symbol_table: &'a SymbolTable,
    current_scope: usize,
}

impl<'a> SemanticAnalyzer<'a> {
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
        self.symbol_table.find(name, self.current_scope)
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

    /// Resolve an expression's type.
    ///
    /// # Panicw
    ///
    /// `resolve` will implicitly typecheck expressions for type soundness
    /// for example applying `!` to a non-boolean expression will cause
    /// a panic.
    pub fn resolve(&self, expr: &ast::Expr) -> DeclType {
        match expr {
            ast::Expr::Named(name) => match self.lookup(name) {
                Some(sym) => sym.t(),
                None => panic!("Identifer {name} was not found, ensure it is declared before use."),
            },
            ast::Expr::Call {
                name: name_ref,
                args,
            } => {
                let name = match self.ast.get_expr(*name_ref) {
                    Some(ast::Expr::Named(name)) => name,
                    _ => unreachable!("Expected call expression to reference `NamedExpr`"),
                };
                match self.lookup(name) {
                    Some(Symbol::FunctionDefinition {
                        name,
                        args: func_args,
                        t,
                    }) => {
                        // Ensure the `Call` expression uses the correct
                        // number of arguments.
                        assert_eq!(
                            args.len(),
                            func_args.len(),
                            "Expected function `{name}` to have {} arguments got {} in call.",
                            func_args.len(),
                            args.len()
                        );
                        // Iterate of the function definition arguments and
                        // the call expression references and ensure they
                        // are of the same type.
                        for (arg_ref, symbol_t) in args.iter().zip(func_args) {
                            if let Some(expr) = self.ast.get_expr(*arg_ref) {
                                let expr_t = self.resolve(expr);
                                assert_eq!(*symbol_t, expr_t, "Expected function `{name}` call argument to be of the same type")
                            } else {
                                unreachable!("Expression at ref {} was not found", arg_ref.get())
                            }
                        }
                        *t
                    }
                    _ => unreachable!("Call expression is only allowed for functions."),
                }
            }
            ast::Expr::Grouping(group_ref) => {
                if let Some(group) = self.ast.get_expr(*group_ref) {
                    self.resolve(group)
                } else {
                    unreachable!("Expression at ref {} was not found.", group_ref.get())
                }
            }
            ast::Expr::Assignment { name, value } => {
                // The type of an assignment expression is ambigious since
                // the left hand side might resolve to a type different
                // from the right hand side, so we must ensure the compatibility
                // of the assignment by resolving the right handside and compare
                // it to the left hand side.
                //
                // `lvalue` must be assignable.
                let lvalue = match self.ast.get_expr(*name) {
                    // Must be a named expression and the named identifier must
                    // resolve to a valid symbol.
                    Some(ast::Expr::Named(identifier)) => {
                        if let Some(symbol) = self.lookup(identifier) {
                            match symbol {
                                Symbol::FunctionDefinition { .. } => {
                                    unreachable!("Can't assign to function.")
                                }
                                _ => symbol.t(),
                            }
                        } else {
                            unreachable!("No symbol with name {identifier} was found.")
                        }
                    }
                    _ => unreachable!("Expression at ref {} was not found", name.get()),
                };
                let rvalue = match self.ast.get_expr(*value) {
                    Some(expr) => {
                        // Ensure r-value is assignable.
                        match expr {
                            ast::Expr::Assignment { .. } => {
                                unreachable!("R-value can't be assignment expression.")
                            }
                            _ => self.resolve(expr),
                        }
                    }
                    None => unreachable!("Expression at ref {} was not found.", value.get()),
                };
                assert_eq!(
                    lvalue, rvalue,
                    "Expected assignment lvalue type to match rvalue type."
                );
                lvalue
            }
            ast::Expr::BinOp {
                left,
                operator,
                right,
            } => {
                let lhs = if let Some(lhs) = self.ast.get_expr(*left) {
                    self.resolve(lhs)
                } else {
                    unreachable!("LHS expression at ref {} was not found.", left.get())
                };
                let rhs = if let Some(rhs) = self.ast.get_expr(*right) {
                    self.resolve(rhs)
                } else {
                    unreachable!("RHS expression at ref {} was not found.", right.get())
                };
                match operator {
                    &ast::BinaryOperator::Add
                    | &ast::BinaryOperator::Div
                    | &ast::BinaryOperator::Mul
                    | &ast::BinaryOperator::Sub => {
                        assert!(
                            lhs == DeclType::Int,
                            "expected left handside to {operator} to be of type `int` got {}",
                            lhs
                        );
                        assert!(
                            rhs == DeclType::Int,
                            "expected right handside to {operator} to be of type `int` got {}",
                            rhs
                        );
                        DeclType::Int
                    }
                    &ast::BinaryOperator::Eq
                    | &ast::BinaryOperator::Neq
                    | &ast::BinaryOperator::Gt
                    | &ast::BinaryOperator::Gte
                    | &ast::BinaryOperator::Lt
                    | &ast::BinaryOperator::Lte => {
                        assert_eq!(lhs, rhs, "expected left handisde and right handside of {operator} to be of the same type");
                        DeclType::Bool
                    }
                }
            }
            ast::Expr::UnaryOp { operator, operand } => match operator {
                ast::UnaryOperator::Neg => {
                    if let Some(expr) = self.ast.get_expr(*operand) {
                        match self.resolve(expr) {
                            DeclType::Int => DeclType::Int,
                            t => {
                                unreachable!("Unexpected `-` operator on expression of type {t}")
                            }
                        }
                    } else {
                        unreachable!("Expected unary operator `-` to have a valid operand.")
                    }
                }
                ast::UnaryOperator::Not => {
                    if let Some(expr) = self.ast.get_expr(*operand) {
                        match self.resolve(expr) {
                            DeclType::Bool => DeclType::Bool,
                            t => {
                                unreachable!("Unexpected `!` operator on expression of type {t}")
                            }
                        }
                    } else {
                        unreachable!("Expected unary operator `!` to have a valid operand.")
                    }
                }
            },
            ast::Expr::BoolLiteral(_) => DeclType::Bool,
            ast::Expr::IntLiteral(_) => DeclType::Int,
            ast::Expr::CharLiteral(_) => DeclType::Char,
        }
    }
}

impl<'a> ast::Visitor<()> for SemanticAnalyzer<'a> {
    fn visit_expr(&mut self, _: &Expr) {
        unreachable!("Semantic analysis pass is handled by internal `resolve` function.")
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            ast::Stmt::LocalVar {
                decl_type,
                name,
                value,
            } => {
                if let Some(assignee) = self.ast.get_expr(*value) {
                    let rvalue_t = self.resolve(assignee);
                    assert_eq!(decl_type, &rvalue_t, "Invalid assignment of expression with type {rvalue_t} to l-value {name} with type {decl_type}")
                } else {
                    unreachable!("Expression at ref {} was not found.", value.get())
                }
            }
            ast::Stmt::Expr(expr_ref) => {
                if let Some(assignee) = self.ast.get_expr(*expr_ref) {
                    let _ = self.resolve(assignee);
                } else {
                    unreachable!("Expression at ref {} was not found.", expr_ref.get())
                }
            }
            ast::Stmt::Return(expr_ref) => {
                if let Some(assignee) = self.ast.get_expr(*expr_ref) {
                    let _ = self.resolve(assignee);
                } else {
                    unreachable!("Expression at ref {} was not found.", expr_ref.get())
                }
            }
            ast::Stmt::Block(stmts) => {
                self.enter_scope();
                for stmt_ref in stmts.iter() {
                    if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                        self.visit_stmt(stmt)
                    } else {
                        unreachable!("Expression at ref {} was not found.", stmt_ref.get())
                    }
                }
                self.exit_scope();
            }
            ast::Stmt::For {
                init,
                condition,
                iteration,
                body,
            } => {
                // Validate initialization expression of the `for` loop is
                // either a named or assignment expression.
                if let Some(init_ref) = init {
                    if let Some(init) = self.ast.get_expr(*init_ref) {
                        let _ = self.resolve(init);
                    } else {
                        unreachable!("Expression at ref {} was not found.", init_ref.get())
                    }
                    match self.ast.get_expr(*init_ref) {
                        Some(ast::Expr::Named(_)) | Some(ast::Expr::Assignment { .. }) => (),
                        expr  => unreachable!("Expected `for` loop initialization to be named or assignment expression got {:?}", expr)
                    }
                }
                // Validate condition expression of the `for` loop resolves
                // to `bool`  type.
                if let Some(condition_ref) = condition {
                    if let Some(condition) = self.ast.get_expr(*condition_ref) {
                        let t = self.resolve(condition);
                        assert_eq!(
                            t,
                            DeclType::Bool,
                            "Expected `for` loop condition to be a boolean expression got {:?}",
                            condition
                        )
                    } else {
                        unreachable!("Expression at ref {} was not found.", condition_ref.get())
                    }
                }
                // Validate iteration expression of the `for` loop is an
                // assignment expression.
                if let Some(iteration_ref) = iteration {
                    match self.ast.get_expr(*iteration_ref) {
                        Some(ast::Expr::Assignment { .. }) => println!("Got assignment"),
                        expr => unreachable!(
                            "Expected `for` loop iteration to be assignment expression got {:?}",
                            expr
                        ),
                    }
                }
                // Recurisvely validate the statements in the block.
                if let Some(block) = self.ast.get_stmt(*body) {
                    self.visit_stmt(block)
                } else {
                    unreachable!("Expected `for` loop body to be `Block` statement.")
                }
            }
            ast::Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                // Validate `condition` resolves to a boolean expression.
                if let Some(condition) = self.ast.get_expr(*condition) {
                    let t = self.resolve(condition);
                    assert_eq!(
                        t,
                        DeclType::Bool,
                        "invalid expression type in `if` statement, must be of type bool found {t}"
                    )
                } else {
                    unreachable!("Expression at ref {} was not found.", condition.get())
                }
                if let Some(block) = self.ast.get_stmt(*then_block) {
                    self.visit_stmt(block)
                }
                if let Some(else_block) = else_block {
                    if let Some(block) = self.ast.get_stmt(*else_block) {
                        self.visit_stmt(block)
                    }
                }
            }
            _ => todo!("Unimplemented visitor for stmt of kind {:?}", stmt),
        }
    }

    fn visit_decl(&mut self, decl: &Decl) {
        match decl {
            ast::Decl::GlobalVar {
                decl_type, value, ..
            } => {
                if let Some(assignee) = self.ast.get_expr(*value) {
                    // Resolve the r-value type.
                    let rvalue_t = self.resolve(assignee);
                    assert_eq!(decl_type, &rvalue_t)
                } else {
                    unreachable!("Expression at ref {} was not found.", value.get())
                }
            }
            ast::Decl::Function {
                return_type, body, ..
            } => {
                let mut has_return_stmt = false;
                self.enter_scope();
                match self.ast.get_stmt(*body) {
                    Some(Stmt::Block(stmts)) => {
                        for stmt_ref in stmts {
                            if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                                self.visit_stmt(stmt);
                            }
                            match self.ast.get_stmt(*stmt_ref) {
                                Some(ast::Stmt::Return(ret_expr)) => {
                                    has_return_stmt = true;
                                    if let Some(expr) = self.ast.get_expr(*ret_expr) {
                                        let ret_expr_t = self.resolve(expr);
                                        assert_eq!(ret_expr_t, *return_type, "Invalid return statement with expression of type {ret_expr_t}, function is of type {return_type}")
                                    }
                                }
                                Some(stmt) => self.visit_stmt(stmt),
                                None => unreachable!(
                                    "Statement at ref {} was not found.",
                                    stmt_ref.get()
                                ),
                            }
                        }
                    }
                    _ => unreachable!("Expected function body to be a `Block` statement."),
                }
                if !has_return_stmt {
                    // panic!("Function {name} with type {return_type} has no `return` statement.")
                }
                self.exit_scope();
            }
        }
    }
}

/// Analyze the input AST and return the symbol table.
pub fn analyze(ast: &ast::AST) -> SymbolTable {
    let mut decl_analyzer = DeclAnalyzer::new(ast);
    ast::walk(ast, &mut decl_analyzer);
    let symbol_table = decl_analyzer.symbol_table();
    let mut semantic_analyzer = SemanticAnalyzer::new(ast, symbol_table);
    ast::walk(ast, &mut semantic_analyzer);
    symbol_table.clone()
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use crate::parser::Parser;
    use crate::scanner::Scanner;
    use crate::sema::{DeclAnalyzer, SemanticAnalyzer};

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
                let symbol_table = decl_analyzer.analyze();

                for (ii, table) in symbol_table.tables.iter().enumerate() {
                    println!("Scope @ {}", ii);
                    for (name, symbol) in table {
                        println!("{} => {}", name, symbol);
                    }
                }
            }
        };
    }

    macro_rules! test_semantic_analyzer {
        ($name:ident, $source:expr ) => {
            #[test]
            #[should_panic] // Poor man's error handling but works.
            fn $name() {
                let source = $source;
                let mut scanner = Scanner::new(source);
                let tokens = scanner
                    .scan()
                    .expect("expected test case source to be valid");
                let mut parser = Parser::new(&tokens);
                parser.parse();

                let mut decl_analyzer = DeclAnalyzer::new(parser.ast());
                let symbol_table = decl_analyzer.analyze();
                let mut semantic_analyzer = SemanticAnalyzer::new(parser.ast(), symbol_table);
                println!("AST: {}", parser.ast());
                for (ii, table) in symbol_table.tables.iter().enumerate() {
                    println!("Scope @ {}", ii);
                    for (name, symbol) in table {
                        println!("{} => {}", name, symbol);
                    }
                }

                ast::walk(parser.ast(), &mut semantic_analyzer)
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

    test_decl_analyzer!(
        can_process_declarations_in_a_for_loop,
        "int main() { int i = 0; for(i=0;i<10;i = i + 1) {int b; int c; int d = i;} }"
    );

    test_semantic_analyzer!(
        can_find_duplicate_redefinition,
        "int main() {} int main() {}"
    );

    test_semantic_analyzer!(
        can_find_invalid_bool_literal_assignment,
        "int main() { int a = true; }"
    );
    test_semantic_analyzer!(
        can_find_invalid_call_assignment,
        "int f () { return -1; } int main() { char a = f(); }"
    );
    test_semantic_analyzer!(
        can_find_function_with_invalid_return_statement,
        "int f () { return true  } int main() { char a = f();  }"
    );
    test_semantic_analyzer!(
        can_find_invalid_condition_in_if_statement,
        "int f() { if (1) { return 0;} }"
    );
    test_semantic_analyzer!(
        can_find_invalid_call_expressions,
        "int f(int a, int b) { return a + b;} int main() { int x = f(true, false); }"
    );
    test_semantic_analyzer!(
        can_find_invalid_call_expressions_in_nested_scope,
        "int f(int a, int b) { return a + b; } int main() { { f(false, true); } }"
    );
    test_semantic_analyzer!(
        can_find_invalid_for_statement_with_non_assignment_as_iteration,
        "int main() { int i; for(i = 0;;i < 1) {}}"
    );
    test_semantic_analyzer!(
        can_find_invalid_for_statement_with_non_boolean_condition,
        "int main() { int i; for(i = 0; i = i + 1; i = i + 1) {}}"
    );
}
