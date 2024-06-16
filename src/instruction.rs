//! Glouton IR instructions.
use std::{fmt, marker::PhantomData};

use crate::{ast, ir::Argument, sema};

/// Types used in the IR.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    // Empty type.
    #[default]
    Unit,
    // Integers, defaults to i32.
    Int,
    // Booleans.
    Bool,
    // Characters.
    Char,
}

impl Type {
    /// Returns an IR type from an AST declaration type.
    fn from(value: &ast::DeclType) -> Self {
        match value {
            ast::DeclType::Int => Self::Int,
            ast::DeclType::Char => Self::Char,
            ast::DeclType::Bool => Self::Bool,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unit => write!(f, ""),
            Self::Int => write!(f, "int"),
            Self::Bool => write!(f, "bool"),
            Self::Char => write!(f, "char"),
        }
    }
}

/// Literal values.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Literal {
    /// Empty value.
    #[default]
    Empty,
    /// Integers
    Int(i32),
    /// Booleans.
    Bool(bool),
    /// Characters.
    Char(char),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "NONE"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::Char(value) => write!(f, "{value}"),
        }
    }
}
/// Symbol references are used as an alternative to variable names.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolRef(usize /* Reference */, Type);

/// Symbols can represent variable or function names.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Symbol(String, Type);

impl Symbol {
    pub fn new(name: &str, t: Type) -> Self {
        Self(name.to_string(), t)
    }
}

/// Labels are used to designate branch targets in control flow operations.
///
/// Each label is a relative offset to the target branch first instruction.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(usize);

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "__LABEL_{}", self.0)
    }
}

/// Every value in the intermediate representation is either a symbol reference
/// to a storage location or a literal value.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    StorageLocation(Symbol),
    ConstantLiteral(Literal),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageLocation(symbol) => write!(f, "{}", symbol.0),
            Self::ConstantLiteral(lit) => write!(f, "{lit}"),
        }
    }
}

/// OPCode is a type wrapper around all opcodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OPCode {
    // Indirect jumps.
    Jump,
    // Condtional branches.
    Branch,
    // Function calls that don't produce values.
    Call,
    // Return statements.
    Return,
    /// `const` operation.
    Const,
    // Arithmetic operators.
    Add,
    Sub,
    Mul,
    Div,
    // Comparison operators.
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    // Unary operators.
    Not,
    Neg,
    // Boolean operators.
    And,
    Or,
    // Identity operator.
    Id,
    // Label pseudo instruction.
    Label,
    // Nop instruction.
    Nop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Instruction {
    // `const` operation.
    Const(
        Symbol,  /* Destination */
        Literal, /* Literal value assigned to the destination */
    ),
    // Arithmetic operators.
    Add(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Sub(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Mul(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Div(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    // Binary boolean operators.
    And(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Or(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    // Unary operators.
    Not(Symbol /* Destination */, Value /* LHS */),
    Neg(Symbol /* Destination */, Value /* LHS */),
    // Comparison operators.
    Eq(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Neq(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Lt(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Lte(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Gt(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    Gte(
        Symbol, /* Destination */
        Value,  /* LHS */
        Value,  /* RHS */
    ),
    // Return statements.
    Return(Value /* Return value */),
    // Function calls.
    Call(
        Symbol,     /* Call Target */
        Vec<Value>, /* Arguments */
    ),
    // Direct jump to label.
    Jump(Label /* Offset */),
    // Condtional branches.
    Branch(
        Value, /* Condition */
        Label, /* Then Target Offset */
        Label, /* Else Target Offset */
    ),
    // Identity operator.
    Id(Symbol /* Destination symbol*/, Value),
    // Label pseudo instruction, acts as a data marker when generating code.
    Label(usize /* Label handle or offset */),
    // Nop instruction.
    Nop,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Const(dst, lit) => {
                write!(f, "{}: {} const = {lit}", dst.0, dst.1)
            }
            Instruction::Add(dst, lhs, rhs) => {
                write!(f, "{} : {} = add {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Sub(dst, lhs, rhs) => {
                write!(f, "{} : {} = sub {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Mul(dst, lhs, rhs) => {
                write!(f, "{} : {} = mul {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Div(dst, lhs, rhs) => {
                write!(f, "{} : {} = div {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::And(dst, lhs, rhs) => {
                write!(f, "{} : {} = and {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Or(dst, lhs, rhs) => {
                write!(f, "{} : {} = or {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Neg(dst, operand) => {
                write!(f, "{} : {} = neg {operand}", dst.0, dst.1)
            }
            Instruction::Not(dst, operand) => {
                write!(f, "{} : {} = not {operand}", dst.0, dst.1)
            }
            Instruction::Eq(dst, lhs, rhs) => {
                write!(f, "{} : {} = or {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Neq(dst, lhs, rhs) => {
                write!(f, "{} : {} = or {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Lt(dst, lhs, rhs) => {
                write!(f, "{} : {} = or {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Lte(dst, lhs, rhs) => {
                write!(f, "{} : {} = or {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Gt(dst, lhs, rhs) => {
                write!(f, "{} : {} = or {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Gte(dst, lhs, rhs) => {
                write!(f, "{} : {} = or {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Return(value) => write!(f, "return {value}"),
            Instruction::Call(def, args) => {
                write!(f, "@{}", def.0)?;
                for arg in args {
                    write!(f, "{arg} ")?;
                }
                write!(f, "")
            }
            Instruction::Jump(target) => write!(f, "jump {}", target),
            Instruction::Branch(cond, then_target, else_target) => {
                write!(f, "br {} {then_target} {else_target}", cond)
            }
            Instruction::Id(dst, value) => {
                write!(f, "{} : {} = id {value}", dst.0, dst.1)
            }
            Instruction::Nop => write!(f, "nop"),
            Instruction::Label(addr) => write!(f, "__LABEL_{addr}"),
        }
    }
}

impl Instruction {
    pub fn opcode(&self) -> OPCode {
        match self {
            Instruction::Const(..) => OPCode::Const,
            Instruction::Add(..) => OPCode::Add,
            Instruction::Sub(..) => OPCode::Sub,
            Instruction::Mul(..) => OPCode::Mul,
            Instruction::Div(..) => OPCode::Div,
            Instruction::And(..) => OPCode::And,
            Instruction::Or(..) => OPCode::Or,
            Instruction::Neg(..) => OPCode::Neg,
            Instruction::Not(..) => OPCode::Not,
            Instruction::Eq(..) => OPCode::Eq,
            Instruction::Neq(..) => OPCode::Neq,
            Instruction::Lt(..) => OPCode::Lt,
            Instruction::Lte(..) => OPCode::Lte,
            Instruction::Gt(..) => OPCode::Gt,
            Instruction::Gte(..) => OPCode::Gte,
            Instruction::Return(..) => OPCode::Return,
            Instruction::Call(..) => OPCode::Call,
            Instruction::Jump(..) => OPCode::Jump,
            Instruction::Branch(..) => OPCode::Branch,
            Instruction::Id(..) => OPCode::Id,
            Instruction::Nop => OPCode::Nop,
            Instruction::Label(..) => OPCode::Label,
        }
    }
}

/// Scope of the current AST node we are processing, this is an internal detail
/// of the `IRBuilder` and is used to decide where the current declaration will
/// live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Scope {
    Global,
    Local,
}

/// `Function` represents a function declaration in the AST, a `Function`
/// is composed as a linear sequence of GIR instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    // Function name.
    name: String,
    // List of arguments the function accepts.
    args: Vec<Symbol>,
    // Body of the function as GIR instructions.
    body: Vec<Instruction>,
    // Return type of the function if any.
    return_type: Type,
}

impl Function {
    fn new(name: &str, args: Vec<Symbol>, return_type: Type) -> Self {
        Self {
            name: name.to_string(),
            args: args,
            body: vec![],
            return_type: return_type,
        }
    }

    fn push(&mut self, inst: &Instruction) {
        self.body.push(inst.clone())
    }
}

/// `IRBuilderTrackingRef` is a tuple of position within the function
/// being currently lowered, a scope enum value to deal with nesting
/// and a pointer to the symbol table level we start symbol resolution
/// from.
struct IRBuilderTrackingRef(usize, usize, Scope);

impl IRBuilderTrackingRef {
    // Return the index of the current function we are generating instructions
    // for.
    fn pos(&self) -> usize {
        self.0
    }

    // Return the current scope.
    fn scope(&self) -> Scope {
        self.2
    }

    // Enter switches to a local scope view and increments the symbol table
    // level.
    fn enter(&mut self) {
        self.1 += 1;
        self.2 = Scope::Local;
    }

    // Exit back to the global scope, since the IR does not have nested scopes
    // calling `exit` always restores the scope back to `Scope::Global`.
    fn exit(&mut self) {
        // Exit signals that we completed building a single functional unit
        // so our position in the global program IR is increased by 1 since
        // the next unit will be a new function in the program.
        self.0 += 1;
        // Decide if we are exiting back to a local scope or the top level
        // global scope.
        //
        // If `self.1` is equal to `1` then we are exiting back to the global
        // scope, otherwise we are still in a local scope.
        if self.1 == 1 {
            self.2 = Scope::Global;
        } else {
            self.2 = Scope::Local;
        }
        self.1 -= 1;
    }
}

/// `LocationLabelCounter` is a tuple of variable and label counters used
/// to generate monotonically increasing indices for temporaries and labels.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct LocationLabelCounter(usize, usize);

impl LocationLabelCounter {
    fn new() -> Self {
        Self(0, 0)
    }

    fn next_location(&mut self) -> usize {
        let next = self.0;
        self.0 += 1;
        next
    }

    fn next_label(&mut self) -> usize {
        let next = self.1;
        self.1 += 1;
        next
    }
}

/// `GlobalValue` is a tuple of variable name and a compile time literal.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct GlobalValue(Symbol, Literal);

/// `IRBuilder` is responsible for lowering the AST to the intermediate
/// representation, the first lowering phase results in a program represented
/// as a tuple of global values and functions. This first representation is
/// linear (functions are just `Vec<Instruction>`) and is used as a starting
/// point for building a graph representation.
pub struct IRBuilder<'a> {
    // Reference to the AST we are processing.
    ast: &'a ast::AST,
    // Symbol table built during semantic analysis phase.
    symbol_table: &'a sema::SymbolTable,
    // Program as a sequence of declared functions.
    program: Vec<Function>,
    // Global scope declarations.
    globals: Vec<GlobalValue>,
    // Counter used to keep track of temporaries, temporaries are storage
    // assignemnts for transient or non assigned values such as a literals.
    llc: LocationLabelCounter,
    // TrackingRef for the `IRBuilder` acts as a composite pointer to keep track
    // of metadata that's useful during the lowering phase.
    tracker: IRBuilderTrackingRef,
}

impl<'a> IRBuilder<'a> {
    /// Push a slice of instructions to the current function's body.
    fn push(&mut self, instrs: &[Instruction]) {
        match self.tracker.scope() {
            Scope::Local => {
                for inst in instrs {
                    self.program[self.tracker.pos()].push(inst)
                }
            }
            Scope::Global => {
                // The global IR scope can only contain constant or id
                // instructions.
                for inst in instrs {
                    assert!(matches!(
                        inst.opcode(),
                        OPCode::Const | OPCode::Id
                    ));
                    self.program[0].push(inst);
                }
            }
        }
        for inst in instrs {}
    }
}

impl<'a> ast::Visitor<(Option<Value>, Vec<Instruction>)> for IRBuilder<'a> {
    fn visit_decl(
        &mut self,
        decl: &ast::Decl,
    ) -> (Option<Value>, Vec<Instruction>) {
        match decl {
            // Function declarations are the only place where we need to switch
            // scopes explicitely, since the scope only affects where variable
            // declarations are positioned.
            ast::Decl::Function {
                name,
                return_type,
                args,
                body,
            } => {
                // Build and push a new function frame.
                let func_args = args
                    .iter()
                    .map(|arg| match self.ast.get_stmt(*arg) {
                        Some(ast::Stmt::FuncArg { decl_type, name }) => {
                            Symbol::new(name, Type::from(decl_type))
                        }
                        _ => unreachable!(
                        "expected argument reference to be valid and to be `ast::Stmt::FuncArg`"
                    ),
                    })
                    .collect::<Vec<_>>();
                let func_return_type = Type::from(return_type);
                let func = Function::new(name, func_args, func_return_type);
                // Enter a new scope and push the new function frame.
                self.program.push(func);
                self.tracker.enter();
                // Generate IR for the body.
                let (span, code) = match self.ast.get_stmt(*body) {
                    Some(stmt) => self.visit_stmt(stmt),
                    _ => unreachable!("expected body reference to be valid !"),
                };
                self.push(&code);
                // Exit back to the global scope.
                self.tracker.exit();
                (span, code)
            }
            ast::Decl::GlobalVar {
                decl_type,
                name,
                value,
            } => {
                let dst = Symbol::new(name, Type::from(decl_type));
                let (arg, mut code) =
                    if let Some(expr) = self.ast.get_expr(*value) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };
                // Get the destination of the right hand side.
                code.push(Instruction::Id(
                    dst.clone(),
                    arg.expect(
                        "Expected right handside to be a valid temporary",
                    ),
                ));
                (Some(Value::StorageLocation(dst)), code)
            }
        }
    }

    fn visit_stmt(
        &mut self,
        stmt: &ast::Stmt,
    ) -> (Option<Value>, Vec<Instruction>) {
        match stmt {
            // Variable declaration are
            ast::Stmt::LocalVar {
                decl_type,
                name,
                value,
            } => {
                let dst = Symbol::new(name, Type::from(decl_type));
                let (arg, mut code) =
                    if let Some(expr) = self.ast.get_expr(*value) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };
                // Get the destination of the right hand side.
                // Get the destination of the right hand side.
                code.push(Instruction::Id(
                    dst.clone(),
                    arg.expect(
                        "Expected right handside to be a valid temporary",
                    ),
                ));
                (Some(Value::StorageLocation(dst)), code)
            }
            // Blocks.
            ast::Stmt::Block(stmts) => {
                // self.level += 1;
                self.tracker.1 += 1;
                let mut code = vec![];
                for stmt_ref in stmts {
                    let (_, mut block) =
                        if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                            self.visit_stmt(stmt)
                        } else {
                            unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                        };

                    code.append(&mut block);
                }
                // self.level -= 1;
                self.tracker.1 -= 1;
                (None, code)
            }
            // Return statements.
            ast::Stmt::Return(expr_ref) => {
                let (value, mut code) =
                    if let Some(expr) = self.ast.get_expr(*expr_ref) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };
                let ret = Instruction::Return(value.expect(
                    "Expected right handside to be temporary or named",
                ));

                code.push(ret);
                (value, code)
            }
            // Expression statements.
            ast::Stmt::Expr(expr_ref) => {
                let (name, code) =
                    if let Some(expr) = self.ast.get_expr(*expr_ref) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };
                (name, code)
            }
            // Conditional blocks.
            ast::Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                let (condition, mut code) =
                    if let Some(cond) = self.ast.get_expr(*condition) {
                        self.visit_expr(cond)
                    } else {
                        unreachable!(
                        "Expected condition to reference a valid expression"
                    )
                    };
                let then_label = self.llc.next_label();
                let else_label = self.llc.next_label();
                let end_label = self.llc.next_label();

                // Push branch instruction.
                let inst = Instruction::Branch(
                    condition.expect("Expected condition variable to be valid"),
                    Label(then_label),
                    Label(else_label),
                );
                code.push(inst);
                // Push then label.
                let inst = Instruction::Label(then_label);
                code.push(inst);
                // Generate instruction for the then block.
                let (_, mut block) =
                    if let Some(block) = self.ast.get_stmt(*then_block) {
                        self.visit_stmt(block)
                    } else {
                        unreachable!(
                        "Expected reference to block to be a valid statement"
                    )
                    };
                code.append(&mut block);
                // Push a jump instruction to the end label iif the last
                // instruction was not a return..
                if code.last().is_some_and(|inst| match inst.opcode() {
                    OPCode::Return => false,
                    _ => true,
                }) {
                    let inst = Instruction::Jump(Label(end_label));
                    code.push(inst);
                }
                // Push else label.
                let inst = Instruction::Label(else_label);
                code.push(inst);
                // Generate instruction for the else block if one exists.
                if else_block.is_some() {
                    let (_, mut block) = if let Some(block) = self.ast.get_stmt(
                        else_block.expect("Expected else block to be `Some`"),
                    ) {
                        self.visit_stmt(block)
                    } else {
                        unreachable!("Expected reference to block to be a valid statement")
                    };
                    code.append(&mut block);
                }
                // Push a jump instruction to the end label iif the last
                // instruction was not a return..
                if code.last().is_some_and(|inst| match inst.opcode() {
                    OPCode::Return => false,
                    _ => true,
                }) {
                    let inst = Instruction::Jump(Label(end_label));
                    code.push(inst);
                }
                // Push end label.
                let inst = Instruction::Label(end_label);
                code.push(inst);

                (None, code)
            }
            ast::Stmt::FuncArg {
                decl_type: _,
                name: _,
            } => {
                unreachable!(
                    "Expected function argument to be handled in `visit_decl`"
                )
            }
            ast::Stmt::For {
                init,
                condition,
                iteration,
                body,
            } => {
                // Generate labels for the loop.
                let loop_body_label = self.llc.next_label();
                let loop_exit_label = self.llc.next_label();
                let mut code = Vec::new();
                // Generate initializer block if it exists.
                if init.is_some() {
                    let (_, mut block) = if let Some(block) = self.ast.get_expr(
                        init.expect("Expected init block to be `Some`"),
                    ) {
                        self.visit_expr(block)
                    } else {
                        unreachable!("Expected reference to initializer to be a valid expression")
                    };
                    code.append(&mut block);
                }
                // Generate the loop body label.
                let inst = Instruction::Label(loop_body_label);
                code.push(inst);
                // Generate the loop body block.
                let (_, mut block) =
                    if let Some(block) = self.ast.get_stmt(*body) {
                        self.visit_stmt(block)
                    } else {
                        unreachable!(
                        "Expected reference to body to be a valid statement"
                    )
                    };
                code.append(&mut block);
                // Generate the iteration expression.
                if iteration.is_some() {
                    let (_, mut block) = if let Some(block) =
                        self.ast.get_expr(iteration.expect(
                            "Expected iteration expression to be `Some`",
                        )) {
                        self.visit_expr(block)
                    } else {
                        unreachable!("Expected reference to iteration to be a valid expression")
                    };
                    code.append(&mut block);
                }
                // Generate the condition block if it exists.
                if condition.is_some() {
                    let (condition, mut block) = if let Some(block) =
                        self.ast.get_expr(
                            condition.expect(
                                "Expected condition block to be `Some`",
                            ),
                        ) {
                        self.visit_expr(block)
                    } else {
                        unreachable!("Expected reference to initializer to be a valid expression")
                    };
                    code.append(&mut block);
                    // Generate branch to exit code.
                    let inst = Instruction::Branch(
                        condition
                            .expect("Expected condition variable to be valid"),
                        Label(loop_body_label),
                        Label(loop_exit_label),
                    );
                    code.push(inst);
                }
                // Generate the loop exit code.
                let inst = Instruction::Label(loop_exit_label);
                code.push(inst);
                (None, code)
            }
            ast::Stmt::While { condition, body } => {
                // Generate labels for the loop.
                let loop_body_label = self.llc.next_label();
                let loop_exit_label = self.llc.next_label();
                let mut code = Vec::new();
                if body.is_some() {
                    // Generate the loop body label.
                    let inst = Instruction::Label(loop_body_label);
                    code.push(inst);
                    // Generate the loop body block.
                    let (_, mut block) = if let Some(block) = self.ast.get_stmt(
                        body.expect("Expected loop body to be `Some`"),
                    ) {
                        self.visit_stmt(block)
                    } else {
                        unreachable!("Expected reference to body to be a valid statement")
                    };
                    code.append(&mut block);
                }
                if condition.is_some() {
                    let (condition, mut block) = if let Some(block) =
                        self.ast.get_expr(
                            condition.expect(
                                "Expected condition block to be `Some`",
                            ),
                        ) {
                        self.visit_expr(block)
                    } else {
                        unreachable!("Expected reference to initializer to be a valid expression")
                    };
                    code.append(&mut block);
                    // Generate branch to exit code.
                    let inst = Instruction::Branch(
                        condition
                            .expect("Expected condition variable to be valid"),
                        Label(loop_body_label),
                        Label(loop_exit_label),
                    );
                    code.push(inst);
                }
                // Generate the loop exit code.
                let inst = Instruction::Label(loop_exit_label);
                code.push(inst);
                (None, code)
            }
            ast::Stmt::Empty => (None, vec![]),
        }
    }

    fn visit_expr(
        &mut self,
        expr: &ast::Expr,
    ) -> (Option<Value>, Vec<Instruction>) {
        match *expr {
            ast::Expr::IntLiteral(value) => {
                let mut code = vec![];
                let dst = Symbol::new(
                    format!("%v{}", self.llc.next_location()).as_str(),
                    Type::Int,
                );
                code.push(Instruction::Const(dst, Literal::Int(value)));
                (Some(Value::StorageLocation(dst)), code)
            }
            ast::Expr::BoolLiteral(value) => {
                let mut code = vec![];
                let dst = Symbol::new(
                    format!("%v{}", self.llc.next_location()).as_str(),
                    Type::Int,
                );
                code.push(Instruction::Const(dst, Literal::Bool(value)));
                (Some(Value::StorageLocation(dst)), code)
                /*
                let mut code = vec![];
                let dst = self.next_var();
                code.push(Instruction::constant(
                    dst.clone(),
                    Type::Bool,
                    Literal::Bool(value),
                ));
                (Some(dst), code)
                 */
            }
            ast::Expr::CharLiteral(value) => {
                let mut code = vec![];
                let dst = Symbol::new(
                    format!("%v{}", self.llc.next_location()).as_str(),
                    Type::Int,
                );
                code.push(Instruction::Const(dst, Literal::Char(value)));
                (Some(Value::StorageLocation(dst)), code)
                /*
                let mut code = vec![];
                let dst = self.next_var();
                code.push(Instruction::constant(
                    dst.clone(),
                    Type::Char,
                    Literal::Char(value),
                ));
                (Some(dst), code)
            }
            ast::Expr::UnaryOp { operator, operand } => {
                // TODO: Visitor for IR will return a lhs assignee and an instruction
                // for the rhs assignment.
                // let src = self.visit_expr(self.ast.get_expr(operand).unwrap());
                let (name, mut code) =
                    if let Some(expr) = self.ast.get_expr(operand) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };

                let dst = self.next_var();
                let inst = match operator {
                    ast::UnaryOperator::Neg => Instruction::arith(
                        dst.clone(),
                        Type::Int,
                        ValueOp::Neg,
                        vec![name
                            .expect(
                                "Expected right handside to be in a temporary",
                            )
                            .clone()],
                    ),
                    ast::UnaryOperator::Not => Instruction::arith(
                        dst.clone(),
                        Type::Bool,
                        ValueOp::Not,
                        vec![name
                            .expect(
                                "Expected right handside to be in a temporary",
                            )
                            .clone()],
                    ),
                };
                code.push(inst);
                (Some(dst), code)
            }
            ast::Expr::BinOp {
                left,
                operator,
                right,
            } => {
                let mut code = vec![];
                let (lhs, mut code_left) =
                    if let Some(expr) = self.ast.get_expr(left) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };
                code.append(&mut code_left);

                let (rhs, mut code_right) =
                    if let Some(expr) = self.ast.get_expr(right) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };
                code.append(&mut code_right);

                let dst = self.next_var();
                let inst = match operator {
                    ast::BinaryOperator::Add
                    | ast::BinaryOperator::Sub
                    | ast::BinaryOperator::Mul
                    | ast::BinaryOperator::Div => Instruction::arith(
                        dst.clone(),
                        Type::Int,
                        ValueOp::from(&operator),
                        vec![
                            lhs.expect(
                                "Expected right handside to be in a temporary",
                            )
                            .clone(),
                            rhs.expect(
                                "Expected right handside to be in a temporary",
                            )
                            .clone(),
                        ],
                    ),
                    ast::BinaryOperator::Eq
                    | ast::BinaryOperator::Neq
                    | ast::BinaryOperator::Gt
                    | ast::BinaryOperator::Gte
                    | ast::BinaryOperator::Lt
                    | ast::BinaryOperator::Lte => Instruction::cmp(
                        dst.clone(),
                        Type::Bool,
                        ValueOp::from(&operator),
                        vec![
                            lhs.expect(
                                "Expected right handside to be in a temporary",
                            )
                            .clone(),
                            rhs.expect(
                                "Expected right handside to be in a temporary",
                            )
                            .clone(),
                        ],
                    ),
                };

                code.push(inst);
                (Some(dst), code)
            }
            ast::Expr::Named(ref name) => {
                let _t = match self.symbol_table.find(name, self.level) {
                    Some(symbol) => symbol.t(),
                    None => unreachable!(
                        "Expected a symbol for named expression : `{name}`"
                    ),
                };
                (Some(name.clone()), vec![])
            }
            ast::Expr::Grouping(expr_ref) => {
                let (name, code) =
                    if let Some(expr) = self.ast.get_expr(expr_ref) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };
                (name, code)
            }
            ast::Expr::Assignment { name, value } => {
                let mut code = vec![];
                let (rhs, mut code_right) =
                    if let Some(expr) = self.ast.get_expr(value) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };
                code.append(&mut code_right);
                // We know that by definition assignment left handside will
                // always be a named expression, unless we are dealing with
                // arrays or struct fields.
                let (lhs, _) = if let Some(named) = self.ast.get_expr(name) {
                    self.visit_expr(named)
                } else {
                    unreachable!(
                        "Expected right handside to be a valid expression"
                    )
                };
                let t = match self
                    .symbol_table
                    .find(&lhs.clone().unwrap(), self.level)
                {
                    Some(symbol) => symbol.t(),
                    None => unreachable!(
                        "Expected a symbol for named expression : `{}`",
                        lhs.unwrap()
                    ),
                };
                code.push(Instruction::id(
                    lhs.clone().unwrap(),
                    Type::from(&t),
                    vec![rhs.unwrap()],
                ));
                (lhs, code)
            }
            ast::Expr::Call { name, ref args } => {
                let name = match self.ast.get_expr(name) {
                    Some(ast::Expr::Named(name)) => name,
                    _ => unreachable!("Expected reference to be a named expression for a function"),
                };
                let t = match self.symbol_table.find(name, self.level) {
                    Some(symbol) => symbol.t(),
                    None => unreachable!(
                        "Expected a symbol for named expression : `{name}`"
                    ),
                };
                let (vars, code): (Vec<String>, Vec<Vec<Instruction>>) = args
                    .iter()
                    .map(|arg| {
                        if let Some(expr) = self.ast.get_expr(*arg) {
                            let (arg, code) = self.visit_expr(expr);
                            (arg.unwrap(), code)
                        } else {
                            unreachable!(
                                "Expected argument to be a valid expression"
                            )
                        }
                    })
                    .unzip();
                let mut code: Vec<Instruction> =
                    code.into_iter().flatten().collect();
                let dst = self.next_var();
                let inst = Instruction::call(
                    dst.clone(),
                    Type::from(&t),
                    name.to_string(),
                    vars,
                );
                code.push(inst);
                (Some(dst), code)
            }
        }
    }
}
