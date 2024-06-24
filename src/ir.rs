//! The glouton intermediate representation is a three-address code linear IR
//! used to represent Glouton programs.
//!
//! This iteration of the implementation improves significantly on the initial
//! design; with the actual instruction semantics remaining unchanged.
//!
//! `const` instruction exists to encode all literal values into a temporary
//! storage location (virtual registers) and the `id` instruction similarly
//! assigns named values to new temporaries.
//!
//! `const` and `id` are core to the way the IR is structured as they allow us
//! to easily translate into and out of SSA form; with potentially translating
//! out of SSA can forgo the rename phase and just prune all the phi nodes.
use std::fmt;

use crate::{ast, ast::Visitor, sema};

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

/// Symbols can represent variable or function names.
///
/// TODO: Potentially rename `Symbol` to storage location since it acts more
/// as a virtual register than a symbol; also let's keep the virtual register
/// number so that building things like use-defs becomes a set of `usize` vs
/// a set of `String`.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Symbol(String, Type);

impl Symbol {
    pub fn new(name: &str, t: Type) -> Self {
        Self(name.to_string(), t)
    }

    /// Returns a non-mutable reference to the symbol's name.
    pub fn name(&self) -> &str {
        &self.0
    }

    /// Returns the symbol's type.
    pub fn t(&self) -> Type {
        self.1
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.0, self.1)
    }
}

/// Labels are used to designate branch targets in control flow operations.
///
/// Each label is a relative offset to the target branch first instruction.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(usize);

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ".LABEL_{}", self.0)
    }
}

/// Every value in the intermediate representation is either a symbol reference
/// to a storage location or a literal value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
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

/// Instructions in the intermediate representation are in three-address form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    // `const` operation.
    Const(Symbol, Value),
    // Instructions for arithmetic operations are in three address form
    // and wrap the storage location, left and right handside operands.
    Add(Symbol, Value, Value),
    Sub(Symbol, Value, Value),
    Mul(Symbol, Value, Value),
    Div(Symbol, Value, Value),
    // Logical operations, similar to arithmetic operations but for boolean
    // values.
    And(Symbol, Value, Value),
    Or(Symbol, Value, Value),
    // Unary operators.
    Not(Symbol, Value),
    Neg(Symbol, Value),
    // Comparison operators.
    Eq(Symbol, Value, Value),
    Neq(Symbol, Value, Value),
    Lt(Symbol, Value, Value),
    Lte(Symbol, Value, Value),
    Gt(Symbol, Value, Value),
    Gte(Symbol, Value, Value),
    // Return statements.
    Return(Value),
    // Function calls.
    Call(
        // Storage location for the function call result.
        Symbol,
        // Called function name.
        Symbol,
        // Function arguments.
        Vec<Value>,
    ),
    // Direct jump to label.
    Jump(Label),
    // Condtional branches.
    Branch(
        // Conditional value.
        Value,
        // Label for the `then` branch.
        Label,
        // Label for the `else` branch.
        Label,
    ),
    // Identity operator.
    Id(Symbol, Value),
    // Label pseudo instruction, acts as a data marker when generating code.
    Label(usize),
    // Nop instruction.
    Nop,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Const(dst, lit) => {
                write!(f, "{}: {} = const {lit}", dst.0, dst.1)
            }
            Instruction::Add(dst, lhs, rhs) => {
                write!(f, "{}: {} = add {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Sub(dst, lhs, rhs) => {
                write!(f, "{}: {} = sub {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Mul(dst, lhs, rhs) => {
                write!(f, "{}: {} = mul {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Div(dst, lhs, rhs) => {
                write!(f, "{}: {} = div {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::And(dst, lhs, rhs) => {
                write!(f, "{}: {} = and {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Or(dst, lhs, rhs) => {
                write!(f, "{}: {} = or {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Neg(dst, operand) => {
                write!(f, "{}: {} = neg {operand}", dst.0, dst.1)
            }
            Instruction::Not(dst, operand) => {
                write!(f, "{}: {} = not {operand}", dst.0, dst.1)
            }
            Instruction::Eq(dst, lhs, rhs) => {
                write!(f, "{}: {} = eq {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Neq(dst, lhs, rhs) => {
                write!(f, "{}: {} = neq {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Lt(dst, lhs, rhs) => {
                write!(f, "{}: {} = lt {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Lte(dst, lhs, rhs) => {
                write!(f, "{}: {} = lte {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Gt(dst, lhs, rhs) => {
                write!(f, "{}: {} = gt {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Gte(dst, lhs, rhs) => {
                write!(f, "{}: {} = gte {lhs} {rhs}", dst.0, dst.1)
            }
            Instruction::Return(value) => write!(f, "ret {value}"),
            Instruction::Call(dst, def, args) => {
                write!(f, "{}: {} = call @{}", dst.0, dst.1, def.0)?;
                for arg in args {
                    write!(f, " {arg}")?;
                }
                write!(f, "")
            }
            Instruction::Jump(target) => write!(f, "jmp {}", target),
            Instruction::Branch(cond, then_target, else_target) => {
                write!(f, "br {} {then_target} {else_target}", cond)
            }
            Instruction::Id(dst, value) => {
                write!(f, "{}: {} = id {value}", dst.0, dst.1)
            }
            Instruction::Nop => write!(f, "nop"),
            Instruction::Label(addr) => write!(f, ".LABEL_{addr}"),
        }
    }
}

impl Instruction {
    /// Returns `true` if the instruction is considered a terminator.
    pub fn terminator(&self) -> bool {
        match self {
            Self::Label(..)
            | Self::Jump(..)
            | Self::Branch(..)
            | Self::Return(..) => true,
            _ => false,
        }
    }

    /// Returns `true` if the instruction is a label, which is a pseudo
    /// instruction used to mark offsets in the instructions slice.
    pub fn label(&self) -> bool {
        match self {
            Self::Label(..) => true,
            _ => false,
        }
    }

    /// Returns the assignment destination of an IR instruction.
    pub fn destination(&self) -> Option<&Symbol> {
        match self {
            Self::Id(dst, ..) => Some(dst),
            Self::Const(dst, ..) => Some(dst),
            Self::Add(dst, ..) => Some(dst),
            Self::Sub(dst, ..) => Some(dst),
            Self::Mul(dst, ..) => Some(dst),
            Self::Div(dst, ..) => Some(dst),
            Self::Eq(dst, ..) => Some(dst),
            Self::Neq(dst, ..) => Some(dst),
            Self::Lt(dst, ..) => Some(dst),
            Self::Lte(dst, ..) => Some(dst),
            Self::Gt(dst, ..) => Some(dst),
            Self::Gte(dst, ..) => Some(dst),
            Self::Branch(..) => None,
            Self::Jump(..) => None,
            Self::Return(..) => None,
            Self::Label(..) => None,
            _ => todo!("Todo {self}"),
        }
    }

    /// Returns the operands of an IR instruction, our IR is in three-address
    /// form so the operands will at most be two. The return value convention
    /// will be left to right.
    pub fn operands(&self) -> (Option<&Value>, Option<&Value>) {
        match self {
            Self::Id(.., operand) => (Some(operand), None),
            Self::Const(.., operand) => (Some(operand), None),
            Self::Add(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Sub(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Mul(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Div(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::And(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Or(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Neg(.., operand) => (Some(operand), None),
            Self::Not(.., operand) => (Some(operand), None),
            Self::Eq(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Neq(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Lt(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Lte(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Gt(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Gte(.., lhs, rhs) => (Some(lhs), Some(rhs)),
            Self::Branch(operand, ..) => (Some(operand), None),
            Self::Jump(..) => (None, None),
            Self::Label(..) => (None, None),
            Self::Return(operand) => (Some(operand), None),
            _ => todo!("{self}"),
        }
    }

    /// Returns the instruction opcode as `OPCode`.
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

/// `BasicBlock` is the atomic unit of the graph IR and represent a node in
/// the control flow graph. Each basic block represent a single block of
/// instructions assumed to execute sequentially, branches, indirect jumps
/// are edges between basic blocks.
///
/// Each basic block has a set of input edges and a set of output edges
/// each edge can describe either an unconditional jump with a target label
/// a conditional jump with two target labels, a functional call or a function
/// return.
///
/// Edges hold a single instruction that describes which control flow operation
/// is executed on that particular edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    // BlockID, assigned during initialization.
    id: BlockRef,
    // Instructions that constitute the basic block.
    instrs: Vec<Instruction>,
    // Set of possible entry edges.
    entry_points: Vec<BlockRef>,
    // Set of possible exit edges.
    exit_points: Vec<BlockRef>,
}

/// We follow the same approach in the AST when building the graph, it's
/// flattened and doesn't use pointers.
///
/// The first handle or reference we expose is a `BlockRef` which is used
/// to reference basic blocks (the nodes in the graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockRef(pub usize);

impl Default for BasicBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicBlock {
    /// Create a new `BasicBlock` instance.
    pub fn new() -> Self {
        Self {
            id: BlockRef(0),
            instrs: Vec::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
        }
    }

    /// Returns the number of instructions in the block.
    pub fn len(&self) -> usize {
        self.instrs.len()
    }

    /// Returns a non-mutable reference to the block leader.
    pub fn leader(&self) -> Option<&Instruction> {
        self.instrs.first()
    }

    /// Returns a non-mutable reference to the block terminator.
    pub fn terminator(&self) -> Option<&Instruction> {
        self.instrs.last()
    }

    /// Returns a non-mutable reference to the block instructions.
    pub fn instructions(&self) -> &[Instruction] {
        &self.instrs
    }

    /// Returns a mutable reference to the block instructions.
    pub fn instructions_mut(&mut self) -> &mut Vec<Instruction> {
        &mut self.instrs
    }

    /// Drop the instruction at the given index and return it, has the same
    /// semantics as `Vec::remove`.
    pub fn remove(&mut self, index: usize) -> Instruction {
        self.instrs.remove(index)
    }

    /// Kill the instruction at the given index by swapping it with a `Nop`.
    pub fn kill(&mut self, index: usize) -> Instruction {
        assert!(index < self.instrs.len());
        std::mem::replace(&mut self.instrs[index], Instruction::Nop)
    }

    /// Push an instruction to the basic block.
    pub fn push(&mut self, inst: &Instruction) {
        self.instrs.push(inst.clone())
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for inst in &self.instrs {
            writeln!(f, "{inst}")?;
        }
        Ok(())
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
    pub body: Vec<Instruction>,
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

    /// Push an instruction to the function's body.
    fn push(&mut self, inst: &Instruction) {
        self.body.push(inst.clone())
    }

    /// Returns a non-mutable slice of the function's body.
    pub fn instructions(&self) -> &[Instruction] {
        &self.body
    }

    /// Returns a mutable slice of the function's body.
    pub fn instructions_mut(&mut self) -> &mut [Instruction] {
        self.body.as_mut()
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.name)?;

        if !self.args.is_empty() {
            write!(f, "(")?;

            for (i, arg) in self.args.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{arg}")?;
            }
            write!(f, ")")?;
        }

        write!(f, ": {}", self.return_type)?;

        writeln!(f, " {{")?;

        for inst in &self.body {
            writeln!(f, "   {inst}")?;
        }
        writeln!(f, "}}")
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
pub struct GlobalValue(Symbol, Literal);

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
    pub fn new(ast: &'a ast::AST, symbol_table: &'a sema::SymbolTable) -> Self {
        Self {
            program: vec![],
            globals: vec![],
            llc: LocationLabelCounter(0, 0),
            tracker: IRBuilderTrackingRef(0, 0, Scope::Global),
            ast,
            symbol_table,
        }
    }
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
    }

    /// Returns a non-mutable reference to the program functions.
    pub const fn functions(&self) -> &Vec<Function> {
        &self.program
    }

    /// Returns a mutable reference to the program functions.
    pub fn functions_mut(&mut self) -> &mut [Function] {
        &mut self.program
    }

    /// Returns a non-mutable reference to the program globals.
    pub const fn globals(&self) -> &Vec<GlobalValue> {
        &self.globals
    }

    pub fn gen(&mut self) {
        for decl in self.ast.declarations() {
            let _ = self.visit_decl(decl);
        }
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
                let ret = Instruction::Return(value.clone().expect(
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
                code.push(Instruction::Const(
                    dst.clone(),
                    Value::ConstantLiteral(Literal::Int(value)),
                ));
                (Some(Value::StorageLocation(dst)), code)
            }
            ast::Expr::BoolLiteral(value) => {
                let mut code = vec![];
                let dst = Symbol::new(
                    format!("%v{}", self.llc.next_location()).as_str(),
                    Type::Bool,
                );
                code.push(Instruction::Const(
                    dst.clone(),
                    Value::ConstantLiteral(Literal::Bool(value)),
                ));
                (Some(Value::StorageLocation(dst)), code)
            }
            ast::Expr::CharLiteral(value) => {
                let mut code = vec![];
                let dst = Symbol::new(
                    format!("%v{}", self.llc.next_location()).as_str(),
                    Type::Char,
                );
                code.push(Instruction::Const(
                    dst.clone(),
                    Value::ConstantLiteral(Literal::Char(value)),
                ));
                (Some(Value::StorageLocation(dst)), code)
            }
            ast::Expr::UnaryOp { operator, operand } => {
                let (operand, mut code) =
                    if let Some(expr) = self.ast.get_expr(operand) {
                        self.visit_expr(expr)
                    } else {
                        unreachable!(
                            "Expected right handside to be a valid expression"
                        )
                    };

                let dst = match operator {
                    ast::UnaryOperator::Neg => Symbol::new(
                        format!("%v{}", self.llc.next_location()).as_str(),
                        Type::Int,
                    ),
                    ast::UnaryOperator::Not => Symbol::new(
                        format!("%v{}", self.llc.next_location()).as_str(),
                        Type::Bool,
                    ),
                };
                let _dst = dst.clone();
                let inst = match operator {
                    ast::UnaryOperator::Neg => Instruction::Neg(
                        dst,
                        operand.expect(
                            "Expected right handside to be in a temporary",
                        ),
                    ),
                    ast::UnaryOperator::Not => Instruction::Not(
                        dst,
                        operand.expect(
                            "Expected right handside to be in a temporary",
                        ),
                    ),
                };
                code.push(inst);
                (Some(Value::StorageLocation(_dst)), code)
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

                let t = match operator {
                    ast::BinaryOperator::Add
                    | ast::BinaryOperator::Sub
                    | ast::BinaryOperator::Mul
                    | ast::BinaryOperator::Div => Type::Int,
                    _ => Type::Bool,
                };

                let dst = Symbol::new(
                    format!("%v{}", self.llc.next_location()).as_str(),
                    t,
                );
                let lhs = lhs.expect("Expected valid left handside value");
                let rhs = rhs.expect("Expected valid right handside value");
                let _dst = dst.clone();
                let inst = match operator {
                    ast::BinaryOperator::Add => Instruction::Add(dst, lhs, rhs),
                    ast::BinaryOperator::Sub => Instruction::Sub(dst, lhs, rhs),
                    ast::BinaryOperator::Mul => Instruction::Mul(dst, lhs, rhs),
                    ast::BinaryOperator::Div => Instruction::Div(dst, lhs, rhs),
                    ast::BinaryOperator::Eq => Instruction::Eq(dst, lhs, rhs),
                    ast::BinaryOperator::Neq => Instruction::Neq(dst, lhs, rhs),
                    ast::BinaryOperator::Gt => Instruction::Gt(dst, lhs, rhs),
                    ast::BinaryOperator::Gte => Instruction::Gte(dst, lhs, rhs),
                    ast::BinaryOperator::Lt => Instruction::Lt(dst, lhs, rhs),
                    ast::BinaryOperator::Lte => Instruction::Lte(dst, lhs, rhs),
                    ast::BinaryOperator::And => Instruction::And(dst, lhs, rhs),
                    ast::BinaryOperator::Or => Instruction::Or(dst, lhs, rhs),
                };

                code.push(inst);
                (Some(Value::StorageLocation(_dst)), code)
            }
            ast::Expr::Named(ref name) => {
                let _t = match self.symbol_table.find(name, self.tracker.1) {
                    Some(symbol) => symbol.t(),
                    None => unreachable!(
                        "Expected a symbol for named expression : `{name}`"
                    ),
                };
                let name = Symbol::new(name, Type::from(&_t));
                (Some(Value::StorageLocation(name)), vec![])
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
                let location = match lhs {
                    Some(Value::StorageLocation(ref sym)) => sym.0.clone(),
                    _ => unreachable!(
                        "Expected assignment lvalue to be a storage location"
                    ),
                };
                let t = match self.symbol_table.find(&location, self.tracker.1)
                {
                    Some(symbol) => symbol.t(),
                    None => unreachable!(
                        "Expected a symbol for named expression : `{}`",
                        lhs.unwrap()
                    ),
                };
                let inst = Instruction::Id(
                    Symbol::new(&location, Type::from(&t)),
                    rhs.expect(
                        "Expected assignment rvalue to be a valid value",
                    ),
                );
                code.push(inst);
                (lhs.clone(), code)
            }
            ast::Expr::Call { name, ref args } => {
                let name = match self.ast.get_expr(name) {
                    Some(ast::Expr::Named(name)) => name,
                    _ => unreachable!("Expected reference to be a named expression for a function"),
                };
                let t = match self.symbol_table.find(name, self.tracker.1) {
                    Some(symbol) => symbol.t(),
                    None => unreachable!(
                        "Expected a symbol for named expression : `{name}`"
                    ),
                };
                let (vars, code): (Vec<Value>, Vec<Vec<Instruction>>) = args
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
                let dst = Symbol::new(
                    format!("%v{}", self.llc.next_location()).as_str(),
                    Type::from(&t),
                );
                let inst = Instruction::Call(
                    dst.clone(),
                    Symbol::new(name, Type::from(&t)),
                    vars,
                );
                code.push(inst);
                (Some(Value::StorageLocation(dst)), code)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;
    use crate::scanner::Scanner;
    use crate::sema::analyze;

    // Macro to generate test cases.
    macro_rules! test_ir_gen {
        ($name:ident, $source:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let source = $source;
                let mut scanner = Scanner::new(source);
                let tokens = scanner
                    .scan()
                    .expect("expected test case source to be valid");
                let mut parser = Parser::new(&tokens);
                parser.parse();
                let symbol_table = analyze(parser.ast());

                let mut irgen = IRBuilder::new(parser.ast(), &symbol_table);
                irgen.gen();

                let mut actual = "".to_string();
                for func in irgen.functions() {
                    // println!("{func}");
                    actual.push_str(format!("{func}").as_str());
                }
                // For readability trim the newlines at the start and end
                // of our IR text fixture.
                let expected = $expected
                    .strip_suffix("\n")
                    .and($expected.strip_prefix("\n"));
                assert_eq!(actual, expected.unwrap())
            }
        };
    }

    test_ir_gen!(
        can_generate_const_ops,
        "int main() { return 0;}",
        r#"
@main: int {
   %v0: int = const 0
   ret %v0
}
"#
    );

    test_ir_gen!(
        can_generate_unary_ops,
        "int main() { bool a = !true; return 0;}",
        r#"
@main: int {
   %v0: bool = const true
   %v1: bool = not %v0
   a: bool = id %v1
   %v2: int = const 0
   ret %v2
}
"#
    );

    test_ir_gen!(
        can_generate_binary_logical_ops,
        "int main() { bool a = (true && false) || false; return 0;}",
        r#"
@main: int {
   %v0: bool = const true
   %v1: bool = const false
   %v2: bool = and %v0 %v1
   %v3: bool = const false
   %v4: bool = or %v2 %v3
   a: bool = id %v4
   %v5: int = const 0
   ret %v5
}
"#
    );
    test_ir_gen!(
        can_generate_multiple_assignments,
        r#"int main() {
            bool a = !true;
            bool b = !false;
            bool c = !a;
            bool d = !b;
            int e = 12345679;
            int f = -e;
            return f;
        }"#,
        r#"
@main: int {
   %v0: bool = const true
   %v1: bool = not %v0
   a: bool = id %v1
   %v2: bool = const false
   %v3: bool = not %v2
   b: bool = id %v3
   %v4: bool = not a
   c: bool = id %v4
   %v5: bool = not b
   d: bool = id %v5
   %v6: int = const 12345679
   e: int = id %v6
   %v7: int = neg e
   f: int = id %v7
   ret f
}
"#
    );
    test_ir_gen!(
        can_generate_function_arguments,
        "int main() {} int f(int a, int b) { return a + b;}",
        r#"
@main: int {
}
@f(a: int, b: int): int {
   %v0: int = add a b
   ret %v0
}
"#
    );
    test_ir_gen!(
        can_generate_binary_ops,
        r#"int main() {
            int a = 1 + 1;
            int b = 2 - 2;
            int c = 3 * 3;
            int d = 4 / 4;
            bool e = a == b;
            bool f = b != c;
            bool g = c > d;
            bool h = c >= d;
            bool i = d < c;
            bool j = d <= c;
            return 0;
        }"#,
        r#"
@main: int {
   %v0: int = const 1
   %v1: int = const 1
   %v2: int = add %v0 %v1
   a: int = id %v2
   %v3: int = const 2
   %v4: int = const 2
   %v5: int = sub %v3 %v4
   b: int = id %v5
   %v6: int = const 3
   %v7: int = const 3
   %v8: int = mul %v6 %v7
   c: int = id %v8
   %v9: int = const 4
   %v10: int = const 4
   %v11: int = div %v9 %v10
   d: int = id %v11
   %v12: bool = eq a b
   e: bool = id %v12
   %v13: bool = neq b c
   f: bool = id %v13
   %v14: bool = gt c d
   g: bool = id %v14
   %v15: bool = gte c d
   h: bool = id %v15
   %v16: bool = lt d c
   i: bool = id %v16
   %v17: bool = lte d c
   j: bool = id %v17
   %v18: int = const 0
   ret %v18
}
"#
    );
    test_ir_gen!(
        can_generate_assignment,
        r#"int main() {
            int a;
            a = 42;
            return a;
}"#,
        r#"
@main: int {
   %v0: int = const 0
   a: int = id %v0
   %v1: int = const 42
   a: int = id %v1
   ret a
}
"#
    );
    test_ir_gen!(
        can_generate_assignments,
        r#"int main() {
            int a;
            int b;
            int c = 42;
            a = c;
            b = a;
            return b;
        }"#,
        r#"
@main: int {
   %v0: int = const 0
   a: int = id %v0
   %v1: int = const 0
   b: int = id %v1
   %v2: int = const 42
   c: int = id %v2
   a: int = id c
   b: int = id a
   ret b
}
"#
    );
    test_ir_gen!(
        can_generate_function_calls,
        r#"
        int f(int a, int b) { return a + b;}
            int main() {
                return f(1,2);
            }
        "#,
        r#"
@f(a: int, b: int): int {
   %v0: int = add a b
   ret %v0
}
@main: int {
   %v1: int = const 1
   %v2: int = const 2
   %v3: int = call @f %v1 %v2
   ret %v3
}
"#
    );
    test_ir_gen!(
        can_generate_if_block_without_else_branch,
        r#"
        int main() {
            int a = 42;
            int b = 17;
            if (a > b) {
                return a - b;
            }
            return a + b;
        }
        "#,
        r#"
@main: int {
   %v0: int = const 42
   a: int = id %v0
   %v1: int = const 17
   b: int = id %v1
   %v2: bool = gt a b
   br %v2 .LABEL_0 .LABEL_1
   .LABEL_0
   %v3: int = sub a b
   ret %v3
   .LABEL_1
   jmp .LABEL_2
   .LABEL_2
   %v4: int = add a b
   ret %v4
}
"#
    );
    test_ir_gen!(
        can_generate_for_loop,
        r#"
            int main() {
                int i = 0;
                int x = 0;
                for (i = 1;i <= 100;i = i+1) {
                    x = i + 1;
                }
                return 0;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 0
   i: int = id %v0
   %v1: int = const 0
   x: int = id %v1
   %v2: int = const 1
   i: int = id %v2
   .LABEL_0
   %v3: int = const 1
   %v4: int = add i %v3
   x: int = id %v4
   %v5: int = const 1
   %v6: int = add i %v5
   i: int = id %v6
   %v7: int = const 100
   %v8: bool = lte i %v7
   br %v8 .LABEL_0 .LABEL_1
   .LABEL_1
   %v9: int = const 0
   ret %v9
}
"#
    );
    test_ir_gen!(
        can_generate_while_loop,
        r#"
            int main() {
                int i = 0;
                int x = 0;
                while (i <= 100) {
                    x = x + 1;
                    i = i + 1;
                }
                return x;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 0
   i: int = id %v0
   %v1: int = const 0
   x: int = id %v1
   .LABEL_0
   %v2: int = const 1
   %v3: int = add x %v2
   x: int = id %v3
   %v4: int = const 1
   %v5: int = add i %v4
   i: int = id %v5
   %v6: int = const 100
   %v7: bool = lte i %v6
   br %v7 .LABEL_0 .LABEL_1
   .LABEL_1
   ret x
}
"#
    );

    test_ir_gen!(
        can_generate_nested_scopes,
        r#"
            int main() {
                int i = 0;
                int x = 0;
                {
                    int x = 1;
                    int y = 2;
                    int i = x + y;
                }
                return x;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 0
   i: int = id %v0
   %v1: int = const 0
   x: int = id %v1
   %v2: int = const 1
   x: int = id %v2
   %v3: int = const 2
   y: int = id %v3
   %v4: int = add x y
   i: int = id %v4
   ret x
}
"#
    );

    test_ir_gen!(
        can_expand_nested_expressions,
        r#"
            int main() {
                int x = (5 * 3 + (7 / 2) - 1) * 2/7;
                int y = ((4 - 3) * (7 + 5)) * 1/7;
                int z = (5 - 5) * (4 + 4 - 16);
                int w = x * y - z;

                return w;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 5
   %v1: int = const 3
   %v2: int = mul %v0 %v1
   %v3: int = const 7
   %v4: int = const 2
   %v5: int = div %v3 %v4
   %v6: int = add %v2 %v5
   %v7: int = const 1
   %v8: int = sub %v6 %v7
   %v9: int = const 2
   %v10: int = mul %v8 %v9
   %v11: int = const 7
   %v12: int = div %v10 %v11
   x: int = id %v12
   %v13: int = const 4
   %v14: int = const 3
   %v15: int = sub %v13 %v14
   %v16: int = const 7
   %v17: int = const 5
   %v18: int = add %v16 %v17
   %v19: int = mul %v15 %v18
   %v20: int = const 1
   %v21: int = mul %v19 %v20
   %v22: int = const 7
   %v23: int = div %v21 %v22
   y: int = id %v23
   %v24: int = const 5
   %v25: int = const 5
   %v26: int = sub %v24 %v25
   %v27: int = const 4
   %v28: int = const 4
   %v29: int = add %v27 %v28
   %v30: int = const 16
   %v31: int = sub %v29 %v30
   %v32: int = mul %v26 %v31
   z: int = id %v32
   %v33: int = mul x y
   %v34: int = sub %v33 z
   w: int = id %v34
   ret w
}
"#
    );
    test_ir_gen!(
        can_canonicalize_constant_literals,
        r#"
            int main() {
                return 1+1;
            }
        "#,
        r#"
@main: int {
   %v0: int = const 1
   %v1: int = const 1
   %v2: int = add %v0 %v1
   ret %v2
}
"#
    );
}
