//! Intermediate representation definition and implementation.
//!
//! Glouton Intermediate Representation is a linear SSA oriented intermediate
//! representation. The intermediate representation is based on Bril and
//! has three core instruction types, constant operations which produce
//! constant values, value operations which take operands and produce values
//! and effect based operations which take operands and produce no values.
use std::fmt::{self};

use crate::{
    ast::{self, Visitor},
    sema::{self, SymbolTable},
};

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

/// ConstOp are opcodes for constant operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstOp {
    /// `const` operation.
    Const,
}

impl fmt::Display for ConstOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const => write!(f, "const"),
        }
    }
}

/// EffectOp are opcodes for effect operations such as control flow operations
/// (indirect jumps, conditional branches, calls...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectOp {
    // Indirect jumps.
    Jump,
    // Condtional branches.
    Branch,
    // Function calls that don't produce values.
    Call,
    // Return statements.
    Return,
}

impl fmt::Display for EffectOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jump => write!(f, "jmp"),
            Self::Branch => write!(f, "br"),
            Self::Call => write!(f, "call"),
            Self::Return => write!(f, "ret"),
        }
    }
}

/// ValueOp are opcodes for value operations, a value operation is any value
/// producing operation. Such as arithemtic operations, comparison operations
/// and loads or stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueOp {
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
    // Function calls that produce values.
    Call,
    // Identity operator.
    Id,
}

/// OPCode is a type wrapper around all opcodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    // Label instructions.
    Label,
    // Nop instruction.
    Nop,
}

impl fmt::Display for ValueOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::Mul => write!(f, "mul"),
            Self::Div => write!(f, "div"),
            Self::Eq => write!(f, "eq"),
            Self::Neq => write!(f, "neq"),
            Self::Lt => write!(f, "lt"),
            Self::Gt => write!(f, "gt"),
            Self::Lte => write!(f, "lte"),
            Self::Gte => write!(f, "gte"),
            Self::Not => write!(f, "not"),
            Self::Neg => write!(f, "neg"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Call => write!(f, "call"),
            Self::Id => write!(f, "id"),
        }
    }
}

impl ValueOp {
    /// Convert an AST operator node to an instruction.
    fn from(operator: &ast::BinaryOperator) -> ValueOp {
        match operator {
            ast::BinaryOperator::Add => ValueOp::Add,
            ast::BinaryOperator::Sub => ValueOp::Sub,
            ast::BinaryOperator::Mul => ValueOp::Mul,
            ast::BinaryOperator::Div => ValueOp::Div,
            ast::BinaryOperator::Eq => ValueOp::Eq,
            ast::BinaryOperator::Neq => ValueOp::Neq,
            ast::BinaryOperator::Lt => ValueOp::Lt,
            ast::BinaryOperator::Lte => ValueOp::Lte,
            ast::BinaryOperator::Gt => ValueOp::Gt,
            ast::BinaryOperator::Gte => ValueOp::Gte,
        }
    }
}

/// Symbol references are used as an alternative to variable names.
pub struct SymbolRef(usize);

pub struct Label(usize);

/// Every value in the intermediate representation is either a symbol reference
/// or a literal value.
///
/// Consider the `add` instruction, it is always applied to two operands which
/// can either be a symbol reference to a variable or a literal value. Since
/// the intermediate representation is linear and follows SSA-form everything
/// gets assigned to a storage location before it being used.
enum Value {
    StorageLocation(SymbolRef),
    ConstantLiteral(Literal),
}

enum Inst {
    Const(
        SymbolRef, /* Destination */
        Literal,   /* Literal value assigned to the destination */
    ),
    Add(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Sub(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Mul(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Div(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Neg(
        SymbolRef, /* Destination */
        Value,     /* LHS */
    ),
    And(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Or(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Not(
        SymbolRef, /* Destination */
        Value,     /* LHS */
    ),
    Eq(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Neq(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Lt(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Lte(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Gt(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Gte(
        SymbolRef, /* Destination */
        Value,     /* LHS */
        Value,     /* RHS */
    ),
    Return(Value /* Return value */),
    Call(SymbolRef /*Call Target */),
    Jump(Label /* Offset */),
    Branch(SymbolRef /*Condition */, Label /* Then Target Offset */,  Label/* Else Target Offset */),
    Id(Value),
    Nop,
}

impl Inst {
    pub fn opcode(&self) -> OPCode {
        match self {
            Inst::Add(..) => OPCode::Add,
            Inst::Const(..) => OPCode::Const,
            Inst::Sub(..) => OPCode::Sub,
            Inst::Mul(..) => OPCode::Mul,
            Inst::Div(..) => OPCode::Div,
            Inst::Eq(..) => OPCode::Eq,
            Inst::Neq(..) => OPCode::Neq,
            Inst::Lt(..) => OPCode::Lt,
            Inst::Lte(..) => OPCode::Lte,
            Inst::Gt(..) => OPCode::Gt,
            Inst::Gte(..) => OPCode::Gte,
            _ => todo!(),
        }
    }
}

/// Why we want to above implementation ?
///
/// Consider the following algorithm that does constant propagations.
///
/// def run(f: function):
///   bbs = f.basic_blocks()
///   for bb in bbs:
///     folded = fold(bb->insts())
///     bb.rewrite_as(folded)
///
/// def fold(insts):
///   foreach(inst, insts):
///     _fold(inst)
///
///
/// def _fold(inst):
///   match inst:
///     add(dst, lhs, rhs) => {
///       _lhs = _resolve_folded(lhs)
///       _rhs = _resolve_folded(rhs)
///       inst.rewrite(add(dst, _lhs, _rhs)
///     }
///     
///

/// Instructions in GIR are split into three groups, each group describe a set
/// of behaviors that the instructions implement.
///
/// Each behavior set describe certain semantics, enumerated below :
///
/// * Constant Operations: produce constant values to a destination and have
///   no side effects.
///
/// * Value Operations: produce dynamic values to a destination, consequently
///   they have no side effects.
///
/// * Effect Operations: produce behavior such as redirecting control flow and
///   thus have side effects.
///
/// `%a: int = const 5` for example is a constant operation that produces an
/// integer value (5) to a target variable `%a`.
///
/// `br cond .label_1 .label_2` is an effect operation that produces no value
/// but describes the behavior of a branch condition to either one of the two
/// provided labels.
///
/// `%res = add %var1 %var2` is a value operaiton that produces the sum of two
/// variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    // Nop instruction are instructions that produce no values or side effects.
    Nop,
    // Constant instructions are instructions that produce a single constant
    // typed value to a destination variable.
    Constant {
        // Destination variable name.
        dst: String,
        // Opcode for the instruction, currently only `const`.
        op: ConstOp,
        // Type of the variable.
        const_type: Type,
        // Literal value stored.
        value: Literal,
    },

    // Value instructions that produce a value from arguments where each opcode
    // enforces the rules for the arguments.
    Value {
        // List of argument names (variable, function or label).
        args: Vec<String>,
        // Destination variable name.
        dst: String,
        // Opcode for the instruction.
        op: ValueOp,
        // Type of the variable.
        op_type: Type,
    },
    // Effect instructions that incur a side effect.
    Effect {
        // List of argument names (variable, function or label).
        args: Vec<String>,
        // Opcode for the instruction.
        op: EffectOp,
    },
    // Label instructions are unique, they have no effect of any kind and only
    // serve to mark blocks in conditions, jumps and loops.
    Label {
        name: String,
    },
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nop => write!(f, "NOP"),
            Self::Constant {
                dst,
                op,
                const_type,
                value,
            } => {
                write!(f, "{dst}: {const_type} = {op} {value}")
            }
            Self::Value {
                args,
                dst,
                op,
                op_type,
            } => {
                write!(f, "{dst}: {op_type} = {op} ")?;
                for arg in args {
                    match op {
                        ValueOp::Call => write!(f, " @{arg} ")?,
                        _ => write!(f, "{arg} ")?,
                    }
                }
                Ok(())
            }
            Self::Effect { args, op } => {
                write!(f, "{op} ")?;
                for arg in args {
                    match op {
                        EffectOp::Jump => write!(f, "{arg}")?,
                        EffectOp::Call => write!(f, "@{arg}")?,
                        EffectOp::Branch => write!(f, "{arg} ")?,
                        EffectOp::Return => write!(f, "{arg}")?,
                    }
                }
                Ok(())
            }
            Self::Label { name } => {
                write!(f, "{name}")
            }
        }
    }
}

/// Instruction emitters are all owning functions, they take ownership
/// of their arguments to build an `Instruction`.
impl Instruction {
    /// Returns `true` if the instruction is considered a terminator.
    pub fn is_terminator(&self) -> bool {
        match *self {
            Self::Effect { op, .. } => match op {
                EffectOp::Jump | EffectOp::Branch | EffectOp::Return => true,
                EffectOp::Call => false,
            },
            Self::Label { .. } => true,
            _ => false,
        }
    }

    /// Returns `true` if the instruction is a label.
    pub fn is_label(&self) -> bool {
        matches!(self, &Self::Label { .. })
    }

    /// Return the instruction's opcode..
    pub fn opcode(&self) -> OPCode {
        match self {
            Self::Effect { op, .. } => match op {
                EffectOp::Call => OPCode::Call,
                EffectOp::Return => OPCode::Return,
                EffectOp::Branch => OPCode::Branch,
                EffectOp::Jump => OPCode::Jump,
            },
            Self::Value { op, .. } => match op {
                ValueOp::Eq => OPCode::Eq,
                ValueOp::Lt => OPCode::Lt,
                ValueOp::Gt => OPCode::Gt,
                ValueOp::Or => OPCode::Or,
                ValueOp::Id => OPCode::Id,
                ValueOp::Add => OPCode::Add,
                ValueOp::Sub => OPCode::Sub,
                ValueOp::Mul => OPCode::Mul,
                ValueOp::Div => OPCode::Div,
                ValueOp::Neq => OPCode::Neq,
                ValueOp::Lte => OPCode::Lte,
                ValueOp::Gte => OPCode::Gte,
                ValueOp::Not => OPCode::Not,
                ValueOp::Neg => OPCode::Neq,
                ValueOp::And => OPCode::And,
                ValueOp::Call => OPCode::Call,
            },
            Self::Nop => OPCode::Nop,
            Self::Constant { .. } => OPCode::Const,
            Self::Label { .. } => OPCode::Label,
        }
    }

    /// Returns the destination of the instruction if there is one.
    pub fn dst(&self) -> Option<&String> {
        match self {
            Instruction::Constant { dst, .. } | Instruction::Value { dst, .. } => Some(dst),
            _ => None,
        }
    }

    /// Returns the arguments of an instruction if there are any.
    pub fn args(&self) -> Option<&Vec<String>> {
        match self {
            Instruction::Effect { args, .. } | Instruction::Value { args, .. } => Some(args),
            _ => None,
        }
    }

    /// Return the `Nop` instruction.
    pub fn nop() -> Instruction {
        Instruction::Nop
    }

    /// Emit a constant instruction.
    fn constant(dst: String, const_type: Type, value: Literal) -> Instruction {
        Instruction::Constant {
            dst,
            op: ConstOp::Const,
            const_type,
            value,
        }
    }

    /// Emit a jump instruction.
    fn jmp(target: String) -> Instruction {
        Instruction::Effect {
            args: vec![target],
            op: EffectOp::Jump,
        }
    }

    /// Emit a branch instruction.
    fn branch(cond: String, then_target: String, else_target: String) -> Instruction {
        Instruction::Effect {
            args: vec![cond, then_target, else_target],
            op: EffectOp::Branch,
        }
    }

    /// Emit a label instruction.
    fn label(name: String) -> Instruction {
        Instruction::Label { name }
    }

    /// Emit a return instruction.
    fn ret(value: String) -> Instruction {
        Instruction::Effect {
            args: vec![value],
            op: EffectOp::Return,
        }
    }

    /// Emit an arithmetic operation.
    fn arith(dst: String, op_type: Type, op: ValueOp, args: Vec<String>) -> Instruction {
        Instruction::Value {
            args,
            dst,
            op,
            op_type,
        }
    }

    /// Emit a comparison operation.
    fn cmp(dst: String, op_type: Type, op: ValueOp, args: Vec<String>) -> Instruction {
        Instruction::Value {
            args,
            dst,
            op,
            op_type,
        }
    }

    /// Emit an identity operation.
    fn id(dst: String, op_type: Type, args: Vec<String>) -> Instruction {
        Instruction::Value {
            args,
            dst,
            op: ValueOp::Id,
            op_type,
        }
    }

    /// Emit a call operation.
    fn call(dst: String, op_type: Type, name: String, mut args: Vec<String>) -> Instruction {
        // Name is the first argument.
        let mut func_args = vec![name];
        func_args.append(&mut args);

        Instruction::Value {
            args: func_args,
            dst,
            op: ValueOp::Call,
            op_type,
        }
    }
}

/// `Argument` is a wrapper around a name and type it represents a single
/// function argument.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Argument {
    // Argument name.
    name: String,
    // Argument type.
    kind: Type,
}

impl fmt::Display for Argument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.kind)
    }
}

impl Argument {
    fn new(name: String, kind: Type) -> Self {
        Self { name, kind }
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

/// `Function` represents a function declaration in the AST, a `Function`
/// is composed as a linear sequence of GIR instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    // Function name.
    name: String,
    // List of arguments the function accepts.
    args: Vec<Argument>,
    // Body of the function as GIR instructions.
    pub body: Vec<Instruction>,
    // Return type of the function if any.
    return_type: Type,
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

impl Function {
    /// Create a new function with a name.
    fn new(name: String, args: Vec<Argument>, return_type: Type) -> Self {
        Self {
            name,
            args,
            return_type,
            body: vec![],
        }
    }

    /// Push an instruction.
    fn push(&mut self, inst: Instruction) {
        self.body.push(inst)
    }

    /// Return a non-mutable reference to the function instructions.
    pub fn instructions(&self) -> &[Instruction] {
        &self.body
    }

    /// Return a mutable reference to the function instructions.
    pub fn instructions_mut(&mut self) -> &mut [Instruction] {
        self.body.as_mut()
    }

    /// Return a mutable reference to the function instructions as a `Vec`.
    pub fn instructions_mut_vec(&mut self) -> &mut Vec<Instruction> {
        &mut self.body
    }

    /// Insert the instruction at the given position.
    pub fn insert(&mut self, inst: &Instruction, pos: usize) {
        self.body.insert(pos, inst.clone())
    }

    /// Remove the instruction at the given position.
    pub fn remove(&mut self, pos: usize) {
        let _ = self.body.remove(pos);
    }

    /// Form a list of basic blocks from the function, the ownership of
    /// the returned `Vec` is transferred to the caller.
    pub fn form_basic_blocks(&self) -> Vec<BasicBlock> {
        let mut blocks = Vec::new();
        let mut current = BasicBlock::new();
        for inst in &self.body {
            if inst.is_label() {
                if !current.instructions().is_empty() {
                    blocks.push(current)
                }
                match inst {
                    Instruction::Label { .. } => {
                        current = BasicBlock::new();
                        current.push(inst);
                    }
                    _ => unreachable!(),
                }
            } else {
                current.push(inst);
                if inst.is_terminator() {
                    blocks.push(current);
                    current = BasicBlock::new();
                }
            }
        }
        blocks
    }

    /// Reassemble a list of basic blocks into the current function body
    /// replacing its existing instructions.
    ///
    /// The function consumes the input blocks, leaving them empty.
    pub fn reassemble_from_basic_blocks(&mut self, blocks: &mut [BasicBlock]) {
        let mut instructions = Vec::with_capacity(self.body.len());

        for block in blocks {
            instructions.append(block.instructions_mut())
        }

        instructions.clone_into(&mut self.body)
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

/// `IRBuilder` is responsible for building the intermediate representation of
/// an AST. The result of the process is a `Vec` of all functions defined in
/// the program and a `Vec` of all the globals declared in the program.
pub struct IRBuilder<'a> {
    // Reference to the AST we are processing.
    ast: &'a ast::AST,
    // Symbol table built during semantic analysis phase.
    symbol_table: &'a SymbolTable,
    // Program as a sequence of declared functions.
    program: Vec<Function>,
    // Global scope declarations.
    globals: Vec<Instruction>,
    // Counter used to keep track of temporaries, temporaries are storage
    // assignemnts for transient or non assigned values such as a literals.
    var_count: u32,
    // Counter used to keep track of labels used as jump targets.
    label_count: u32,
    // Index in the `program` of the current function we are building.
    curr: usize,
    // Current scope.
    scope: Scope,
    // Symbol table level we are at currently, increments by 1 when we enter
    // a scope and decrements by 1 when we exit. This is used to set the
    // starting point when resolving symbols in the symbol table.
    level: usize,
}

impl<'a> IRBuilder<'a> {
    #[must_use]
    pub fn new(ast: &'a ast::AST, symbol_table: &'a sema::SymbolTable) -> Self {
        Self {
            program: vec![],
            globals: vec![],
            var_count: 0,
            label_count: 0,
            curr: 0,
            level: 0,
            scope: Scope::Global,
            ast,
            symbol_table,
        }
    }

    /// Returns a non-mutable reference to the program functions.
    pub const fn functions(&self) -> &Vec<Function> {
        &self.program
    }

    /// Returns a mutable reference to the program functions.
    pub fn functions_mut(&mut self) -> &mut Vec<Function> {
        self.program.as_mut()
    }

    /// Returns a non-mutable reference to the program globals.
    pub const fn globals(&self) -> &Vec<Instruction> {
        &self.globals
    }

    /// Returns a fresh variable name, used for intermediate literals.
    fn next_var(&mut self) -> String {
        let var = format!("v{}", self.var_count);
        self.var_count += 1;
        var
    }

    /// Returns a fresh label, used for jumps.
    fn next_label(&mut self) -> String {
        let label = format!(".LABEL_{}", self.label_count);
        self.label_count += 1;
        label
    }

    /// Push an instruction to the current scope.
    fn push(&mut self, inst: Instruction) {
        match self.scope {
            Scope::Local => self.program[self.curr].push(inst),
            Scope::Global => {
                // The only instruction allowed in the global scope is
                // the constant instruction.
                assert!(
                    inst.opcode() == OPCode::Const || inst.opcode() == OPCode::Id,
                    "found {:?} ",
                    inst.opcode()
                );
                self.globals.push(inst)
            }
        }
    }

    /// Switch to a local scope view.
    fn enter(&mut self, func: Function) {
        self.program.push(func);
        self.level += 1;
        self.scope = Scope::Local
    }

    /// Exit back to the global scope.
    fn exit(&mut self) {
        self.scope = Scope::Global;
        self.level -= 1;
        self.curr += 1
    }

    pub fn gen(&mut self) {
        for decl in self.ast.declarations() {
            let _ = self.visit_decl(decl);
        }
    }
}

impl<'a> ast::Visitor<(Option<String>, Vec<Instruction>)> for IRBuilder<'a> {
    fn visit_decl(&mut self, decl: &ast::Decl) -> (Option<String>, Vec<Instruction>) {
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
                            Argument::new(name.to_string(), Type::from(decl_type))
                        }
                        _ => unreachable!(
                        "expected argument reference to be valid and to be `ast::Stmt::FuncArg`"
                    ),
                    })
                    .collect::<Vec<_>>();
                let func_return_type = Type::from(return_type);
                let func = Function::new(name.to_string(), func_args, func_return_type);
                // Enter a new scope and push the new function frame.
                self.enter(func);
                // Generate IR for the body.
                let (span, code) = match self.ast.get_stmt(*body) {
                    Some(stmt) => self.visit_stmt(stmt),
                    _ => unreachable!("expected body reference to be valid !"),
                };
                for inst in &code {
                    self.push(inst.clone());
                }
                // Exit back to the global scope.
                self.exit();
                (span, code)
            }
            ast::Decl::GlobalVar {
                decl_type,
                name,
                value,
            } => {
                let dst = name.clone();
                let (arg, mut code) = if let Some(expr) = self.ast.get_expr(*value) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };
                // Get the destination of the right hand side.
                code.push(Instruction::id(
                    dst.clone(),
                    Type::from(decl_type),
                    vec![arg.expect("Expected right handside to be in a temporary variable")],
                ));
                (Some(dst), code)
            }
        }
    }

    fn visit_stmt(&mut self, stmt: &ast::Stmt) -> (Option<String>, Vec<Instruction>) {
        match stmt {
            // Variable declaration are
            ast::Stmt::LocalVar {
                decl_type,
                name,
                value,
            } => {
                let dst = name.clone();
                let (arg, mut code) = if let Some(expr) = self.ast.get_expr(*value) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };
                // Get the destination of the right hand side.
                code.push(Instruction::id(
                    dst.clone(),
                    Type::from(decl_type),
                    vec![arg.expect("Expected right handside to be in a temporary variable")],
                ));
                (Some(dst), code)
            }
            // Blocks.
            ast::Stmt::Block(stmts) => {
                self.level += 1;
                let mut code = vec![];
                for stmt_ref in stmts {
                    let (_, mut block) = if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                        self.visit_stmt(stmt)
                    } else {
                        unreachable!("Expected right handside to be a valid expression")
                    };

                    code.append(&mut block);
                }
                self.level -= 1;
                (None, code)
            }
            // Return statements.
            ast::Stmt::Return(expr_ref) => {
                let (name, mut code) = if let Some(expr) = self.ast.get_expr(*expr_ref) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };

                let ret = Instruction::ret(
                    name.clone()
                        .expect("Expected right handside to be temporary or named"),
                );
                code.push(ret);
                (name, code)
            }
            // Expression statements.
            ast::Stmt::Expr(expr_ref) => {
                let (name, code) = if let Some(expr) = self.ast.get_expr(*expr_ref) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };
                (name, code)
            }
            // Conditional blocks.
            ast::Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                let (condition, mut code) = if let Some(cond) = self.ast.get_expr(*condition) {
                    self.visit_expr(cond)
                } else {
                    unreachable!("Expected condition to reference a valid expression")
                };
                let then_label = self.next_label();
                let else_label = self.next_label();
                let end_label = self.next_label();

                // Push branch instruction.
                let inst = Instruction::branch(
                    condition.expect("Expected condition variable to be valid"),
                    then_label.clone(),
                    else_label.clone(),
                );
                code.push(inst);
                // Push then label.
                let inst = Instruction::label(then_label);
                code.push(inst);
                // Generate instruction for the then block.
                let (_, mut block) = if let Some(block) = self.ast.get_stmt(*then_block) {
                    self.visit_stmt(block)
                } else {
                    unreachable!("Expected reference to block to be a valid statement")
                };
                code.append(&mut block);
                // Push a jump instruction to the end label iif the last
                // instruction was not a return..
                if code.last().is_some_and(|inst| match inst.opcode() {
                    OPCode::Return => false,
                    _ => true,
                }) {
                    let inst = Instruction::jmp(end_label.clone());
                    code.push(inst);
                }
                // Push else label.
                let inst = Instruction::label(else_label);
                code.push(inst);
                // Generate instruction for the else block if one exists.
                if else_block.is_some() {
                    let (_, mut block) = if let Some(block) = self
                        .ast
                        .get_stmt(else_block.expect("Expected else block to be `Some`"))
                    {
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
                    let inst = Instruction::jmp(end_label.clone());
                    code.push(inst);
                }
                // Push end label.
                let inst = Instruction::label(end_label);
                code.push(inst);

                (None, code)
            }
            ast::Stmt::FuncArg {
                decl_type: _,
                name: _,
            } => {
                unreachable!("Expected function argument to be handled in `visit_decl`")
            }
            ast::Stmt::For {
                init,
                condition,
                iteration,
                body,
            } => {
                // Generate labels for the loop.
                let loop_body_label = self.next_label();
                let loop_exit_label = self.next_label();
                let mut code = Vec::new();
                // Generate initializer block if it exists.
                if init.is_some() {
                    let (_, mut block) = if let Some(block) = self
                        .ast
                        .get_expr(init.expect("Expected init block to be `Some`"))
                    {
                        self.visit_expr(block)
                    } else {
                        unreachable!("Expected reference to initializer to be a valid expression")
                    };
                    code.append(&mut block);
                }
                // Generate the loop body label.
                let inst = Instruction::label(loop_body_label.clone());
                code.push(inst);
                // Generate the loop body block.
                let (_, mut block) = if let Some(block) = self.ast.get_stmt(*body) {
                    self.visit_stmt(block)
                } else {
                    unreachable!("Expected reference to body to be a valid statement")
                };
                code.append(&mut block);
                // Generate the iteration expression.
                if iteration.is_some() {
                    let (_, mut block) = if let Some(block) = self
                        .ast
                        .get_expr(iteration.expect("Expected iteration expression to be `Some`"))
                    {
                        self.visit_expr(block)
                    } else {
                        unreachable!("Expected reference to iteration to be a valid expression")
                    };
                    code.append(&mut block);
                }
                // Generate the condition block if it exists.
                if condition.is_some() {
                    let (condition, mut block) = if let Some(block) = self
                        .ast
                        .get_expr(condition.expect("Expected condition block to be `Some`"))
                    {
                        self.visit_expr(block)
                    } else {
                        unreachable!("Expected reference to initializer to be a valid expression")
                    };
                    code.append(&mut block);
                    // Generate branch to exit code.
                    let inst = Instruction::branch(
                        condition.expect("Expected condition variable to be valid"),
                        loop_body_label.clone(),
                        loop_exit_label.clone(),
                    );
                    code.push(inst);
                }
                // Generate the loop exit code.
                let inst = Instruction::label(loop_exit_label.clone());
                code.push(inst);
                (None, code)
            }
            ast::Stmt::While { condition, body } => {
                // Generate labels for the loop.
                let loop_body_label = self.next_label();
                let loop_exit_label = self.next_label();
                let mut code = Vec::new();
                if body.is_some() {
                    // Generate the loop body label.
                    let inst = Instruction::label(loop_body_label.clone());
                    code.push(inst);
                    // Generate the loop body block.
                    let (_, mut block) = if let Some(block) = self
                        .ast
                        .get_stmt(body.expect("Expected loop body to be `Some`"))
                    {
                        self.visit_stmt(block)
                    } else {
                        unreachable!("Expected reference to body to be a valid statement")
                    };
                    code.append(&mut block);
                }
                if condition.is_some() {
                    let (condition, mut block) = if let Some(block) = self
                        .ast
                        .get_expr(condition.expect("Expected condition block to be `Some`"))
                    {
                        self.visit_expr(block)
                    } else {
                        unreachable!("Expected reference to initializer to be a valid expression")
                    };
                    code.append(&mut block);
                    // Generate branch to exit code.
                    let inst = Instruction::branch(
                        condition.expect("Expected condition variable to be valid"),
                        loop_body_label.clone(),
                        loop_exit_label.clone(),
                    );
                    code.push(inst);
                }
                // Generate the loop exit code.
                let inst = Instruction::label(loop_exit_label.clone());
                code.push(inst);
                (None, code)
            }
            ast::Stmt::Empty => (None, vec![]),
        }
    }

    fn visit_expr(&mut self, expr: &ast::Expr) -> (Option<String>, Vec<Instruction>) {
        match *expr {
            ast::Expr::IntLiteral(value) => {
                let mut code = vec![];
                let dst = self.next_var();
                code.push(Instruction::constant(
                    dst.clone(),
                    Type::Int,
                    Literal::Int(value),
                ));
                (Some(dst), code)
            }
            ast::Expr::BoolLiteral(value) => {
                let mut code = vec![];
                let dst = self.next_var();
                code.push(Instruction::constant(
                    dst.clone(),
                    Type::Bool,
                    Literal::Bool(value),
                ));
                (Some(dst), code)
            }
            ast::Expr::CharLiteral(value) => {
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
                let (name, mut code) = if let Some(expr) = self.ast.get_expr(operand) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };

                let dst = self.next_var();
                let inst = match operator {
                    ast::UnaryOperator::Neg => Instruction::arith(
                        dst.clone(),
                        Type::Int,
                        ValueOp::Neg,
                        vec![name
                            .expect("Expected right handside to be in a temporary")
                            .clone()],
                    ),
                    ast::UnaryOperator::Not => Instruction::arith(
                        dst.clone(),
                        Type::Bool,
                        ValueOp::Not,
                        vec![name
                            .expect("Expected right handside to be in a temporary")
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
                let (lhs, mut code_left) = if let Some(expr) = self.ast.get_expr(left) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };
                code.append(&mut code_left);

                let (rhs, mut code_right) = if let Some(expr) = self.ast.get_expr(right) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
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
                            lhs.expect("Expected right handside to be in a temporary")
                                .clone(),
                            rhs.expect("Expected right handside to be in a temporary")
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
                            lhs.expect("Expected right handside to be in a temporary")
                                .clone(),
                            rhs.expect("Expected right handside to be in a temporary")
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
                    None => unreachable!("Expected a symbol for named expression : `{name}`"),
                };
                (Some(name.clone()), vec![])
            }
            ast::Expr::Grouping(expr_ref) => {
                let (name, code) = if let Some(expr) = self.ast.get_expr(expr_ref) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };
                (name, code)
            }
            ast::Expr::Assignment { name, value } => {
                let mut code = vec![];
                let (rhs, mut code_right) = if let Some(expr) = self.ast.get_expr(value) {
                    self.visit_expr(expr)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };
                code.append(&mut code_right);
                // We know that by definition assignment left handside will
                // always be a named expression, unless we are dealing with
                // arrays or struct fields.
                let (lhs, _) = if let Some(named) = self.ast.get_expr(name) {
                    self.visit_expr(named)
                } else {
                    unreachable!("Expected right handside to be a valid expression")
                };
                let t = match self.symbol_table.find(&lhs.clone().unwrap(), self.level) {
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
                    None => unreachable!("Expected a symbol for named expression : `{name}`"),
                };
                let (vars, code): (Vec<String>, Vec<Vec<Instruction>>) = args
                    .iter()
                    .map(|arg| {
                        if let Some(expr) = self.ast.get_expr(*arg) {
                            let (arg, code) = self.visit_expr(expr);
                            (arg.unwrap(), code)
                        } else {
                            unreachable!("Expected argument to be a valid expression")
                        }
                    })
                    .unzip();
                let mut code: Vec<Instruction> = code.into_iter().flatten().collect();
                let dst = self.next_var();
                let inst = Instruction::call(dst.clone(), Type::from(&t), name.to_string(), vars);
                code.push(inst);
                (Some(dst), code)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::IRBuilder;
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
                println!("{symbol_table}");

                let mut irgen = IRBuilder::new(parser.ast(), &symbol_table);
                irgen.gen();

                for func in irgen.functions() {
                    println!("{func}");
                }
            }
        };
    }

    test_ir_gen!(can_generate_const_ops, "int main() { return 0;}", &vec![]);
    test_ir_gen!(
        can_generate_unary_ops,
        "int main() { bool a = !true; return 0;}",
        &vec![]
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
        &vec![]
    );
    test_ir_gen!(
        can_generate_function_arguments,
        "int main() {} int f(int a, int b) { return a + b;}",
        &vec![]
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
        &vec![]
    );
    test_ir_gen!(
        can_generate_assignment,
        r#"int main() {
            int a;
            a = 42;
            return a;
}"#,
        &vec![]
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
        &vec![]
    );
    test_ir_gen!(
        can_generate_function_calls,
        r#"
        int f(int a, int b) { return a + b;}
            int main() {
                return f(1,2);
            }
        "#,
        &vec![]
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
        &vec![]
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
        &vec![]
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
        &vec![]
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
        &vec![]
    );

    test_ir_gen!(
        can_expand_nested_expressions,
        r#"
            int main() {
                int x = (5 * 3 + (7 / 2) - 1) * 2/7;
                int y = ((4 - 3) * (7 + 5)) * 1/7;
                int z = (5 - 5) * (4 + 4 - 16);
                int v = x * y - z;

                return v;
            }
        "#,
        &vec![]
    );
}
