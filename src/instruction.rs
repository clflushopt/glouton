//! Glouton IR instructions.
use std::fmt;

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
pub struct SymbolRef(usize /* Reference */, Type);

/// Labels are used to designate branch targets in control flow operations.
///
/// Each label is a relative offset to the target branch first instruction.
pub struct Label(usize);

/// Every value in the intermediate representation is either a symbol reference
/// to a storage location or a literal value.
enum Value {
    StorageLocation(SymbolRef),
    ConstantLiteral(Literal),
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
    // Label pseudo instruction.
    Label,
    // Nop instruction.
    Nop,
}

enum Instruction {
    // `const` operation.
    Const(
        SymbolRef, /* Destination */
        Literal,   /* Literal value assigned to the destination */
    ),
    // Arithmetic operators.
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
    // Binary boolean operators.
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
    // Unary operators.
    Not(SymbolRef /* Destination */, Value /* LHS */),
    Neg(SymbolRef /* Destination */, Value /* LHS */),
    // Comparison operators.
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
    // Return statements.
    Return(Value /* Return value */),
    // Function calls.
    Call(SymbolRef /* Call Target */),
    // Direct jump to label.
    Jump(Label /* Offset */),
    // Condtional branches.
    Branch(
        SymbolRef, /* Condition */
        Label,     /* Then Target Offset */
        Label,     /* Else Target Offset */
    ),
    // Identity operator.
    Id(SymbolRef /* Destination symbol*/, Value),
    // Label pseudo instruction, acts as a data marker when generating code.
    Label(usize /* Label handle or offset */),
    // Nop instruction.
    Nop,
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
            Instruction::Label => OPCode::Label,
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Const(..) => write!(f, "const"),
            Instruction::Add(..) => write!(f, "add"),
            Instruction::Sub(..) => write!(f, "sub"),
            Instruction::Mul(..) => write!(f, "mul"),
            Instruction::Div(..) => write!(f, "div"),
            Instruction::And(..) => write!(f, "and"),
            Instruction::Or(..) => write!(f, "or"),
            Instruction::Neg(..) => write!(f, "neg"),
            Instruction::Not(..) => write!(f, "not"),
            Instruction::Eq(..) => write!(f, "eq"),
            Instruction::Neq(..) => write!(f, "neq"),
            Instruction::Lt(..) => write!(f, "lt"),
            Instruction::Lte(..) => write!(f, "lte"),
            Instruction::Gt(..) => write!(f, "gt"),
            Instruction::Gte(..) => write!(f, "gte"),
            Instruction::Return(..) => write!(f, "return"),
            Instruction::Call(..) => write!(f, "call"),
            Instruction::Jump(..) => write!(f, "jump"),
            Instruction::Branch(..) => write!(f, "branch"),
            Instruction::Id(..) => write!(f, "id"),
            Instruction::Nop => write!(f, "nop"),
            Instruction::Label(addr) => write!(f, "__LABEL_{addr}"),
        }
    }
}
