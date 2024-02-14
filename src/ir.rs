//! Global intermediate representation definition and implementation.
//!
//! Glouton Intermediate Representation is a linear SSA based intermediate
//! representation. The intermediate representation is based on Bril and
//! has three core instruction types, constant operations which produce
//! constant values, value operations which take operands and produce values
//! and effect based operations which take operands and produce no values.
use core::fmt;

use crate::ast::{self, Visitor};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Literal {
    /// Integers
    Int(i32),
    /// Booleans.
    Bool(bool),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(value) => write!(f, "{value}"),
            Self::Bool(value) => write!(f, "{value}"),
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
    // No operation.
    Nop,
}

impl fmt::Display for EffectOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jump => write!(f, "jmp"),
            Self::Branch => write!(f, "br"),
            Self::Call => write!(f, "call"),
            Self::Return => write!(f, "ret"),
            Self::Nop => write!(f, "nop"),
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
    // Boolean operators.
    Not,
    And,
    Or,
    // Function calls that produce values.
    Call,
    // Identity operator.
    Id,
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
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Call => write!(f, "call"),
            Self::Id => write!(f, "id"),
        }
    }
}

/// Instruction are the atomic operations of the linear IR part in GIR
/// they compose the building blocks of basic blocks.
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
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant {
                dst,
                op,
                const_type,
                value,
            } => {
                write!(f, "{dst}: {const_type} = {op} {value};")
            }
            Self::Value {
                args,
                dst,
                op,
                op_type,
            } => {
                write!(f, "{dst}: {op_type} = {op}")?;
                for arg in args {
                    match op {
                        ValueOp::Call => write!(f, "@{arg}")?,
                        _ => write!(f, "{arg}")?,
                    }
                }
                write!(f, ";")
            }
            Self::Effect { args, op } => {
                write!(f, "{op} ")?;
                for arg in args {
                    match op {
                        EffectOp::Jump => write!(f, ".{arg}")?,
                        EffectOp::Call => write!(f, "@{arg}")?,
                        EffectOp::Branch => write!(f, ".{arg}")?,
                        EffectOp::Return => write!(f, "{arg}")?,
                        EffectOp::Nop => write!(f, ".")?,
                    }
                }
                write!(f, ";")
            }
        }
    }
}

impl Instruction {
    /// Instruction emitters are all owning functions, they take ownership
    /// of their arguments to build an `Instruction`.
    ///
    /// Emit a constant instruction.
    fn emit_const(dst: String, const_type: Type, value: Literal) -> Instruction {
        Instruction::Constant {
            dst,
            op: ConstOp::Const,
            const_type,
            value,
        }
    }

    /// Emit a jump instruction.
    fn emit_jmp(target: String) -> Instruction {
        Instruction::Effect {
            args: vec![target],
            op: EffectOp::Jump,
        }
    }

    /// Emit a branch instruction.
    fn emit_branch(cond: String, then_target: String, else_target: String) -> Instruction {
        Instruction::Effect {
            args: vec![cond, then_target, else_target],
            op: EffectOp::Branch,
        }
    }

    /// Emit a return instruction.
    fn emit_ret(value: String) -> Instruction {
        Instruction::Effect {
            args: vec![value],
            op: EffectOp::Return,
        }
    }

    /// Emith an arithmetic operation.
    fn emit_arith(dst: String, op_type: Type, op: ValueOp, args: Vec<String>) -> Instruction {
        Instruction::Value {
            args,
            dst,
            op,
            op_type,
        }
    }

    /// Emit a comparison operation.
    fn emit_cmp(dst: String, op_type: Type, op: ValueOp, args: Vec<String>) -> Instruction {
        Instruction::Value {
            args,
            dst,
            op,
            op_type,
        }
    }

    /// Emit a boolean operation.
    fn emit_bool(dst: String, op_type: Type, op: ValueOp, args: Vec<String>) -> Instruction {
        Instruction::Value {
            args,
            dst,
            op,
            op_type,
        }
    }

    /// Emit an identity operation.
    fn emit_ident(dst: String, op_type: Type, args: Vec<String>) -> Instruction {
        Instruction::Value {
            args,
            dst,
            op: ValueOp::Id,
            op_type,
        }
    }

    /// Emit a call operation.
    fn emit_call(dst: String, op_type: Type, name: String, mut args: Vec<String>) -> Instruction {
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

/// `Function` represents a function declaration in the AST, a `Function`
/// is composed as a linear sequence of GIR instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    // Function name.
    name: String,
    // List of arguments the function accepts.
    args: Vec<Argument>,
    // Body of the function as GIR instructions.
    body: Vec<Instruction>,
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
            writeln!(f, "{inst}")?;
        }
        write!(f, "}}")
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
}

/// Scope of the current AST node we are processing, this is an internal detail
/// of the `IRGenerator` and is used to decide where the current declaration
/// lives. This is mostly for global variable declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Scope {
    Global,
    Local,
}

/// `IRGenerator` generates an intermediate representation by walking the AST
/// and building a intermediate representation of the original program.
///
/// The result of the process is a `Vec` of IR generated functions and a global
/// scope used to hold global variable declarations.
pub struct IRGenerator<'a> {
    // Program as a sequence of functions.
    program: Vec<Function>,
    // Global scope declarations.
    glob: Vec<Instruction>,
    // Reference to the AST we are processing.
    ast: &'a ast::AST,
    // Variable counter used to create new variables.
    var_count: u32,
    // Index in the `program` of the current function we are building.
    curr: usize,
    // Current scope.
    scope: Scope,
}

impl<'a> IRGenerator<'a> {
    #[must_use]
    pub fn new(ast: &'a ast::AST) -> Self {
        Self {
            program: vec![],
            glob: vec![],
            ast,
            var_count: 0,
            curr: 0,
            scope: Scope::Global,
        }
    }

    /// Returns a non-mutable reference to the generated program.
    const fn program(&self) -> &Vec<Function> {
        &self.program
    }

    /// Returns a fresh variable name, used for intermediate literals.
    fn next_var(&mut self) -> String {
        let var = format!("v{}", self.var_count);
        self.var_count += 1;
        var
    }

    /// Push an instruction to the current scope.
    fn push(&mut self, inst: Instruction) {
        match self.scope {
            Scope::Local => self.program[self.curr].push(inst),
            // TODO: assert that the instruction is not illegal in the global
            // scope i.e (not a branch for e.x)
            Scope::Global => self.glob.push(inst),
        }
    }

    /// Switch to a local scope view.
    fn enter(&mut self, func: Function) {
        self.program.push(func);
        self.scope = Scope::Local
    }

    /// Exit back to the global scope.
    fn exit(&mut self) {
        self.scope = Scope::Global;
        self.curr += 1
    }

    /// Returns a reference to the last pushed instruction.
    fn last(&self) -> Option<&Instruction> {
        self.program[self.curr].body.last()
    }

    pub fn gen(&mut self) {
        ast::visit(self.ast, self)
    }
}

impl<'a> ast::Visitor<()> for IRGenerator<'a> {
    fn visit_stmt(&mut self, stmt: &ast::Stmt) {
        match stmt {
            // Function declarations are the only place where we need to switch
            // scopes explicitely, since the scope only affects where variable
            // declarations are positioned.
            ast::Stmt::FuncDecl {
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
                match self.ast.get_stmt(*body) {
                    Some(stmt) => self.visit_stmt(stmt),
                    _ => unreachable!("expected body reference to be valid !"),
                }
                // Exit back to the global scope.
                self.exit()
            }
            // Variable declaration are
            ast::Stmt::VarDecl {
                decl_type,
                name,
                value,
            } => {}
            // Blocks.
            ast::Stmt::Block(stmts) => {
                for stmt_ref in stmts {
                    if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                        self.visit_stmt(stmt)
                    }
                }
            }
            // Return statements.
            ast::Stmt::Return(expr_ref) => {
                if let Some(expr) = self.ast.get_expr(*expr_ref) {
                    self.visit_expr(expr)
                }
                let ret_name = match self.last() {
                    Some(Instruction::Value { dst, .. }) => dst.clone(),
                    Some(Instruction::Constant { dst, .. }) => dst.clone(),
                    _ => unreachable!("expected last instruction in return to be value or const"),
                };
                let ret = Instruction::emit_ret(ret_name);
                self.push(ret)
            }
            _ => todo!("Unimplemented visitor for Node {:?}", stmt),
        }
    }
    fn visit_expr(&mut self, expr: &ast::Expr) {
        match *expr {
            ast::Expr::IntLiteral(value) => {
                let dst = self.next_var();
                let inst = Instruction::emit_const(dst, Type::Int, Literal::Int(value));
                self.push(inst)
            }
            _ => todo!("Unimplemented visitor for Node {:?}", expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::IRGenerator;
    use crate::parser::Parser;
    use crate::scanner::Scanner;

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

                let mut irgen = IRGenerator::new(parser.ast());
                irgen.gen();

                for inst in irgen.program() {
                    println!("{inst}");
                }
            }
        };
    }

    test_ir_gen!(can_generate_const_ops, "int main() { return 0;}", &vec![]);
}
