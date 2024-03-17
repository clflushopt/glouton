//! Global intermediate representation definition and implementation.
//!
//! Glouton Intermediate Representation is a linear SSA based intermediate
//! representation. The intermediate representation is based on Bril and
//! has three core instruction types, constant operations which produce
//! constant values, value operations which take operands and produce values
//! and effect based operations which take operands and produce no values.
use std::fmt;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Literal {
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
    fn from_operator(operator: &ast::BinaryOperator) -> ValueOp {
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

/// Instructions are the atomic operations of the linear IR part in GIR
/// they compose the building blocks of the graph based IR.
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
                        ValueOp::Call => write!(f, "@{arg}")?,
                        _ => write!(f, "{arg} ")?,
                    }
                }
                Ok(())
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

/// Instruction emitters are all owning functions, they take ownership
/// of their arguments to build an `Instruction`.
impl Instruction {
    /// Return a copy of the destination name of constant or value operations.
    fn dst(inst: &Instruction) -> Option<String> {
        match inst {
            Self::Constant { dst, .. } => Some(dst.clone()),
            Self::Value { dst, .. } => Some(dst.clone()),
            _ => None,
        }
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

    /// Emit a boolean operation.
    fn bool(dst: String, op_type: Type, op: ValueOp, args: Vec<String>) -> Instruction {
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
    // Program as a sequence of declared functions.
    program: Vec<Function>,
    // Global scope declarations.
    globals: Vec<Instruction>,
    // Variable counter used to create new variables.
    var_count: u32,
    // Index in the `program` of the current function we are building.
    curr: usize,
    // Current scope.
    scope: Scope,
    // Symbol table level we are at current, increments by 1 when we enter
    // a scope and decrements by 1 when we exit. This is used to set the
    // starting point when resolving symbols in the symbol table.
    level: usize,
    // Reference to the AST we are processing.
    ast: &'a ast::AST,
    // Symbol table built during semantic analysis phase.
    symbol_table: &'a SymbolTable,
}

impl<'a> IRGenerator<'a> {
    #[must_use]
    pub fn new(ast: &'a ast::AST, symbol_table: &'a sema::SymbolTable) -> Self {
        Self {
            program: vec![],
            globals: vec![],
            var_count: 0,
            curr: 0,
            level: 0,
            scope: Scope::Global,
            ast,
            symbol_table,
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
            Scope::Global => self.globals.push(inst),
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

    /// Returns a reference to the last pushed instruction.
    fn last(&self) -> Option<&Instruction> {
        self.program[self.curr].body.last()
    }

    pub fn gen(&mut self) {
        for decl in self.ast.declarations() {
            let (_, code) = self.visit_decl(decl);
            for inst in code {
                self.push(inst)
            }
        }
    }
}

impl<'a> ast::Visitor<(Option<String>, Vec<Instruction>)> for IRGenerator<'a> {
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
            _ => todo!("Unimplemented IR generation for {:?}", decl),
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
                let mut code = vec![];
                for stmt_ref in stmts {
                    let (_, mut block) = if let Some(stmt) = self.ast.get_stmt(*stmt_ref) {
                        self.visit_stmt(stmt)
                    } else {
                        unreachable!("Expected right handside to be a valid expression")
                    };

                    code.append(&mut block);
                }
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
            _ => todo!("Unimplemented visitor for Node {:?}", stmt),
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
                let inst = Instruction::arith(
                    dst.clone(),
                    Type::Int,
                    ValueOp::from_operator(&operator),
                    vec![
                        lhs.expect("Expected right handside to be in a temporary")
                            .clone(),
                        rhs.expect("Expected right handside to be in a temporary")
                            .clone(),
                    ],
                );
                code.push(inst);
                (Some(dst), code)
            }
            ast::Expr::Named(ref name) => {
                let t = match self.symbol_table.find(name, self.level) {
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
            _ => todo!("Unimplemented visitor for Node {:?}", expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::IRGenerator;
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

                let mut irgen = IRGenerator::new(parser.ast(), &symbol_table);
                irgen.gen();

                for inst in irgen.program() {
                    println!("{inst}");
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
}
