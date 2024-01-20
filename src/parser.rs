//! The nice thing about this implementation is that it builds a purely flat
//! AST that uses arenas and handles to represent the tree.
use crate::ast::{BinaryOperator, DeclType, Expr, ExprRef, Stmt, UnaryOperator, AST};
use crate::token::Token;

/// Operator precedence tablet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Precedence {
    // Unspecified precedence (shouldn't exist in theory).
    None = 0,
    // Assignment is the lowest precedence level, since we assign to a variable
    // only after evaluating the entire rhs.
    Assignment = 1,
    // Logical OR (||) has lower precedence than Logical (AND).
    Or = 2,
    And = 3,
    // Equality and Inequality.
    Equal = 4,
    // Comparison operations.
    Comparison = 5,
    // Plus, minus.
    Term = 6,
    // Multiply, divide, modulo.
    Factor = 7,
    // Logical not, unary negation, pointer dereference
    // increment, decrement.
    Unary = 8,
    // Function calls, array subscript, structure field reference.
    Call = 9,
}

impl From<u8> for Precedence {
    fn from(prec: u8) -> Self {
        match prec {
            0 => Self::None,
            1 => Self::Assignment,
            2 => Self::Or,
            3 => Self::And,
            4 => Self::Equal,
            5 => Self::Comparison,
            6 => Self::Term,
            7 => Self::Factor,
            8 => Self::Unary,
            9 => Self::Call,
            _ => unreachable!("Unexpected `from({prec})` no matching variant for {prec}"),
        }
    }
}

impl From<Precedence> for u8 {
    fn from(prec: Precedence) -> Self {
        prec as u8
    }
}

/// Parser implements a recursive descent Pratt style parser.
pub struct Parser {
    // Input tokens to process.
    tokens: Vec<Token>,
    // Cursor in the tokens list.
    cursor: usize,
    /// Constructed AST>
    ast: AST,
}

impl Parser {
    /// Returns a new `Parser` instance by creating an owned version `tokens`.
    pub fn new(tokens: &[Token]) -> Self {
        Self {
            tokens: tokens.to_owned(),
            cursor: 0usize,
            ast: AST::new(),
        }
    }

    /// Return a reference to the constructed AST.
    pub fn ast(&self) -> &AST {
        &self.ast
    }

    /// Parse the input and construct an AST.
    pub fn parse(&mut self) {
        // Parse the statements that constitute our program.
        // FIXME: C0 programs are sequence of declarations (or definitions).
        //
        // The BNF is :
        // declaration ->
        //             | funcDecl
        //             | varDecl
        //             | statement;
        while !self.eof() {
            let decl = self.declaration();
            self.ast.push_stmt(decl);
        }
    }

    /// Parse a statement.
    fn statement(&mut self) -> Stmt {
        match self.peek() {
            &Token::Return => self.return_stmt(),
            &Token::LBrace => self.block(),
            _ => self.expr_stmt(),
        }
    }

    /// Parse a declaration.
    fn declaration(&mut self) -> Stmt {
        if self.peek() != &Token::Int && self.peek() != &Token::Char && self.peek() != &Token::Bool
        {
            return self.statement();
        }

        let decl_type = match self.consume() {
            &Token::Int => DeclType::Int,
            &Token::Char => DeclType::Char,
            &Token::Bool => DeclType::Bool,
            _ => unreachable!("Expected declaration type to be one of (int, char, bool)."),
        };
        let identifier = match self.consume() {
            Token::Identifier(ident) => ident.clone(),
            _ => unreachable!("Expected identifier, found {}", self.peek()),
        };

        match self.peek() {
            // Variable declaration without right value assignment.
            &Token::SemiColon => {
                self.eat(&Token::SemiColon);
                let assigned = decl_type.default_value();
                let assigned_ref = self.ast.push_expr(assigned);
                // Declaration without an assignment.
                return Stmt::VarDecl {
                    decl_type,
                    name: identifier,
                    value: assigned_ref,
                };
            }
            // Variable declaration with right value assignment.
            &Token::Equal => {
                let assigned = self.expression();
                self.eat(&Token::SemiColon);
                return Stmt::VarDecl {
                    decl_type,
                    name: identifier,
                    value: assigned,
                };
            }
            // Function declaration.
            &Token::LParen => {
                // Function declaration.
                self.eat(&Token::LParen);
                // Arguments
                self.eat(&Token::RParen);
                // Body
                self.eat(&Token::LBrace);
                // self.block()
                // End of body
                self.eat(&Token::RBrace);
                return Stmt::FuncDecl {
                    name: identifier,
                    return_type: decl_type,
                    args: vec![],
                    body: vec![],
                };
            }
            _ => {
                self.eat(&Token::SemiColon);
                self.statement()
            }
        }
    }

    /// Parse a block.
    fn block(&mut self) -> Stmt {
        Stmt::Block(vec![])
    }

    /// Parse an if statement.
    fn if_stmt(&mut self) -> Stmt {
        self.eat(&Token::If);
        // Opening parenthesis.
        self.eat(&Token::LParen);
        // Parse conditional expression.
        let conditional = self.expression();
        // Closing parenthesis.
        self.eat(&Token::RParen);
        // Body of the conditional branch, maybe some day we will support next
        // line statements. For now, expect a brace.
        self.eat(&Token::LBrace);
        // Conditional block.
        let body = self.block();
        let body_ref = self.ast.push_stmt(body);
        // Optional part.
        Stmt::If(conditional, body_ref, None)
    }

    /// Parse a return statement.
    fn return_stmt(&mut self) -> Stmt {
        self.eat(&Token::Return);
        let expr_ref = self.expression();
        self.eat(&Token::SemiColon);
        Stmt::Return(expr_ref)
    }

    /// Parse an expression.
    fn expression(&mut self) -> ExprRef {
        self.precedence(Precedence::Assignment)
    }

    /// Parse an expression by its precedence level.
    fn precedence(&mut self, prec: Precedence) -> ExprRef {
        // Prefix part.
        let mut prefix_ref = match self.consume() {
            &Token::LParen => self.grouping(),
            &Token::Minus | &Token::Bang => self.unary(),
            &Token::IntLiteral(value) => {
                let literal_expr = Expr::IntLiteral(value);
                self.ast.push_expr(literal_expr)
            }
            &Token::Equal => self.assignment(),
            Token::Identifier(_) => self.named(),
            _ => todo!("Unexpected prefix token {}", self.prev()),
        };

        // println!("Infix expression: {:?}", self.ast.get_expr(prefix_ref));
        // println!("Precedence : {:?}", prec);

        while prec < self.tok_precedence(self.peek()) {
            let infix_ref = match self.consume() {
                // Arithmetic expressions.
                &Token::Plus | &Token::Minus | &Token::Star | &Token::Slash => {
                    self.binary(prefix_ref)
                }
                // Comparison expressions.
                &Token::EqualEqual
                | &Token::BangEqual
                | &Token::GreaterEqual
                | &Token::Greater
                | &Token::LesserEqual
                | &Token::Lesser => self.comparison(prefix_ref),
                // Call expressions.
                &Token::LParen => self.call(prefix_ref),
                _ => todo!("Unexpected infix token {}", self.peek()),
            };

            // println!("Infix expression : {:?}", self.ast.get_expr(infix_ref));
            prefix_ref = infix_ref;
        }

        prefix_ref
    }

    /// Parse a comparison expression.
    fn comparison(&mut self, left: ExprRef) -> ExprRef {
        let operator = match self.prev() {
            &Token::Lesser => BinaryOperator::Lt,
            &Token::LesserEqual => BinaryOperator::Lte,
            &Token::Greater => BinaryOperator::Gt,
            &Token::GreaterEqual => BinaryOperator::Gte,
            &Token::EqualEqual => BinaryOperator::Eq,
            &Token::BangEqual => BinaryOperator::Neq,
            // There is no infix operator.
            _ => unreachable!("Unknown token in binary expression {}", self.peek()),
        };

        let precedence = (self.tok_precedence(self.prev()) as u8 + 1).into();
        let right = self.precedence(precedence);
        self.ast.push_expr(Expr::BinOp {
            left,
            operator,
            right,
        })
    }

    /// Parse a binary expression.
    fn binary(&mut self, left: ExprRef) -> ExprRef {
        let operator = match self.prev() {
            &Token::Plus => BinaryOperator::Add,
            &Token::Minus => BinaryOperator::Sub,
            &Token::Star => BinaryOperator::Mul,
            &Token::Slash => BinaryOperator::Div,
            // There is no infix operator.
            _ => unreachable!("Unknown token in binary expression {}", self.peek()),
        };
        let precedence = self.tok_precedence(self.prev()).into();
        let right = self.precedence(precedence);
        self.ast.push_expr(Expr::BinOp {
            left,
            operator,
            right,
        })
    }

    /// Parse a grouping expression.
    fn grouping(&mut self) -> ExprRef {
        // Parse the grouped expression (inside the parenthesis).
        let expr_ref = self.expression();
        // Consume the closing parenthesis.
        self.eat(&Token::RParen);
        // Push the expression to pool and return a ref to it.
        self.ast.push_expr(Expr::Grouping(expr_ref))
    }

    /// Parse a unary expression.
    fn unary(&mut self) -> ExprRef {
        // Grab the operator.
        let operator = match self.prev() {
            &Token::Minus => UnaryOperator::Neg,
            &Token::Bang => UnaryOperator::Not,
            _ => unreachable!("Unexpected unary operator {}", self.prev()),
        };

        // Parse the operand.
        let operand = self.precedence(Precedence::Unary);
        // Push the grouping expression to the pool.
        self.ast.push_expr(Expr::UnaryOp { operator, operand })
    }

    /// Parse an assignment expression.
    fn assignment(&mut self) -> ExprRef {
        // Parse the right hand side expression.
        let expr_ref = self.precedence(Precedence::None);
        self.eat(&Token::SemiColon);
        expr_ref
    }

    /// Parse a named expression such as "x".
    fn named(&mut self) -> ExprRef {
        // Consume the token and build a named expr.
        match self.prev() {
            Token::Identifier(ident) => self.ast.push_expr(Expr::Named(ident.to_string())),
            _ => unreachable!(
                "Expected identifier in named expression got {}",
                self.prev()
            ),
        }
    }

    /// Parse a call expression such as "f(a,b,c)", `name` is parsed
    /// as a prefix expression and represents the function name since `CallExpr`
    /// is considered infix.
    fn call(&mut self, name: ExprRef) -> ExprRef {
        let mut args = vec![];

        if !self.match_next(&Token::RParen) {
            while !self.match_next(&Token::RParen) {
                args.push(self.expression());

                if !self.match_next(&Token::Comma) {
                    break;
                }
            }
        }

        self.eat(&Token::RParen);

        self.ast.push_expr(Expr::Call { name, args })
    }

    fn match_next(&mut self, expected: &Token) -> bool {
        if self.peek() != expected {
            return false;
        }
        self.consume();
        true
    }

    /// Parse an expression statement.
    fn expr_stmt(&mut self) -> Stmt {
        let expr_ref = self.expression();
        self.eat(&Token::SemiColon);
        // Push the expression to the AST.
        Stmt::Expr(expr_ref)
    }

    /// Return a token's precedence.
    fn tok_precedence(&self, token: &Token) -> Precedence {
        match token {
            &Token::Minus => Precedence::Term,
            &Token::Plus => Precedence::Term,
            &Token::Slash => Precedence::Factor,
            &Token::Star => Precedence::Factor,
            &Token::Or => Precedence::Or,
            &Token::And => Precedence::And,
            &Token::EqualEqual => Precedence::Equal,
            &Token::BangEqual => Precedence::Equal,
            &Token::Greater => Precedence::Comparison,
            &Token::GreaterEqual => Precedence::Comparison,
            &Token::Lesser => Precedence::Comparison,
            &Token::LesserEqual => Precedence::Comparison,
            // Assignment.
            &Token::Equal => Precedence::Assignment,
            // Call.
            &Token::LParen => Precedence::Call,
            _ => Precedence::None,
        }
    }

    /// Match the current token against the given token, if they match
    /// consume the token and return it. Otherwise returns `None`.
    fn eat(&mut self, token: &Token) -> Option<&Token> {
        if self.check(token) {
            return Some(self.consume());
        }
        None
    }

    /// Check if the current token is the expected one.
    fn check(&self, token: &Token) -> bool {
        if self.eof() {
            return false;
        }
        self.peek() == token
    }

    /// Consume the current token and return it.
    fn consume(&mut self) -> &Token {
        if !self.eof() {
            self.cursor += 1;
        }
        self.prev()
    }

    /// Peek and return a reference to the next token without moving the cursor
    /// position.
    fn peek(&self) -> &Token {
        &self.tokens[self.cursor]
    }

    /// Return the previously consumed token.
    fn prev(&self) -> &Token {
        &self.tokens[self.cursor - 1]
    }

    /// Returns true if the next token is `Token::Eof`.
    fn eof(&self) -> bool {
        self.tokens[self.cursor] == Token::Eof
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::Parser;
    use crate::scanner::Scanner;

    // Macro to generate test cases.
    macro_rules! test_parser {
        ($name:ident, $source:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let source = $source;
                let mut scanner = Scanner::new(source);
                let tokens = scanner.scan().unwrap();
                let mut parser = Parser::new(&tokens);
                parser.parse();
                assert_eq!(parser.ast().to_string(), $expected);
            }
        };
    }
    test_parser!(can_parse_return_statements, "return 0;", "Return(0)");
    /*
     * Imaginary AST for this expression
     * Grouping(
     *   Add(
     *       Add(
     *           Add(1,2),
     *           3,
     *       ),
     *       Grouping(
     *           Mul(
     *               4,
     *               5,
     *           )
     *       )
     *   )
     *)
     *
     * */
    test_parser!(
        can_parse_grouping_expression,
        "(1 + 2 + 3 + (4 * 5));",
        "Expr(Grouping(Add(Add(Add(1, 2), 3), Grouping(Mul(4, 5)))))"
    );
    /*
     * Imaginary AST for this expression
     * Add(
     *   Sub(
     *       Mul(
     *           Div(6,3),
     *           4
     *       ),
     *       1
     *   ),
     *   2
     *)
     */
    test_parser!(
        can_parse_non_grouped_expression,
        "6 / 3 * 4 - 1 + 2",
        "Expr(Add(Sub(Mul(Div(6, 3), 4), 1), 2))"
    );

    test_parser!(
        can_parse_variable_declaration,
        "int a = 0;",
        "VAR(INT_TYPE, a, 0)"
    );

    test_parser!(
        can_parse_assignment_with_prefix,
        "int x = !a;",
        "VAR(INT_TYPE, x, Not(Named(a)))"
    );

    test_parser!(
        can_parse_assignment_with_prefix_operator,
        "int x = -5;",
        "VAR(INT_TYPE, x, Neg(5))"
    );

    test_parser!(
        can_parse_non_assigned_declaration,
        "int x;",
        "VAR(INT_TYPE, x, 0)"
    );

    test_parser!(
        can_parse_named_assignment,
        "int x = y;",
        "VAR(INT_TYPE, x, Named(y))"
    );

    test_parser!(
        can_parse_variable_declaration_with_expression,
        "int b = (5 * 3 / 1 + 4 - 2);",
        "VAR(INT_TYPE, b, Grouping(Sub(Add(Div(Mul(5, 3), 1), 4), 2)))"
    );

    test_parser!(
        can_parse_call_expression_with_arguments,
        "int x = f(a,b,c);",
        "VAR(INT_TYPE, x, Call(Named(f), Args(Named(a), Named(b), Named(c))))"
    );

    test_parser!(
        can_parse_call_expression_with_no_arguments,
        "int z = foo()",
        "VAR(INT_TYPE, z, Call(Named(foo), Args()))"
    );

    test_parser!(
        can_parse_unary_call_expression,
        "int z = !foo()",
        "VAR(INT_TYPE, z, Not(Call(Named(foo), Args())))"
    );

    test_parser!(
        can_parse_greater_than_equal_expression,
        "5 + 3 * 10 >= 2",
        "Expr(GreaterEqual(Add(5, Mul(3, 10)), 2))"
    );

    test_parser!(
        can_parse_equality_expression,
        "5 + 7 * 10 / 2 == 0",
        "Expr(Equal(Add(5, Div(Mul(7, 10), 2)), 0))"
    );

    test_parser!(
        can_parse_call_with_equality_expression,
        "1 + 1 == f(a,b);",
        "Expr(Equal(Add(1, 1), Call(Named(f), Args(Named(a), Named(b)))))"
    );

    test_parser!(
        can_parse_inequality_with_calls,
        "g(a,b) != f(a,b);",
        "Expr(NotEqual(Call(Named(g), Args(Named(a), Named(b))), Call(Named(f), Args(Named(a), Named(b)))))"
    );
}
