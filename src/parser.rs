//! The nice thing about this implementation is that it builds a purely flat
//! AST that uses arenas and handles to represent the tree.

use crate::ast::{BinaryOperator, Expr, ExprRef, Stmt, StmtRef, UnaryOperator, AST};
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
    // Logical not, unary minutes, pointer dereference
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
            let statement = self.statement();
            self.ast.push_stmt(statement);
        }
    }

    /// Parse a statement.
    fn statement(&mut self) -> Stmt {
        if self.eat(&Token::Return).is_some() {
            let stmt = self.return_stmt();
            return stmt;
        }

        self.expr_stmt()
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
            &Token::Minus => self.unary(),
            &Token::Bang => self.unary(),
            &Token::IntLiteral(value) => {
                let literal_expr = Expr::IntLiteral(value);
                self.ast.push_expr(literal_expr)
            }
            _ => todo!("Unexpected prefix token {}", self.prev()),
        };

        while prec <= self.tok_precedence(self.peek()) {
            let infix_ref = match self.consume() {
                &Token::Plus => self.binary(prefix_ref),
                &Token::Minus => self.binary(prefix_ref),
                &Token::Star => self.binary(prefix_ref),
                &Token::Slash => self.binary(prefix_ref),
                _ => todo!("Unexpected infix token {}", self.peek()),
            };

            prefix_ref = infix_ref;
        }

        prefix_ref
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
        let precedence = (self.tok_precedence(self.prev()) as u8 + 1).into();
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

    /// Parse a return statement.
    fn return_stmt(&mut self) -> Stmt {
        let expr_ref = self.expression();
        self.eat(&Token::SemiColon);
        Stmt::Return(expr_ref)
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
}
