//! Recursive descent parser that uses Pratt's approach for parsing expressions.
//!
//! The nice thing about this implementation is that it builds a purely flat
//! AST that uses arenas and handles to represent the tree.

use crate::ast::{Expr, ExprRef, Stmt, StmtRef, AST};
use crate::token::Token;

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
    fn expression(&mut self) -> Expr {
        match self.consume() {
            Token::IntLiteral(value) => {
                return Expr::IntLiteral(*value);
            }
            _ => todo!("Unsupprted expression"),
        }
    }

    /// Parse a return statement.
    fn return_stmt(&mut self) -> Stmt {
        let expr = self.expression();
        self.eat(&Token::SemiColon);
        // Push the expression to the AST.
        let expr_ref = self.ast.push_expr(expr);

        Stmt::Return(expr_ref)
    }

    /// Parse an expression statement.
    fn expr_stmt(&mut self) -> Stmt {
        let expr = self.expression();
        self.eat(&Token::SemiColon);
        // Push the expression to the AST.
        let expr_ref = self.ast.push_expr(expr);

        Stmt::Expr(expr_ref)
    }

    /// Match the current token against the given token, if they match
    /// consume the token and return it. Otherwise returns `None`.
    fn eat(&mut self, token: &Token) -> Option<&Token> {
        if self.check(token) {
            println!("Consuming token: {token}");
            return Some(self.consume());
        }
        println!("Token {token} doesn't match {}", self.peek());
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
                println!("Tokens: {:?}", tokens);
                let mut parser = Parser::new(&tokens);
                parser.parse();
                println!("Got AST: {:?}", parser.ast());
            }
        };
    }
    test_parser!(can_parse_return_statements, "return 0;", &vec![]);
}
