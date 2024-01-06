//! Recursive descent parser that uses Pratt's approach for parsing expressions.
//!
//! The nice thing about this implementation is that it builds a purely flat
//! AST that uses arenas and handles to represent the tree.

use crate::token::Token;

/// Parser implements a recursive descent Pratt style parser.
pub struct Parser {
    // Input tokens to process.
    tokens: Vec<Token>,
    // Cursor in the tokens list.
    cursor: usize,
}

impl Parser {
    /// Returns a new `Parser` instance.
    pub fn new(tokens: &[Token]) -> Self {
        Self {
            tokens: tokens.to_owned(),
            cursor: 0usize,
        }
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
