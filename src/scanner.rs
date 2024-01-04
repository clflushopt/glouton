//! Lexical analysis module responsible for building tokens out of source
//! code.
use std::str::Chars;

use crate::token::Token;

/// Scanner is responsible for building a list tokens out of a `String`.
///
/// Since the lifetime of non-tokenizable strings (such as literals) will
/// have to flow through the compiler's entire lifecycle we avoid holding
/// a reference to the individual literals and instead copy them.
pub struct Scanner<'a> {
    // Reference to the input string.
    input: &'a str,
    // Walking cursor used to walk and process tokens.
    cursor: usize,
    // Starting position of the token we are currently processing.
    start: usize,
    // Iterator over the input string.
    iter: Chars<'a>,
}

impl Scanner<'_> {
    /// Initialize a new scanner from a given input string reference.
    pub fn new<'a>(s: &'a str) -> Scanner<'a> {
        Scanner {
            input: s,
            cursor: 0,
            start: 0,
            iter: s.chars(),
        }
    }

    /// Scan the input string and return a list of tokens.
    pub fn scan(&mut self) -> Vec<Token> {
        let mut tokens = vec![];

        while let Some(ch) = self.next() {
            self.start = self.cursor - 1;
            match ch {
                '(' => tokens.push(Token::LParen),
                ')' => tokens.push(Token::RParen),
                '{' => tokens.push(Token::LBrace),
                '}' => tokens.push(Token::RBrace),
                '=' if self.consume(&'=') => tokens.push(Token::EqualEqual),
                ';' => tokens.push(Token::SemiColon),
                '=' => tokens.push(Token::Equal),
                'a'..='z' => tokens.push(Token::Return),
                '0'..='9' => tokens.push(Token::IntLiteral(0)),
                ' ' | '\n' | '\r' | '\t' => (),
                _ => todo!("Unsupported scan for {ch}"),
            }
        }

        tokens
    }

    /// Consume the current character and increment the cursor if the next
    /// character matches `expected`.
    fn consume(&mut self, expected: &char) -> bool {
        if self.eof()
            || self
                .iter
                .by_ref()
                .peekable()
                .peek()
                .is_some_and(|actual| actual.eq(&expected))
        {
            return false;
        }
        self.next();
        true
    }

    /// Returns the next character in the input if there is one remaining
    /// incrementing the cursor along the way, otherwise returns `None`.
    fn next(&mut self) -> Option<char> {
        if !self.eof() {
            self.cursor += 1;
            return self.iter.next();
        }
        None
    }

    /// Returns `true` if we reached the end of input.
    fn eof(&self) -> bool {
        self.cursor >= self.input.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::scanner::Scanner;

    #[test]
    fn can_scan_return_statements() {
        let input = r#"
            return 0;
        "#;
        let mut scanner = Scanner::new(input);
        let tokens = scanner.scan();
        println!("Tokens: {:?}", tokens);
    }

    #[test]
    fn can_scan_return_equals_statement() {
        let input = r#"
            return 0 == 0;
        "#;
        let mut scanner = Scanner::new(input);
        let tokens = scanner.scan();
        println!("Tokens: {:?}", tokens);
    }
}
