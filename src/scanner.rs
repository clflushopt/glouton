//! Lexical analysis module responsible for building tokens out of source
//! code.
use crate::token::{Token, KEYWORDS};
use std::error::Error;
use std::fmt;

/// Scanner is responsible for building a list tokens out of a `String`.
///
/// Since the lifetime of non-tokenizable strings (such as literals) will
/// have to flow through the compiler's entire lifecycle we avoid holding
/// a reference to the individual literals and instead copy them.
pub struct Scanner<'a> {
    // Reference to the input string.
    input: &'a str,
    // Starting position of the token we are currently processing.
    start: usize,
    // Walking cursor used to walk and process tokens.
    cursor: usize,
    // Line in the input we're currently processing, incremented
    // on newlines.
    line: usize,
    // Vec of individual chars of the input.
    source: Vec<char>,
}

/// Scanner error type is used to report scanning errors to the user.
#[derive(Debug, Clone)]
pub struct ScannerError {
    details: String,
    line: usize,
}

impl ScannerError {
    const fn new(line: usize, details: String) -> Self {
        Self { details, line }
    }
}

impl fmt::Display for ScannerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} at line {}.", self.details, self.line)
    }
}

impl Error for ScannerError {
    fn description(&self) -> &str {
        &self.details
    }
}

impl Scanner<'_> {
    /// Creates a new lexer instance from a given `source` string.
    pub fn new<'a>(source: &'a str) -> Scanner<'a> {
        Scanner {
            input: source,
            start: 0,
            cursor: 0,
            line: 1,
            source: source.chars().collect(),
        }
    }

    /// Lex the passed source code and returns a list of tokens.
    /// # Errors
    /// Returns an error when it encounters an unknown token.
    pub fn scan(&mut self) -> Result<Vec<Token>, ScannerError> {
        let mut tokens = Vec::new();
        while let Some(ch) = self.next() {
            // Grab the first lexeme as we will need to used it to scan
            // multi-character tokens such as identifiers, numbers and strings
            self.start = self.cursor - 1;
            match ch {
                '&' if self.consume('&') => tokens.push(Token::And),
                '|' if self.consume('|') => tokens.push(Token::Or),
                '!' if self.consume('=') => tokens.push(Token::BangEqual),
                '=' if self.consume('=') => tokens.push(Token::EqualEqual),
                '>' if self.consume('=') => tokens.push(Token::GreaterEqual),
                '<' if self.consume('=') => tokens.push(Token::LesserEqual),
                '/' if self.consume('/') || self.consume('*') => self.comment(),
                '(' => tokens.push(Token::LParen),
                ')' => tokens.push(Token::RParen),
                '{' => tokens.push(Token::LBrace),
                '}' => tokens.push(Token::RBrace),
                '[' => tokens.push(Token::LBracket),
                ']' => tokens.push(Token::RBracket),
                ';' => tokens.push(Token::SemiColon),
                ':' => tokens.push(Token::Colon),
                ',' => tokens.push(Token::Comma),
                '<' => tokens.push(Token::Lesser),
                '>' => tokens.push(Token::Greater),
                '=' => tokens.push(Token::Equal),
                '!' => tokens.push(Token::Bang),
                '*' => tokens.push(Token::Star),
                '/' => tokens.push(Token::Slash),
                '+' => tokens.push(Token::Plus),
                '-' => tokens.push(Token::Minus),
                '"' => tokens.push(self.string()),
                '\'' => tokens.push(self.char()),
                '0'..='9' => tokens.push(self.integer()),
                '_' | 'a'..='z' | 'A'..='Z' => tokens.push(self.identifier()),
                // Do nothing on whitespace.
                ' ' | '\r' | '\t' => (),
                // Increment line number on newlines.
                '\n' => self.line += 1,
                _ => {
                    return Err(ScannerError::new(
                        self.line,
                        format!("Unrecognized token {ch}"),
                    ))
                }
            }
        }
        tokens.push(Token::Eof);
        Ok(tokens)
    }

    // Return next char and increment cursor position.
    fn next(&mut self) -> Option<char> {
        // Refactor to iterator style
        if !self.eof() {
            self.cursor += 1;
            return Some(self.source[self.cursor - 1]);
        }
        None
    }

    // Match current character, advancing the cursor if we match `expected`.
    fn consume(&mut self, expected: char) -> bool {
        if self.eof() || self.source[self.cursor] != expected {
            return false;
        }
        self.cursor += 1;
        true
    }

    // Consume comments.
    fn comment(&mut self) {
        while self.peek() != '\n' && !self.eof() {
            // Omit return value since we don't process comments
            self.next();
        }
    }

    // Peek next character without advancing the cursor
    fn peek(&self) -> char {
        if self.eof() {
            return '\0';
        }
        self.source[self.cursor]
    }

    // Scan integer literal.
    fn integer(&mut self) -> Token {
        while self.peek().is_ascii_digit() {
            self.next();
        }

        let int_literal = self.source[self.start..self.cursor]
            .iter()
            .collect::<String>();
        let value = int_literal.as_str().parse::<i32>().unwrap();
        Token::IntLiteral(value)
    }

    // Scan string literals enclosed in double quotes.
    fn string(&mut self) -> Token {
        while self.peek() != '"' && !self.eof() {
            // Handle multiline strings
            if self.peek() == '\n' {
                self.line += 1;
            }
            self.next();
        }
        // Consume closing quote
        self.next();
        // Trim surrounding quotes and build the string literal.
        let str_literal = self.source[self.start + 1..self.cursor - 1]
            .iter()
            .collect::<String>();
        Token::StringLiteral(str_literal)
    }
    // Scan literal characters enclosed in single quotes.
    fn char(&mut self) -> Token {
        while self.peek() != '\'' && !self.eof() {
            self.next();
        }
        // Consume closing quote.
        self.next();
        // Trim surrounding quotes and build the char literal.
        let char_literal = self.source[self.start + 1];
        Token::CharLiteral(char_literal)
    }

    // Scan identifiers.
    fn identifier(&mut self) -> Token {
        while self.peek().is_ascii_alphanumeric() {
            self.next();
        }

        let identifier = self.source[self.start..self.cursor]
            .iter()
            .collect::<String>();

        if KEYWORDS.contains(&identifier.as_str()) {
            return self.keyword(&identifier);
        }

        Token::Identifier(identifier)
    }

    /// Check if the given identifier is a keyword and return its equivalent
    /// token.
    fn keyword(&self, identifier: &str) -> Token {
        match identifier {
            "return" => Token::Return,
            "int" => Token::Int,
            _ => todo!("Unsupported identifier or token {identifier}"),
        }
    }

    // Check if we reached the end of the source.
    fn eof(&self) -> bool {
        self.cursor >= self.source.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::scanner::Scanner;
    use crate::token::Token;

    // Macro to generate test cases.
    macro_rules! test_scanner {
        ($name:ident, $source:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let source = $source;
                let mut scanner = Scanner::new(source);
                let tokens = scanner.scan().unwrap();
                assert_eq!(&tokens, $expected);
            }
        };
    }

    test_scanner!(
        can_scan_return_statements,
        "return 0;",
        &vec![
            Token::Return,
            Token::IntLiteral(0),
            Token::SemiColon,
            Token::Eof
        ]
    );

    test_scanner!(
        can_scan_return_statement_with_bool_expression,
        "return 42 == 24;",
        &vec![
            Token::Return,
            Token::IntLiteral(42),
            Token::EqualEqual,
            Token::IntLiteral(24),
            Token::SemiColon,
            Token::Eof
        ]
    );
}
