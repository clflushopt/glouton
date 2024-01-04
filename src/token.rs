//! Implementation of language tokens.
use std::fmt::{self};

/// Token represents the individual language tokens.
#[derive(Debug, Clone)]
pub enum Token {
    LParen,
    RParen,
    LBrace,
    RBrace,
    Equal,
    EqualEqual,
    IntLiteral(i32),
    CharLiteral(char),
    StringLiteral(String),
    BoolLiteral(bool),
    SemiColon,
    Return,
}

/// Implementing display trait for tokens.
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            &Self::LParen => write!(f, "LPAREN"),
            &Self::RParen => write!(f, "RPAREN"),
            &Self::LBrace => write!(f, "LBRACE"),
            &Self::RBrace => write!(f, "RBRACE"),
            &Self::Equal => write!(f, "ASSIGN"),
            &Self::EqualEqual => write!(f, "EQUAL"),
            &Self::SemiColon => write!(f, ";"),
            &Self::IntLiteral(value) => write!(f, "INT({value})"),
            &Self::CharLiteral(value) => write!(f, "CHAR({value})"),
            &Self::StringLiteral(value) => write!(f, "STR({value})"),
            &Self::BoolLiteral(value) => write!(f, "BOOL({value})"),
            &Self::Return => write!(f, "RETURN"),
        }
    }
}
