//! Implementation of language tokens.
use std::fmt::{self};

/// Language defined keywords.
pub const KEYWORDS: &'static [&'static str] = &["return"];

/// Token represents the individual language tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Equal,
    BangEqual,
    EqualEqual,
    GreaterEqual,
    LesserEqual,
    Greater,
    Lesser,
    Plus,
    Minus,
    Slash,
    Star,
    And,
    Or,
    Bang,
    IntLiteral(i32),
    CharLiteral(char),
    StringLiteral(String),
    BoolLiteral(bool),
    Identifier(String),
    SemiColon,
    Colon,
    Comma,
    Return,
    // Eof token used to signal the end of file.
    Eof,
}

/// Implementing display trait for tokens.
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            &Self::LParen => write!(f, "("),
            &Self::RParen => write!(f, ")"),
            &Self::LBrace => write!(f, "{{"),
            &Self::RBrace => write!(f, "}}"),
            &Self::LBracket => write!(f, "["),
            &Self::RBracket => write!(f, "]"),
            &Self::Equal => write!(f, "ASSIGN"),
            &Self::EqualEqual => write!(f, "EQUAL"),
            &Self::BangEqual => write!(f, "NEQ"),
            &Self::Greater => write!(f, "GT"),
            &Self::GreaterEqual => write!(f, "GTE"),
            &Self::Lesser => write!(f, "LT"),
            &Self::LesserEqual => write!(f, "LTE"),
            &Self::Plus => write!(f, "+"),
            &Self::Minus => write!(f, "-"),
            &Self::Slash => write!(f, "/"),
            &Self::Star => write!(f, "*"),
            &Self::And => write!(f, "AND"),
            &Self::Or => write!(f, "OR"),
            &Self::Bang => write!(f, "NOT"),
            &Self::SemiColon => write!(f, ";"),
            &Self::Colon => write!(f, ":"),
            &Self::Comma => write!(f, ","),
            &Self::IntLiteral(value) => write!(f, "INT({value})"),
            &Self::CharLiteral(value) => write!(f, "CHAR({value})"),
            &Self::StringLiteral(value) => write!(f, "STR({value})"),
            &Self::BoolLiteral(value) => write!(f, "BOOL({value})"),
            &Self::Identifier(str) => write!(f, "IDENT({str})"),
            &Self::Return => write!(f, "RETURN"),
            &Self::Eof => write!(f, "EOF"),
        }
    }
}
