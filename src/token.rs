//! Implementation of language tokens.
use std::fmt::{self};

/// Language defined keywords.
pub const KEYWORDS: &[&str] = &[
    "int", "char", "bool", "return", "const", "void", "if", "else", "while", "for", "break",
    "true", "false",
];

/// Token represents the individual language tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    // Single character tokens.
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    SemiColon,
    Colon,
    Comma,
    // Operators.
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
    // Literal values.
    IntLiteral(i32),
    CharLiteral(char),
    StringLiteral(String),
    BoolLiteral(bool),
    // Identifiers.
    Identifier(String),
    // Keywords.
    Const,
    Void,
    If,
    Else,
    While,
    For,
    Break,
    True,
    False,
    Return,
    // Type declarations.
    Int,
    Char,
    Bool,
    // Eof token used to signal the end of file.
    Eof,
}

/// Implementing display trait for tokens.
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::LBrace => write!(f, "{{"),
            Self::RBrace => write!(f, "}}"),
            Self::LBracket => write!(f, "["),
            Self::RBracket => write!(f, "]"),
            Self::SemiColon => write!(f, ";"),
            Self::Colon => write!(f, ":"),
            Self::Comma => write!(f, ","),
            Self::Equal => write!(f, "ASSIGN"),
            Self::EqualEqual => write!(f, "EQUAL"),
            Self::BangEqual => write!(f, "NEQ"),
            Self::Greater => write!(f, "GT"),
            Self::GreaterEqual => write!(f, "GTE"),
            Self::Lesser => write!(f, "LT"),
            Self::LesserEqual => write!(f, "LTE"),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Slash => write!(f, "/"),
            Self::Star => write!(f, "*"),
            Self::And => write!(f, "AND"),
            Self::Or => write!(f, "OR"),
            Self::Bang => write!(f, "NOT"),
            // Literals are wrapped in `TYPE()` for readability.
            Self::IntLiteral(value) => write!(f, "INT({value})"),
            Self::CharLiteral(value) => write!(f, "CHAR({value})"),
            Self::StringLiteral(value) => write!(f, "STR({value})"),
            Self::BoolLiteral(value) => write!(f, "BOOL({value})"),
            // Identifiers are wrapped in `IDENT()` to signify that they
            // are identifiers.
            Self::Identifier(str) => write!(f, "IDENT({str})"),
            // Keywords are display in capital case.
            Self::Return => write!(f, "RETURN"),
            Self::Const => write!(f, "CONST"),
            Self::Void => write!(f, "VOID"),
            Self::If => write!(f, "IF"),
            Self::Else => write!(f, "ELSE"),
            Self::While => write!(f, "WHILE"),
            Self::For => write!(f, "FOR"),
            Self::Break => write!(f, "BREAK"),
            Self::True => write!(f, "TRUE"),
            Self::False => write!(f, "FALSE"),
            // Types supported are shown with a `_T` to signify that this is
            // a type.
            Self::Int => write!(f, "INT_T"),
            Self::Bool => write!(f, "BOOL_T"),
            Self::Char => write!(f, "CHAR_T"),
            // EOF is shown as `EOF` to signify end of token stream.
            Self::Eof => write!(f, "EOF"),
        }
    }
}
