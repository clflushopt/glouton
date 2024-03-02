use glouton::{parser, scanner, sema};

const MAIN_PROGRAM: &str = r#"
int main() {
    return 42;
}
"#;

fn main() {
    compile()
}

fn compile() {
    let tokens = scanner::Scanner::new(MAIN_PROGRAM).scan().unwrap();
    let mut parser = parser::Parser::new(&tokens);
    parser.parse();
    let _symbol_table = sema::analyze(parser.ast());
}
