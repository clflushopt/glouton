use glouton::{ir, parser, scanner, sema};

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
    let symbol_table = sema::analyze(parser.ast());
    let mut irgen = ir::IRBuilder::new(parser.ast(), &symbol_table);
    irgen.build();
    for inst in irgen.functions() {
        println!("{inst}");
    }
}
