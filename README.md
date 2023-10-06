# Glouton

Glouton is an optmizing compiler for the C0 language a safe subset of C.

## C0

C0 is a safe subset of the C language designed at CMU for their freshman class
and [compilers course](https://www.cs.cmu.edu/afs/cs/academic/class/15411-f20/www/index.html).

You can find more information about C0 in the official [website](https://c0.cs.cmu.edu/).

## Usage

Glouton is written in Rust and you can build the compiler drive using `cargo build`.

```sh

$ cargo build --release

$ ./target/release/gloutonc file.c0 -o bi

```

The compiler driver will try and use the default `gcc` linker `ld`.

## Architecture

### Lexer and Parser

Both the lexer and parser are hand written, the lexer is a straight forward tokenizer
and the parser is an implementation of a top down precedence (Pratt) parser.

### SSA Construction

SSA construction uses the approach described in [Simple and Efficient Construction
of Static Single Assignment Form](http://individual.utoronto.ca/dfr/ece467/braun13.pdf).

### IR

The IR used is a custom design based on [Bril](https://www.cs.cornell.edu/courses/cs6120/2022sp/)

### Optimizations and Program Analysis

