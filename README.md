# Glouton: an optimizing compiler playground for a C-subset 

Glouton is an optimizing compiler playground for a subset of C, to be exact
it aims to compile (for now) the C0 subset defined below. The choice of C0
was made for the simple reason that the reference for C0 is small, clear
and well defined.

Obviously the frontend language is not important here as this project aims
to focus on compiler optimizations, runtime support, code generation, data
flow and control flow analysis...

This is a work in progress.

## C0

C0 is a safe subset of C you can find out more in the [official website](https://c0.cs.cmu.edu/).

## Docs

- [Frontend: Design of the scanner and parser](docs/frontend.md)
- [IR: Bril as an intermediate representation](docs/ir.md)
- [IR: Building a control flow graph](docs/cfg.md)
- [IR: Building a sea of nodes representation](docs/sea.md)
- [Optimizations: Transforms and Passes approach](docs/optimizations.md)
- [Optimizations: Catalogue of optimizations (in details)](docs/optimref.md)
- [Backend: The first steps towards a code generator](docs/codegen.md)
- [Backend: Instruction selection, scheduling and register allocation](docs/backend.md)

