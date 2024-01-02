# Glouton: an optimizing compiler playground for C0

Glouton is an optimizing compiler playground for the C0 language, it serves
as a reference for how to implement an optmizing compiler with a focus on
the middle and backend parts of a compiler.

Glouton exposes two different IRs, one that uses a Graph based IR to construct
a Control Flow Graph and one that uses a Sea of Nodes based IR. In both cases
the intermediate representations follow SSA form.

Currently the goal of Glouton is to implement all the optimizations in the paper
by Frances Allen (A Catalog of Optimizing Transformations).

## C0

C0 is a safe subset of C you can find out more in the [official website](https://c0.cs.cmu.edu/).

## Docs

I tried to document a lot of the features and design decisions for the frontend
and backends of the compiler with a specific focus on **how** to build the IR
and how to accomplish the code generation passses (instruction selection, scheduling
and register allocation...).

- [Frontend: Design of the scanner and parser](docs/frontend.md)
- [IR: Building a control flow graph from the AST](docs/cfg.md)
- [IR: Building a sea of nodes from the AST](docs/sea.md)
- [Optimizations: Transforms and Passes approach](docs/optimizations.md)
- [Optimizations: Catalogue of optimizations (in details)](docs/optimref.md)
- [Backend: The first steps towrads a code generator](docs/codegen.md)
- [Backend: Instruction selection, scheduling and register allocation](docs/backend.md)

