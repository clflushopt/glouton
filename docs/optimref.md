# Compiler Optimizations a Guide

This document aims to serve as a reference or wiki to popular compiler optimizations
my goal here is to outline the *what* of each optimization pass with examples, some
optimizations are implemented in Glouton but not all of them.

## Introduction

The goal of program optimization is to transform an input program `P` into an **equivalent**
program `P'` that is more optimal (size or performance wise or both).
It is important that the resulting *optimized* program be **equivalent** to the initial program
otherwise something went horribly wrong.

There is no algorithm to transform a program `P` into a more optimal equivalent program `P'` since
the process subsumes tasks that are known to be **undecidable** hence the famous FET theorem
(which you can find more about [here](https://en.wikipedia.org/wiki/Full-employment_theorem) but
efforts have been made in the last 60 years to build quite a respected set of optimizations that
can heuristically be used to improve program performance.

While this might seem discouraging, I would like to point that there is still a lot to be done
when it comes to compiler infrastructure. If LLVM has taught us anything for the past 25 years
it's that faster compilers are a net good but at the same time handwritten optimizations do not
have a clear payoff, for example a lot of the more complex optimizations that run after the basic
ones such as autovectorizations, loop unrolling, function inlining... often increase the amount
of generated code (the performance improvements are often the result of hardware improvements).

If you would like to know more about what the future may hold I suggest you read the following
[paper](https://arxiv.org/abs/1809.02161) by [Nuno Lopes](https://web.ist.utl.pt/nuno.lopes/)
and [John Regehr](https://blog.regehr.org/archives/1619).

