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

## Optimization Catalogue


#### Global Constant Pool Address Optimization

The goal of this optimization is to group all constant variables in a single contiguous array
in memory to make all loads be addressable via a tuple of `(ptr, offset)`.

For example consider the following code.

```c

const int a = 1;
const int b = 1;
const int c = 2;

int fn() {

  return a * b + c;
}

```

After the optimization pass it would look something like this.

```c

const int* __pool[3] = {1,1,2};

const int* __addr = &(__pool[0]);

int fn() {

  return (__addr * (__addr + 1) + (__addr + 2));
}

```

This kind of optimization hasn't been used in C-like languages in the last 20 years as far
as I know, the reason to **not do this** are that it's an optimization that is highly ABI
dependant and requires coordinations between both the compiler and the linker.

For runtime languages this kind of optimizations can be heavily used in combination with
some form of [interning](https://en.wikipedia.org/wiki/String_interning) to reduce memory
usage and the amount of code needed to generate fast loads and stores, Java for example
uses a constant pool.

#### Alias Optimizations

Alias optimizations come in different flavors (by address, const-qualifier, type...) but their
core idea is the same, if you have two pointers `(ptr p, ptr q)` that respect one of the following
rules :

- `(p, q)` point to different storage locations.
- `(p, q)` point to a const-qualified storage locations (immutable).
- `(p, q)` have different types.

Then the pointers cannot be aliased and thus can be optimized away in certain cases, consider
the examples below :

```c

int a[4] = {0x00, 0x01, 0xef, 0xff};
int b[4] = {0xff, 0xef, 0x01, 0x00};

void frump(int i, int j) { 

  int* _p = &a[i];
  int* _q = &b[j];

  int x = *(_q + 3);
  // We know that _p does not alias _q because they point to two
  // different storage locations, therefore this mutation does
  // not "change" q.
  *_p = 5;
  // We can drop this assignment since by deducing from the above x = y is still
  // valid.
  int y = *(_q + 3);
  // We can rewrite this as broom(x,x);
  return broom(x,y);
}

```

When it comes to alias optimizations, it is largely impossible to conduct with a prior
alias analysis pass (which is quite complex) a great starting source of information is
the LLVM [`AliasAnalysis`](https://llvm.org/docs/AliasAnalysis.html) documentation.

#### Direct Branch Elimination

This one is easy to explain, the pass consists of eliminating redundant direct branches that do
not depend on a condition.

A simple example would be something like this, the first `jmp .L1` is redundant because it only
trampolines you to another direct jump `jmp .L2` so it can be re-written as a single `jmp .L2`.

```

jmp .L1:

.L1:
  jmp .L2

```

Of course this is rarely a problem in languages where jumps are often within the same function
so they can optimized in an interprocedural manner, but in the case of languages that expose
something like `setjmp` and `longjmp` intraprocedural analysis is required to resolve the targets.

#### Loop Fusion

In some cases, where multi-dimensional arrays are amenable to analysis some nested loops can be
fused if they don't require any striding (i.e the array can be accessed in the same pattern).

One such example is if you have an array `arr[128][128]` and you want to zero initialize it, then
instead of a nested loop, you can simply get a pointer `p = &arr[0]` and increment it until it
reaches the end `p < 128 * 128`.

#### Instruction Combining

This is more of an IR targeted optimizations, where duplicate straight line operations can be
fused or combined into a single one. For example a sequential increment `INC i; INC i` can be
rewritten as `ADD i, 2`. 

Combining instructions can be done efficiently within the same basic block and can be merged
with other propagation-style passes to remove unused code and propagate or inline certain
operations.

#### Constant Folding

This is an easy one, essentially any operation that can be evaluated at compile time ends up
being evaluated and the operation replaced with the evaluation results.

```c

void shroump() {
  int a = 10;
  int b = 20;

  return a + b;
}

```

Becomes

```c

void shroump() {
  return 30;
}

```

