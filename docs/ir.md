# Intermediate Representation

The intermediate representation used in Glouton is based on Bril and currently
uses the core subset, might extend to the memory subset once we start handling
pointers and such.

The core bril subset is a well formed intermediate representation with support
for extensions. The IR groups all instructions into three subsets depending on
what each instruction does. 

* Constant operations: operations that produce a literal value.
* Value operations: operations that produce a value.
* Effect operations: operations that produce side effects.

The entire core IR is described in the reference section but please visit 
the Bril [reference](https://capra.cs.cornell.edu/bril/lang/core.html) for more
details.

## Examples

The following are examples of the intermediate representation of some small
functions.

Some oddities you'll notice in the IR.

* Every value ends up in a storage location, for example literals are assigned
  to a local variable for e.g `v0` before they are used.

* Arguments and named expressions are stored via the `id` operation, this is a
  useful pattern to simplify the program.

* There is no notion of scopes anymore when generating programs, the entire
  function de-scoped and duplicate names are renamed similar to LLVM IR.


Let's consider what the "Hello World" of every compiler writer looks like.

```c

int main() {
  return 0;
}

```


```asm

@main: int {
   v0: int = const 0
   ret v0;
}

```

An empty function and another function with arguments.

```c

int main() {} 

int f(int a, int b) {
  return a + b;
}

```

```asm

@main: int {
}

@f(a: int, b: int): int {
   v0: int = id a
   v1: int = id b
   v2: int = add v0 v1
   ret v2;
}

```

A more complex example with all binary operators.

```c

int main() {
    int a = 1 + 1;
    int b = 2 - 2;
    int c = 3 * 3;
    int d = 4 / 4;
    bool e = a == b;
    bool f = b != c;
    bool g = c > d;
    bool h = d >= c;
    bool i = a < b;
    bool j = a <= b;
    return 0;
}

```

```asm

@main: int {
   v0: int = const 1
   v1: int = const 1
   v2: int = add v0 v1
   a: int = id v2
   v3: int = const 2
   v4: int = const 2
   v5: int = sub v3 v4
   b: int = id v5
   v6: int = const 3
   v7: int = const 3
   v8: int = mul v6 v7
   c: int = id v8
   v9: int = const 4
   v10: int = const 4
   v11: int = div v9 v10
   d: int = id v11
   v12: int = id a
   v13: int = id b
   v14: int = eq v12 v13
   e: bool = id v14
   v15: int = id b
   v16: int = id c
   v17: int = neq v15 v16
   f: bool = id v17
   v18: int = id c
   v19: int = id d
   v20: int = gt v18 v19
   g: bool = id v20
   v21: int = id c
   v22: int = id d
   v23: int = gte v21 v22
   h: bool = id v23
   v24: int = id d
   v25: int = id c
   v26: int = lt v24 v25
   i: bool = id v26
   v27: int = id d
   v28: int = id c
   v29: int = lte v27 v28
   j: bool = id v29
   v30: int = const 0
   ret v30;
}

```



## Reference

### Types

Core Bril defines two primitive types:

* `int`: 64-bit, two's complement, signed integers.
* `bool`: True or false.

### Arithmetic

These instructions are the obvious binary integer arithmetic operations.
They all take two arguments, which must be names of variables of type `int`
and produce a result of type `int`:

* `add`: x + y.
* `mul`: x × y.
* `sub`: x - y.
* `div`: x ÷ y.

In each case, overflow follows two's complement rules.
It is an error to `div` by zero.

### Comparison

These instructions compare integers.
They all take two arguments of type `int` and produce a result of type `bool`:

* `eq`: Equal.
* `lt`: Less than.
* `gt`: Greater than.
* `le`: Less than or equal to.
* `ge`: Greater than or equal to.

### Logic

These are the basic Boolean logic operators.
They take arguments of type `bool` and produce a result of type `bool`:

* `not` (1 argument)
* `and` (2 arguments)
* `or` (2 arguments)

### Control

Control operations unlike the value operations above, take labels and functions
in addition to normal arguments.

* `jmp`: Unconditional jump. One label: the label to jump to.
* `br`: Conditional branch. One argument: a variable of type `bool`. Two labels:
  a true label and a false label. Transfer control to one of the two labels
   depending on the value of the variable.
* `call`: Function invocation. Takes the name of the function to call and, as
  its arguments, the function parameters. The `call` instruction can be a Value
  Operation or an Effect Operation, depending on whether the function returns a
  value.
* `ret`: Function return. Stop executing the current activation record and return
  to the parent or exit the program if this is the top-level main activation
  record. It has one optional argument: the return value for the function.

Only `call` may (optionally) produce a result; the rest appear only as Effect
Operations.

### Miscellaneous

* `id`: A type-insensitive identity. Takes one argument, which is a variable of
   any type, and produces the same value.
* `print`: Output values to the console (with a newline). Takes any number of
   arguments of any type and does not produce a result.
* `nop`: Do nothing. Takes no arguments and produces no result.


