# Frontend: Design of the scanner and parser

The scanner in `glouton` is a very simple LL1 lexical analyzer that iterates
over the input and builds a list of tokens. Since we are expecting most inputs
to be fairly small (10 ~ 1000 LOC) no attention was paid to its design.

The parser in `glouton` on the other hand is slightly different, it uses Pratt
style approach to parsing expressions and uses a flattened AST representation.

The flat AST representation comes in three layers of `Vec`s each layer is has
references to objects in the following layer or at the same layer.

```

decls: Vec<Stmt>
stmts: Vec<Stmt>
exprs: Vec<Expr>

```

The first layer `decls` is used to store top level declarations in the program
since our input language is C-like these are variables and functions.

The second layer `stmts` is a pool of all stmts in the program, whereas `decls`
holds `enum::Stmt::VarDecl | enum::Stmt::FuncDecl` with references in the pool
of`stmts` objects in the `stmts` pool can hold backwards references.

The last layer `exprs` is a pool of all expressions in the program, where all
references flow backwards. For example `x + y` would be laid out in the `exprs`
pool as such :

```

x_ref, y_ref, bin_op{add,x_ref, y_ref}

```

Walking the AST must start from the declarations slice and can recurisvely
visit the statements and expressions.

Consider for example this block of code :

```c

int main() {
    int i = 0;
    int x = i;
    int z = x + i;
}

```

This AST would be laid out as such :

```

decls: [FuncDecl(name:str, ret:type, args:Vec<StmtRef>, body: StmtRef[0]]
stmts: [VarDecl(i, 0), VarDecl(x, 0), VarDecl(z, ExprRef[2])]
exprs: [Named(x), Named(y), Add(ExprRef[0], ExprRef[1])]

```


This data oriented approach has several pros, the first being that arenas
are borrow checker friendly. This is especially important in Rust where
an initial design might be invalidated because it doesn't have a borrow
checker friendly representation.

The second argument for this is that complex self references and circular
references are not an issue anymore because there is no lifetime associated
with the reference you use, the lifetime is now associated with the entire
arena and the individual entries hold indices into the arena which are just
`usize`.

Another argument although less impressive at this scale is speed, because
fetches aren't done via pointers and because AST nodes have nice locality
if your arena fits in cache then walking it becomes much faster than going
through the pointer fetch road. There is also no allocations in the hot path.

One downside of this representation is that it requires being careful when
implementing consumers of the AST. This comes from the fact that the AST
is constructed bottom-up and so all child leafs will be located before
their parents in the flat representation (at least at the second two layers).

