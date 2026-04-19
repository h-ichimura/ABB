# Julia Best Practices for This Project

Reference: https://docs.julialang.org/en/v1/manual/style-guide/
           https://docs.julialang.org/en/v1/manual/performance-tips/

## Use existing functions, don't reinvent

- `optimize(f, lo, hi)` from Optim.jl for bounded 1D minimization (Brent's method)
- `optimize(f, x0, NelderMead())` for derivative-free multidimensional
- `optimize(f, g!, x0, LBFGS())` for gradient-based
- `quantile(v, p)` from Statistics
- `sort!` for in-place sorting
- `mul!(C, A, B)` from LinearAlgebra for in-place matrix multiply
- `dot(x, y)` from LinearAlgebra
- `norm(x)` from LinearAlgebra

## Performance

- Put all computation inside functions, not at top level
- Avoid untyped globals; use `const` or pass as arguments
- Pre-allocate arrays; avoid growing with push!/concatenation
- Access arrays in column-major order (first index fastest)
- Use `@views` for slices that don't need copies
- Use `@inbounds` in tight loops after verifying index safety
- Use `@.` for fused broadcast operations
- Use concrete types in struct fields (parametric if needed)

## Style

- Functions: lowercase, `!` suffix for mutating
- Types/modules: CamelCase
- 4-space indentation
- Don't parenthesize conditions: `if a == b` not `if (a == b)`
- Pass functions directly: `map(f, a)` not `map(x->f(x), a)`
- Avoid macros when functions suffice
- Keep functions short and focused
