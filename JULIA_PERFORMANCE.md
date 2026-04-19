---
name: Julia performance best practices
description: Comprehensive Julia performance optimization reference - consult before writing any performance-critical Julia code
type: reference
originSessionId: efc12abc-2bce-4d12-b194-cc92a1046541
---
Reference: https://docs.julialang.org/en/v1/manual/performance-tips/
Also: https://www.matecdev.com/posts/julia-performance-checklist.html

## Critical Rules (always follow)

1. **All computation inside functions** — never at top level
2. **No untyped globals** — pass as arguments or use `const`
3. **Type-stable functions** — use `zero(x)` not `0`, consistent return types
4. **Concrete types in struct fields** — use `struct MyType{T<:AbstractFloat}; a::T; end`
5. **Pre-allocate arrays** — reuse buffers, use `!` convention for in-place
6. **Use `@views` for slices** — avoid unnecessary copies
7. **Column-major access** — inner loop varies first index
8. **Fuse broadcasts** — use `@.` to avoid temporary arrays

## Performance Annotations (use in hot loops)

- `@inbounds` — skip bounds checking (verify indices are valid first)
- `@simd` — enable SIMD vectorization (iterations must be independent)
- `@fastmath` — aggressive float optimizations (may change results)

## Common Patterns

### Pre-allocation
```julia
# Bad: allocates every call
f(x) = [x + i for i in 1:N]
# Good: mutate pre-allocated buffer
function f!(out, x)
    for i in 1:N; out[i] = x + i; end
end
```

### Views vs copies
```julia
# Bad: copies data
sum(x[2:end-1])
# Good: no copy
@views sum(x[2:end-1])
```

### Function barriers
```julia
# Separate type-uncertain setup from hot inner loop
function outer(data)
    typed_data = process(data)  # type resolved here
    inner!(result, typed_data)  # inner function specialized
end
```

### StaticArrays for small fixed-size
```julia
using StaticArrays
SA[x1, x2, x3]  # stack-allocated, no heap allocation (35× speedup)
```

### Avoid allocations in closures
```julia
# Bad: r may be boxed
f = x -> x * r
# Good: let binding prevents boxing
f = let r = r; x -> x * r; end
```

## Profiling

- `@time` — basic timing (run twice: first includes compilation)
- `@btime` from BenchmarkTools — accurate microbenchmarks
- `@code_warntype` — check type stability (red = bad)
- `@allocated` — count bytes allocated
- `--track-allocation=user` — line-by-line allocation tracking

## Numerical Computing Specifics

- `abs2(z)` instead of `abs(z)^2`
- `div(x,y)` instead of `trunc(x/y)`
- `mul!(C, A, B)` for in-place matrix multiply
- `dot(x, y)` from LinearAlgebra
- `ldiv!` for in-place linear solve
- Avoid `inv(A)*b` — use `A\b` instead
