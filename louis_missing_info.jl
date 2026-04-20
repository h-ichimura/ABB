#=
louis_missing_info.jl — Compute missing information I_{X|Y} = Var[∂S_spline/∂θ | Y]

The key insight: the complete-data score decomposes as
  S(η; θ) = ∂S_eval(η; θ)/∂θ − ∂log C(θ)/∂θ

The second term is constant in η, so Var[S|Y] = Var[∂S_eval/∂θ | Y].
We only need the variance of the SPLINE EVALUATION derivatives,
which are much smaller than the full score and don't suffer from
cancellation with the normalizing constant.
=#

if !@isdefined(CSplineWorkspace)
    include("cspline_abb.jl")
end
using Printf, LinearAlgebra, Statistics

function compute_missing_info(v::Vector{Float64}, y::Matrix{Float64},
                               K::Int, σy::Float64, τ::Vector{Float64}; G::Int=201)
    a_Q, M_Q, a_init, M_init, a_eps1, a_eps3, M_eps = unpack_profiled(v, K)
    N, T_obs = size(y)
    nk = K + 1; np = length(v)
    ws = CSplineWorkspace(G, K)
    G = ws.G_base

    idx_Q = 1:9; idx_init = 10:12; idx_eps = 13:14

    # Build transition matrix
    cspline_transition_matrix!(ws.T_mat, ws.grid, G, a_Q, M_Q, K, σy, τ,
                               ws.hv_buf, ws.t_buf, ws.s_buf, ws.masses_buf, ws.c1buf)

    # ---- Spline EVALUATION functions (no log C) ----
    function S_init_at(η, vv)
        _,_,ai,Mi,_,_,_ = unpack_profiled(vv, K)
        si=zeros(3); bL=Ref(0.0); bR=Ref(0.0); k1=Ref(0.0); k3=Ref(0.0)
        solve_cspline_c2!(si,bL,bR,k1,k3,ai,τ,Mi)
        cspline_eval(η, ai, si, bL[], bR[], k1[], k3[])
    end

    function S_eps_at(x, vv)
        _,_,_,_,ae1,ae3,Me = unpack_profiled(vv, K)
        ae=[ae1,0.0,ae3]; se=zeros(3); bL=Ref(0.0); bR=Ref(0.0); k1=Ref(0.0); k3=Ref(0.0)
        solve_cspline_c2!(se,bL,bR,k1,k3,ae,τ,Me)
        cspline_eval(x, ae, se, bL[], bR[], k1[], k3[])
    end

    function S_trans_at(η_t, η_lag, vv)
        aQ,MQ,_,_,_,_,_ = unpack_profiled(vv, K)
        z=η_lag/σy; hv=zeros(nk); hv[1]=1.0; K>=1&&(hv[2]=z)
        for k in 2:K; hv[k+1]=z*hv[k]-(k-1)*hv[k-1]; end
        t_loc=[dot(view(aQ,:,l),hv) for l in 1:3]
        (t_loc[2]<=t_loc[1]||t_loc[3]<=t_loc[2]) && return 0.0
        st=zeros(3); bL=Ref(0.0); bR=Ref(0.0); k1=Ref(0.0); k3=Ref(0.0)
        solve_cspline_c2!(st,bL,bR,k1,k3,t_loc,τ,MQ)
        cspline_eval(η_t, t_loc, st, bL[], bR[], k1[], k3[])
    end

    # FD of spline eval w.r.t. block params
    function dS_eval(eval_fn, args, block_idx)
        nb = length(block_idx)
        sc = zeros(nb)
        vt = copy(v)
        for (jl, jg) in enumerate(block_idx)
            hj = max(1e-5, 1e-4 * abs(v[jg]))
            vt[jg] = v[jg]+hj; fp = eval_fn(args..., vt)
            vt[jg] = v[jg]-hj; fm = eval_fn(args..., vt)
            vt[jg] = v[jg]; sc[jl] = (fp-fm)/(2hj)
        end
        sc
    end

    # Log-density (with log C) for forward-backward
    function log_f_init(η, vv)
        _,_,ai,Mi,_,_,_ = unpack_profiled(vv, K)
        si=zeros(3); bL=Ref(0.0); bR=Ref(0.0); k1=Ref(0.0); k3=Ref(0.0)
        solve_cspline_c2!(si,bL,bR,k1,k3,ai,τ,Mi)
        lr=max(si[1],si[2],si[3]); mi=zeros(4)
        cspline_masses!(mi,ai,si,bL[],bR[],k1[],k3[],lr)
        Ci=sum(mi); Ci<1e-300 && return -1e10
        cspline_eval(η,ai,si,bL[],bR[],k1[],k3[])-log(Ci)-lr
    end

    function log_f_eps(x, vv)
        _,_,_,_,ae1,ae3,Me = unpack_profiled(vv, K)
        ae=[ae1,0.0,ae3]; se=zeros(3); bL=Ref(0.0); bR=Ref(0.0); k1=Ref(0.0); k3=Ref(0.0)
        solve_cspline_c2!(se,bL,bR,k1,k3,ae,τ,Me)
        lr=max(se[1],se[2],se[3]); me=zeros(4)
        cspline_masses!(me,ae,se,bL[],bR[],k1[],k3[],lr)
        Ce=sum(me); Ce<1e-300 && return -1e10
        cspline_eval(x,ae,se,bL[],bR[],k1[],k3[])-log(Ce)-lr
    end

    # ---- Precompute ----
    @printf("  Precomputing...\n"); flush(stdout)

    # f_eps for forward-backward
    log_fe = zeros(G, N, T_obs)
    @inbounds for t in 1:T_obs, i in 1:N, g in 1:G
        log_fe[g,i,t] = log_f_eps(y[i,t]-ws.grid[g], v)
    end
    f_init_g = [exp(log_f_init(ws.grid[g], v)) for g in 1:G]

    # ∂S_init/∂θ_init at each grid point (no log C)
    dS_init = zeros(G, 3)
    for g in 1:G
        dS_init[g,:] = dS_eval(S_init_at, (ws.grid[g],), idx_init)
    end

    # ∂S_trans/∂θ_Q at each (g1,g2)
    dS_trans = zeros(G, G, 9)
    @printf("  dS_trans %d×%d...", G, G); flush(stdout)
    for g1 in 1:G, g2 in 1:G
        dS_trans[g1,g2,:] = dS_eval(S_trans_at, (ws.grid[g2], ws.grid[g1]), idx_Q)
    end
    @printf(" done.\n"); flush(stdout)

    # ---- Forward-backward + Var[∂S/∂θ | Y] ----
    sw = ws.sw[1:G]
    T_mat = view(ws.T_mat, 1:G, 1:G)
    α = zeros(G, T_obs); β_bw = zeros(G, T_obs); γ = zeros(G, T_obs)

    # Accumulate Var[∂S_spline/∂θ | Y_i] over individuals
    Var_total = zeros(np, np)  # Σ_i Var[∂S_spline/∂θ | Y_i]

    @printf("  Processing %d individuals...\n", N); flush(stdout)

    for i in 1:N
        # Forward
        @inbounds for g in 1:G; α[g,1] = f_init_g[g]*exp(log_fe[g,i,1]); end
        L=dot(view(α,:,1),sw); α[:,1]./=L
        for t in 2:T_obs
            pw=α[:,t-1].*sw; p_pred=transpose(T_mat)*pw
            @inbounds for g in 1:G; α[g,t]=p_pred[g]*exp(log_fe[g,i,t]); end
            L=dot(view(α,:,t),sw); α[:,t]./=L
        end

        # Backward
        β_bw[:,T_obs].=1.0
        for t in T_obs-1:-1:1
            @inbounds for g1 in 1:G
                val=0.0; for g2 in 1:G; val+=T_mat[g1,g2]*exp(log_fe[g2,i,t+1])*β_bw[g2,t+1]*sw[g2]; end
                β_bw[g1,t]=val
            end
            Lb=dot(view(β_bw,:,t),sw); Lb>0&&(β_bw[:,t]./=Lb)
        end

        # Smoothed marginals
        for t in 1:T_obs
            @inbounds for g in 1:G; γ[g,t]=α[g,t]*β_bw[g,t]; end
            Z=dot(view(γ,:,t),sw); Z>0&&(γ[:,t]./=Z)
        end

        # ∂S_eps/∂θ_eps at each grid point for this individual (no log C)
        dS_eps = zeros(G, T_obs, 2)
        for t in 1:T_obs, g in 1:G
            dS_eps[g,t,:] = dS_eval(S_eps_at, (y[i,t]-ws.grid[g],), idx_eps)
        end

        # ---- Compute E[dS|Y_i] and E[dS dS'|Y_i] ----
        # dS_total = [dS_Q; dS_init; dS_eps] summed over t
        # t=1: dS_init(η₁) + dS_eps(y₁-η₁), depends on η₁
        # t≥2: dS_trans(η_t,η_{t-1}) + dS_eps(y_t-η_t), depends on (η_{t-1},η_t)

        E_dS = zeros(np)
        E_dSdS = zeros(np, np)

        # t=1: marginal γ(·,1)
        for g in 1:G
            w = γ[g,1]*sw[g]
            ds = zeros(np)
            ds[idx_init] = dS_init[g,:]
            ds[idx_eps] = dS_eps[g,1,:]
            E_dS .+= w .* ds
            E_dSdS .+= w .* (ds * ds')
        end

        # t=2: pairwise ξ(g1,g2,2)
        for g1 in 1:G, g2 in 1:G
            ξ = α[g1,1]*T_mat[g1,g2]*exp(log_fe[g2,i,2])*β_bw[g2,2]*sw[g1]*sw[g2]
            ξ < 1e-300 && continue
            ds = zeros(np)
            ds[idx_Q] = dS_trans[g1,g2,:]
            ds[idx_eps] = dS_eps[g2,2,:]
            E_dS .+= ξ .* ds
            E_dSdS .+= ξ .* (ds * ds')
        end

        # t=3 (if T≥3): pairwise ξ(g2,g3,3)
        if T_obs >= 3
            for g2 in 1:G, g3 in 1:G
                ξ = α[g2,2]*T_mat[g2,g3]*exp(log_fe[g3,i,3])*β_bw[g3,3]*sw[g2]*sw[g3]
                ξ < 1e-300 && continue
                ds = zeros(np)
                ds[idx_Q] = dS_trans[g2,g3,:]
                ds[idx_eps] = dS_eps[g3,3,:]
                E_dS .+= ξ .* ds
                E_dSdS .+= ξ .* (ds * ds')
            end
        end

        # Cross terms between t=1 and t=2: needs ξ(g1,g2,2)
        for g1 in 1:G, g2 in 1:G
            ξ = α[g1,1]*T_mat[g1,g2]*exp(log_fe[g2,i,2])*β_bw[g2,2]*sw[g1]*sw[g2]
            ξ < 1e-300 && continue
            ds1 = zeros(np); ds1[idx_init]=dS_init[g1,:]; ds1[idx_eps]=dS_eps[g1,1,:]
            ds2 = zeros(np); ds2[idx_Q]=dS_trans[g1,g2,:]; ds2[idx_eps]=dS_eps[g2,2,:]
            E_dSdS .+= ξ .* (ds1*ds2' + ds2*ds1')
        end

        # Cross terms between t=2 and t=3: needs triple (g1,g2,g3)
        # and between t=1 and t=3: also triple
        # Use coarse grid for these
        if T_obs >= 3
            stride = max(1, G ÷ 30)
            for g1 in 1:stride:G, g2 in 1:stride:G, g3 in 1:stride:G
                ξ123 = α[g1,1]*T_mat[g1,g2]*exp(log_fe[g2,i,2]) *
                        T_mat[g2,g3]*exp(log_fe[g3,i,3]) * sw[g1]*sw[g2]*sw[g3]*stride^3
                ξ123 < 1e-300 && continue

                ds2 = zeros(np); ds2[idx_Q]=dS_trans[g1,g2,:]; ds2[idx_eps]=dS_eps[g2,2,:]
                ds3 = zeros(np); ds3[idx_Q]=dS_trans[g2,g3,:]; ds3[idx_eps]=dS_eps[g3,3,:]
                E_dSdS .+= ξ123 .* (ds2*ds3' + ds3*ds2')

                # t=1 and t=3 cross
                ds1 = zeros(np); ds1[idx_init]=dS_init[g1,:]; ds1[idx_eps]=dS_eps[g1,1,:]
                E_dSdS .+= ξ123 .* (ds1*ds3' + ds3*ds1')
            end
        end

        # Var[dS|Y_i] = E[dS dS'|Y_i] - E[dS|Y_i] E[dS|Y_i]'
        Var_i = E_dSdS - E_dS * E_dS'
        Var_total .+= Var_i

        (i % 20 == 0 || i == N) && (@printf("    i=%d/%d\n", i, N); flush(stdout))
    end

    # Average per individual
    Var_per_i = Var_total / N

    Var_per_i
end

# ================================================================
#  MAIN: compute and display (only runs when executed directly)
if abspath(PROGRAM_FILE) == @__FILE__
# ================================================================

K=2; σy=1.0; τ=[0.25,0.50,0.75]; N=100
tp = make_true_cspline()
v_prof = pack_profiled(tp.a_Q, tp.a_init, tp.a_eps1, tp.a_eps3)
y, _ = generate_data_cspline(N, tp.a_Q, tp.M_Q, tp.a_init, tp.M_init,
                              tp.a_eps1, tp.a_eps3, tp.M_eps, K, σy, τ; seed=1, T=3)

@printf("Computing missing information Var[∂S_spline/∂θ | Y]...\n")
@printf("N=%d, G=201, T=3\n\n", N)
Var_S = compute_missing_info(v_prof, y, K, σy, τ; G=201)

param_names = ["med_q0","med_q1","med_q2","δL1","δL2","δL3","δR1","δR2","δR3",
                "init_m","log_gL","log_gR","l_ae1","l_ae3"]
np = 14

@printf("\nMissing information Var[∂S_spline/∂θ | Y] (per individual, ×N for total):\n")
@printf("%-10s  %12s  %12s\n", "Param", "Var (per i)", "√Var")
println("-" ^ 40)
for j in 1:np
    @printf("%-10s  %12.6f  %12.6f\n", param_names[j], Var_S[j,j], sqrt(max(Var_S[j,j],0)))
end

# Eigendecomposition
evals = eigvals(Symmetric(Var_S))
evecs = eigvecs(Symmetric(Var_S))
order = sortperm(evals, rev=true)

@printf("\nEigenvalues of Var[∂S/∂θ|Y] (largest = most missing info):\n")
for (i,idx) in enumerate(order)
    ev=evecs[:,idx]; top=sortperm(abs.(ev),rev=true)
    @printf("  λ_%2d=%+12.6f  ", i, evals[idx])
    for t in top[1:min(3,np)]; abs(ev[t])>0.15 && @printf("%s(%.2f) ",param_names[t],ev[t]); end
    println()
end
@printf("\nCondition of Var: %.2e\n", maximum(evals)/max(minimum(evals[evals.>0]),1e-15))

# Off-diagonal blocks: correlation between init and eps params
@printf("\nCorrelation between init and eps parameters (off-diagonal of Var):\n")
for j in 10:12, k in 13:14
    corr = Var_S[j,k] / sqrt(max(Var_S[j,j]*Var_S[k,k], 1e-30))
    @printf("  Corr(%s, %s) = %.4f\n", param_names[j], param_names[k], corr)
end

end # if abspath(PROGRAM_FILE) == @__FILE__
