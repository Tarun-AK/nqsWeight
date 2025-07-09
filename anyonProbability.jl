using Pkg
# Pkg.add.(["PyPlot", "BenchmarkTools", "LaTeXStrings", "IOCapture",
#           "JLD2", "FileIO", "TensorOperations", "MKL", 
#           "CSV", "DataFrames"])

using Distributed
np = length(Sys.cpu_info())
addprocs(np-2);
print(nprocs())
# addprocs(26)
using PyPlot
using Profile, BenchmarkTools
using LaTeXStrings
using IOCapture



@everywhere using JLD2, FileIO
@everywhere using LinearAlgebra, Random, TensorOperations, Statistics

for i in 1:nprocs()
    s = rand(1:9999)
    @spawnat i Random.seed!(s)
end


PyPlot.rc("font", family="serif")
matplotlib.style.use("default")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["mathtext.fontset"] = "cm"
rcParams["mathtext.rm"] = "serif"

cm = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f","#bcbd22", "#17becf", "b", "g", "r", "c", "m", "y", 
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f","#bcbd22", "#17becf", "b", "g", "r", "c", "m", "y",
];

@everywhere using MKL
@everywhere BLAS.set_num_threads(1)
# BLAS.set_num_threads(29)
"""
Functions for MPS canonicalization, modified from Yijian's original code
"""

@everywhere begin

    struct myMPS{T<:Number}
        TensorList::Array{Array{T,3},1} #List of myMPS tensors that represent the purification 
        #Tensor indices - left bond, system spin, right bond
    end
@everywhere Base.length(M::myMPS) = length(M.TensorList)
    function truncate(S::Vector{<:Real}, max_bd::Int, max_err::Float64)
        ## Given an array S (descending), determine the truncation 
        ## based on which of max bond dimesion or max err is reached
        err = 0.0
        set_bd = max_bd
        for i in length(S):-1:1
            err = err + S[i]^2
            if(err>max_err)
                if(i<max_bd)
                    set_bd = i
                end
                break
            end
        end
        return set_bd
    end

    function canonicalize_left_one_site(M::myMPS, site::Int;truncation = false, max_bd = 1024, max_err = 1E-10) where T
        ## A1 := M[site], A2 := M[site+1]
        ## A1 = USV' => A1=U, A2 = SVt (update)
        ## Truncate S if truncation = true
        ## return S and the updated myMPS M
        A1 = M.TensorList[site]
        DL,d,DR = size(A1)
        A1_mat = reshape(A1, (DL*d,DR))
        U = nothing; S=nothing; V=nothing;
        try
            U,S,V = svd(A1_mat,alg=LinearAlgebra.DivideAndConquer())
        catch
            U,S,V = svd(A1_mat,alg=LinearAlgebra.QRIteration())
        end
        if(norm(S)<eps(Float64))
            throw("zero norm")
        end
        S = S./norm(S)
        if(truncation == true)
            set_bd = truncate(S,max_bd,max_err)
            trunc_err = norm(S[set_bd+1:end])^2
            if(trunc_err>1E-6)
                println("truncation error:",trunc_err)
            end
            S = S[1:set_bd]
            U = U[:,1:set_bd]
            V = V[:,1:set_bd]
        end
        M.TensorList[site] = reshape(U, (DL,d,length(S)))
        if(site<length(M))
            SVt = diagm(0=>S)*V'
            
            A = M.TensorList[site+1]
            @tensor tmp[i, j, k] := SVt[i, x] * A[x, j, k]
            M.TensorList[site+1] = tmp
    #         M.TensorList[site+1] = ncon([SVt, M.TensorList[site+1]],[[-1,1],[1,-2,-3]])
        end
        return S, M
    end

    function canonicalize_right_one_site(M::myMPS, site::Int; truncation = false, max_bd = 1024, max_err = 1E-10)
        ## A1 := M[site], A2 :=M[site-1]
        ## A1 = USV' => A1=V', A2 = US
        ## Truncate S if truncation = true
        A1 = M.TensorList[site]
        DL,d,DR = size(A1)
        A1_mat = reshape(A1, (DL,DR*d))
        U = nothing; S=nothing; V=nothing;
        try
            U,S,V = svd(A1_mat,alg=LinearAlgebra.DivideAndConquer())
        catch
            U,S,V = svd(A1_mat,alg=LinearAlgebra.QRIteration())
        end
        if(norm(S)<eps(Float64))
            throw("zero norm")
        end
        S = S./norm(S)
        if(truncation == true)
            set_bd = truncate(S,max_bd,max_err)
            trunc_err = norm(S[set_bd+1:end])^2
            if(trunc_err>1E-6)
                println("truncation error:",trunc_err)
            end
            S = S[1:set_bd]
            U = U[:,1:set_bd]
            V = V[:,1:set_bd]
        end
        M.TensorList[site] = reshape(V', (length(S),d,DR))
        if(site>1)
            US = U*diagm(0=>S)
            
            A = M.TensorList[site-1]
            @tensor tmp[i, j, k] := US[x, k] * A[i, j, x]
            M.TensorList[site-1] = tmp
    #         M.TensorList[site-1] = ncon([US, M.TensorList[site-1]],[[1,-3],[-1,-2,1]])
        end
        return S, M
    end

    function canonicalize_left(M::myMPS;truncation = false, max_bd = 1024, max_err = 1E-10)
        ## Return a left canonical form of the purification (normalized automatically)
        N = length(M)
        for i in 1:N
            ~, M = canonicalize_left_one_site(M, i, truncation=truncation,max_bd=max_bd,max_err=max_err)
        end
        return M
    end

    function canonicalize_right(M::myMPS;truncation = false, max_bd = 1024, max_err = 1E-10)
        ## Return a right canonical form of the purification (normalized automatically)
        N = length(M)
        for i in N:-1:1
            ~, M = canonicalize_right_one_site(M, i, truncation=truncation,max_bd=max_bd,max_err=max_err)
        end
        return M
    end
end
@everywhere function log_abs_inner(A1s::Vector{Array{T, 3}}, A2s::Vector{Array{T, 3}}) where T
    @assert length(A1s) == length(A2s)
    n = length(A1s)
    q = ones(T, 1, 1)
    log_abs = zero(T)
    for ii in 1:n
        @tensor tmp[i, j] := q[a, b] * A1s[ii][a, k, i] * A2s[ii][b, k, j]
        log_abs += log2(norm(tmp))
        q = tmp / norm(tmp)
    end
    return log_abs
end
    

@everywhere function normalize_and_retruncate(As::Vector{Array{T, 3}}, err) where T
    n = length(As)
    
    q = ones(T, 1, 1)
    log_norm = zero(T)
    for i in 1:n
        A = As[i]
        @tensor tmp[i, j] := q[a, b] * A[a, k, i] * A[b, k, j]
        log_norm += 0.5*log2(norm(tmp))
        q = tmp / norm(tmp)
    end
    @assert size(q) == (1, 1)
    
    M = myMPS(As)
    M = canonicalize_left(M)
    M = canonicalize_right(M; truncation=true, max_bd=1000, max_err=err)
    return M.TensorList, log_norm
end

@everywhere function max_bd(As)
    return maximum([size(A, 3) for A in As])
end

@everywhere function id_mat(n)
    m = zeros(n, n)
    [m[i,i]=1 for i in 1:n]
    return m
end


@everywhere function four_legs(T, p, s)
    tmp = zeros(T, 2, 2, 2, 2)
    for i in 0:1, j in 0:1, k in 0:1, l in 0:1
        if isodd(i+j+k+l+s)
            tmp[i+1, j+1, k+1, l+1] = 0
        else
            tmp[i+1, j+1, k+1, l+1] = sqrt(p^(i+j+k+l) * (1-p)^(4-i-j-k-l))
        end
    end
    return tmp
end    


@everywhere function splitted_four_legs(T, p, s)
    t4 = four_legs(Float64, p, s)
    m = reshape(t4, (4,4))
    U, D, Vt = svd(m)

    L = reshape(U[:, 1:2]*diagm(D[1:2]), (2,2,2))
    R = reshape(Vt[:, 1:2]', (2,2,2))
    
    @tensor tmp[i,j,k,l] := L[i,j,x]*R[x,k,l]
    @assert tmp ≈ t4
    
    return L, R
end


@everywhere function two_legs(T, p)
    tmp = zeros(T, 1, 2)
    tmp[1, 1], tmp[1, 2] = sqrt(1-p), sqrt(p)
    return tmp
end


@everywhere function sample_ss(Lx, Ly, p)
    tmp = zeros(Bool, Lx+2, Ly+2)
    
    # x-links
    for x in 1:Lx+2, y in 1:Ly+1
        s = (rand() < p)
        tmp[x, y] ⊻= s
        tmp[x, y+1] ⊻= s
    end
    
    # y-links
    for x in 1:Lx+1, y in 1:Ly+2
        s = (rand() < p)
        tmp[x, y] ⊻= s
        tmp[x+1, y] ⊻= s
    end
    
    return tmp[2:Lx+1, 2:Ly+1]
end

@everywhere function t2_product_state(T, n, p)
    t2 = reshape(two_legs(T, p), (1,2,1))
    return [t2 for _ in 1:n]
end

@everywhere function mpo_on_mps(mpo, mps::Vector{T}) where T
    @assert size(mpo) == size(mps)
    n = length(mps)
    new_mps = T[]
    for i in 1:n
        A = mps[i]
        B = mpo[i]
#         @show size(A), size(B)
        @tensor C[l1, l2, k, r1, r2] := B[l1, k, x, r1] * A[l2, x, r2]
        tmp = reshape(C, size(A, 1)*size(B,1), size(B, 2), size(A,3)*size(B,4))
        push!(new_mps, tmp)
    end
    return new_mps
end

@everywhere function decorated_parity_state(T, n, s, p)
    As = Array{T, 3}[]
    
    A = zeros(T, 1, 2, 2)
    for i in 0:1, j in 0:1
        A[1, i+1, j+1] = 1 - mod(i+j+s, 2)
    end
    A[:, 1, :] .*= sqrt(1-p)
    A[:, 2, :] .*= sqrt(p)
    push!(As, A)
    
    A = zeros(T, 2, 2, 2)
    for i in 0:1, j in 0:1, k in 0:1
        A[i+1, j+1, k+1] = 1 - mod(i+j+k, 2)
    end
    A[:, 1, :] .*= sqrt(1-p)
    A[:, 2, :] .*= sqrt(p)
    [push!(As, copy(A)) for _ in 1:n-2]
    
    A = zeros(T, 2, 2, 1)
    for i in 0:1, j in 0:1
        A[i+1, j+1, 1] = 1 - mod(i+j, 2)
    end
    A[:, 1, :] .*= sqrt(1-p)
    A[:, 2, :] .*= sqrt(p)
    push!(As, A)
    
    return As
end

@everywhere get_bds(As) = [size(A,3) for A in As]


@everywhere function pbc_to_obc(As::Vector{Array{T,3}}) where T
    @assert size(As[1], 1) == size(As[end], 3)
    @assert size(As[1], 1) != 1
    d = size(As[1], 1)
    idd = zeros(T, d, d)
    [idd[x,x]=1 for x in 1:d]
    As_new = Array{T,3}[]
    
    @tensor tmp[j, k1, k2] := As[1][k2, j, k1]
    push!(As_new, reshape(tmp, 1, size(As[1], 2), d * size(As[1], 3)))
    
    for u in 2:length(As)-1
        A = As[u]
        @tensor tmp[i1, i2, j, k1, k2] := A[i1, j, k1] * idd[i2, k2]
        push!(As_new, reshape(tmp, d * size(A,1), size(A, 2), d * size(A,3)))
    end
    
    @tensor tmp[i1, i2, j] := As[end][i1, j, i2]
    push!(As_new, reshape(tmp, size(As[end], 1)*d, size(As[end], 2), 1))
    
    return As_new
end


@everywhere function get_mpo_geomII(T, ss, l, p)
    # counter-clockwise
    @assert size(ss, 1) == size(ss, 2)
    n = size(ss, 1)
    corner_ss = [ss[l, l], ss[n-l+1, l], ss[n-l+1,n-l+1], ss[l, n-l+1]]
    edge_ss = [ss[l+1:n-l, l], ss[n-l+1, l+1:n-l], ss[n-l:(-1):l+1, n-l+1], ss[l, n-l:(-1):l+1]]
    
    dummy_t4 = zeros(T, 1,1,1,1)
    Bs = Array{T, 4}[]
    
    L1, R1 = splitted_four_legs(T, p, corner_ss[1])
    push!(Bs, reshape(R1, (2, 1, 2, 2)))
    for c in 1:4
        if c > 1
            L, R = splitted_four_legs(T, p, corner_ss[c])
            push!(Bs, reshape(L, (2, 1, 2, 2)))
            push!(Bs, reshape(R, (2, 1, 2, 2)))
        end
        for s in edge_ss[c]
            push!(Bs, four_legs(T, p, s))
        end
    end
    push!(Bs, reshape(L1, (2, 1, 2, 2)))
    @assert length(Bs) == 4n-8(l-1)
    return Bs
end

@everywhere function up_side_down(Bs)
    return [permutedims(B, [1, 3, 2, 4]) for B in Bs]
end
@everywhere function mpo_on_mps_geomII(mpo, mps::Vector{T}) where T
    As0 = mpo_on_mps(mpo, mps)
    x = div(length(As0)-8, 4)
    @assert x>0
    to_be_shrinked = [1, x+2, x+3, 2x+4, 2x+5, 3x+6, 3x+7, 4x+8]
    [@assert size(As0[i], 2)==1 for i in to_be_shrinked]
    [@assert size(As0[i], 3)==size(As0[i+1], 1) for i in 1:4x+7]
    
    # shinks dummy As
    @tensor tmp1[i, j, k, u] := As0[1][i, u, x] * As0[2][x, j, k]
    As0[2   ] = tmp1[:,:,:,1]
    @tensor tmp2[i, j, k] := As0[1x+1][i, j, x] * As0[1x+2][x, u, y] * As0[1x+3][y, u, k]
    As0[1x+1] = tmp2
    @tensor tmp3[i, j, k] := As0[2x+3][i, j, x] * As0[2x+4][x, u, y] * As0[2x+5][y, u, k]
    As0[2x+3] = tmp3
    @tensor tmp4[i, j, k] := As0[3x+5][i, j, x] * As0[3x+6][x, u, y] * As0[3x+7][y, u, k]
    As0[3x+5] = tmp4
    @tensor tmp5[i, j, k, u] := As0[4x+7][i, j, x] * As0[4x+8][x, u, k]
    As0[4x+7] = tmp5[:,:,:,1]
    
    As1 = [As0[i] for i in setdiff(1:4x+8, to_be_shrinked)]
    [@assert size(As1[i], 3)==size(As1[i+1], 1) for i in 1:4x-1]
    
    return As1
end


@everywhere function get_f1_f2(T, p, ss, w, err)
    n = size(ss, 1)
    As = t2_product_state(T, 4n, p)
    f1, f2 = 0., 0.
    
    for i in 1:w
        Bs = get_mpo_geomII(T, ss, i, p)
        @show i, length(As), size(Bs[1], 1), size(Bs[end], 4)
        As = mpo_on_mps_geomII(Bs, As)
        As = pbc_to_obc(As)
        As, a = normalize_and_retruncate(As, err)
        f1 += a
        f2 += a
#         @show max_bd(As)
    end
    s = mod(sum(ss[1+w:end-w, 1+w:end-w]), 2)
    tmp = decorated_parity_state(T, length(As), s, p)
    f1 += log_abs_inner(As, tmp)
    
    for i in w+1:div(n-1,2)
        Bs = get_mpo_geomII(T, ss, i, p)
        @show i, length(As), size(Bs[1], 1), size(Bs[end], 4)
        As = mpo_on_mps_geomII(Bs, As)
#         @show i, length(As), size(As[1], 1), size(As[end], 3)
        As = pbc_to_obc(As)
        As, a = normalize_and_retruncate(As, err)
        f2 += a
#         @show max_bd(As)
    end
    @show length(As)
    @assert length(As)==4
    @assert size(As[1], 1)==1
    c = four_legs(T, p, ss[div(n-1,2)+1, div(n-1,2)+1])
    @tensor A4[i, j, k, l] := As[1][x,i,y]*As[2][y,j,z]*As[3][z,k,w]*As[4][w,l,x]
    tmpp = sum(reshape(A4, 16) .* reshape(c, 16))
    f2 += log2(abs(tmpp))
    return f1, f2
end

using CSV
using DataFrames
using Printf


for i in 0:19
    out_probs = []
    pfloat = 0.01*i
    p = @sprintf("%.3f", pfloat)
    # matrix = Matrix(CSV.read("traindata/toric/d=2/measurements_L=13_p=$p.csv", DataFrame; delim=' ', ignorerepeated=true, header=false))
    matrix = Matrix(CSV.read("traindata/toric/d=2/exactMs_L=3.csv", DataFrame; delim=' ', ignorerepeated=true, header=false))
    for j in 1:2^9
        samp = matrix[j, :]
        print(samp)
        samp = reshape(samp, 3, 3)
        print(size(samp))
        samp = -(samp .- 1)/2
        f1, p_out = get_f1_f2(Float64, pfloat, samp, 1, 1e-8)
        push!(out_probs,2^p_out)
    end
    using DelimitedFiles
    writedlm("traindata/toric/d=2/ps_L=3_p=$p.csv", out_probs)
end
