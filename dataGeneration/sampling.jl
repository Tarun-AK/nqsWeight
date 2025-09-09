using ITensors, LinearAlgebra, ITensorMPS, NPZ
using DelimitedFiles
using Printf
include("evolMPS.jl")

function MIE_sample(M::myMPS, Ps::Vector{<:Matrix}; truncation = true, max_bd = 1024, max_err = 1E-8) 
    ## measuring all sites with POVM specified by positive observables Ps = [P_1,P_2,...]
    ## output the string s with probability p = |<s|psi>|^2
    M_cp = copy(M)
    N = length(M)
    p_out = 1.
    m_out = Int64[]
    for i in 1:N
        Ac = M_cp.TensorList[i]
        ps = zeros(length(Ps))
        for s in 1:length(ps)
            p = real(expect_value_onsite(Ac, Ps[s]))
            ps[s] = p
        end
        r = rand()
        set_projector = length(Ps)
        for l in 1:length(ps)
            r = r-ps[l]
            if(r<=sqrt(eps(Float64)))
                set_projector = l
                break
            end
        end
        P = Ps[set_projector]
        p = ps[set_projector]
        p_out = p*p_out
        push!(m_out,set_projector)
        Anew = ncon([Ac,P],[[-1,1,-3],[1,-2]])./sqrt(p)
        M_cp.TensorList[i] = Anew
        ~, M_cp = canonicalize_left_one_site(M_cp, i, truncation = truncation, max_bd = max_bd, max_err = max_err)
    end
    return m_out, p_out
end


function Ising_GS_DMRG_symmetric(N,h=1.0,pbc=true,parity::Int = 1;max_bd=100,nsweeps = 20)
    ## parity = 1 => symmetric under Z2
    ## parity = -1 => odd under Z2
    sites = siteinds("S=1/2",N, conserve_szparity = true)

    os = OpSum()
    for j in 1:N-1
        os += -4.0,"Sx",j,"Sx",j+1
    end
    if (pbc)
        os += -4.0,"Sx",N,"Sx",1
    end
    for j in 1:N
        os += -2.0*h,"Sz",j
    end
    H = MPO(os,sites)

    
    maxdim = [20,32,64,max_bd] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error
    noise = [1E-6,1E-7,1E-8,1E-8,1E-8,0.0]

    state = ["Up" for n=1:N]
    if(parity == -1)
        state[1] = "Down"
    end
        
    psi0 = MPS(sites,state)

    E0,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff,noise)
    return psi
end

function strip_QNs(ψ::MPS)
    dense_tensors = ITensor[]
    
    for A in ψ
        # Convert QN tensor to dense tensor
        A_dense = dense(A)
        push!(dense_tensors, A_dense)
    end
    
    return MPS(dense_tensors)
end

function MPS_to_array(psi::MPS)
    N=length(psi)
    As=[];
    for i=1:N
        if(i<N)
            rightind=intersect(inds(psi[i]),inds(psi[i+1]))
        else
            rightind=[]
        end
        if(i>1)
            leftind=intersect(inds(psi[i]),inds(psi[i-1]))
        else
            leftind=[]
        end
        physind=setdiff(inds(psi[i]),union(rightind,leftind))
        MPSinds=vcat(leftind,physind,rightind)
        A=Array(psi[i],MPSinds...)                   # Array(T::ITensor, inds::Index...), returns a plain Julia Array with raw numerical values
        push!(As,A)
    end
    As[1]=reshape(As[1],(1,2,2))
    As[end]=reshape(As[end],(2,2,1));
    As = [convert(Array{Float64,3},A) for A in As];
    return As
end

function get_Ising_GS(N::Int, h::Float64, max_bd = 100)
    psi = Ising_GS_DMRG_symmetric(N,h,true,1,max_bd=max_bd);
    psi_dense = strip_QNs(psi);
    As = MPS_to_array(psi_dense);
    M = myMPS(As);
    return M
end

function get_Ising_GS_Zbasis(N::Int, h::Float64, max_bd = 100)
    Hadmard = sqrt(1/2).*[[1 1];[1 -1]]
    psi = Ising_GS_DMRG_symmetric(N,h,true,1,max_bd=max_bd);
    psi_dense = strip_QNs(psi);
    As = MPS_to_array(psi_dense);
    Bs = Array{Float64, 3}[]
    for A in As
        @tensor B[a,s,b] := A[a,ss,b] * Hadmard[s,ss]
        push!(Bs,B)
    end
    M = myMPS(Bs);
    return M
end

for h in 0.05:0.05:2 
    M = get_Ising_GS_Zbasis(32,h, 100)
    sigma_pauli = diagm([1.0,-1.0]) # Z basis
    Ps = [(I-sigma_pauli)./2, (I+sigma_pauli)./2]
    out_strings = [] ## output strings of outcomes
    out_probs = [] ##each output probability
    n_samp = 100000;
    @time for _ in 1:n_samp
        m_out,p_out = MIE_sample(M, Ps)

        push!(out_strings,(2*m_out .- 3))
        push!(out_probs,p_out)
    end
    h_formatted = @sprintf("%.3f", h)
    writedlm("traindata/isingz/d=1/measurements_L=32_h=$h_formatted.csv", out_strings)
    writedlm("traindata/isingz/d=1/ps_L=32_h=$h_formatted.csv", out_probs)
end
