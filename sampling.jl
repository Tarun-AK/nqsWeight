using ITensors, LinearAlgebra, ITensorMPS, NPZ
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


function Ising_GS(N,max_bd, h)
    sites = siteinds("S=1/2",N)

    os = OpSum()
    for j in 1:N
        os += -4.0,"Sz",j,"Sz",j%N+1
    end
    for j in 1:N
        os += -h,"Sx",j
    end
    H = MPO(os,sites)

    nsweeps = 10 # number of sweeps is 5
    maxdim = [10,20,40,max_bd] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error
    noise = [1E-7]
    
    psi0 = randomMPS(sites,8)

    energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)
    return psi
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
        A=Array(psi[i],MPSinds...)
        push!(As,A)
    end
    As[1]=reshape(As[1],(1,2,2))
    As[end]=reshape(As[end],(2,2,1));
    As = [convert(Array{Float64,3},A) for A in As];
    return As
end

for h in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    psi = Ising_GS(32,100, h)
    M = myMPS(MPS_to_array(psi));

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

    using DelimitedFiles

    writedlm("traindata/isingTrainingSet_ZZ+X_h=$h.csv", out_strings)
    writedlm("traindata/ps_ZZ+X_h=$h.csv", out_probs)
end
