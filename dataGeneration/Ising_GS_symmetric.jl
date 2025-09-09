using ITensors, ITensorMPS
include("evolMPS.jl")

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

M = get_Ising_GS(32, 0.1, 100);