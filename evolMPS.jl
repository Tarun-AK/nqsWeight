using LinearAlgebra, TensorOperations, Statistics

struct myMPS{T<:Number}
    TensorList::Array{Array{T,3},1} #List of myMPS tensors that represent the purification 
    #Tensor indices - left bond, system spin, right bond
end

Base.length(M::myMPS) = length(M.TensorList)
phys_dim(M::myMPS) = size(M.TensorList[1],2)
testnan(M::myMPS) = sum([sum(isnan.(ten)) for ten in M.TensorList])
testnorm(M::myMPS) = findmin([norm(ten) for ten in M.TensorList])[1]
max_bond_dim(M::myMPS) = findmax([size(ten,1) for ten in M.TensorList])[1]
Base.copy(M::myMPS) = myMPS(copy(M.TensorList))

function product_state_init(T::Type, d::Int, N::Int) 
    ## Initialize a product state |0101010101...>
    ## T - data type
    ## d - local dimension
    ## N - number of sites
    Ten_even = zeros(T,1,d,1)
    Ten_odd = zeros(T,1,d,1) 
    Ten_even[1,1,1] = 1.0
    Ten_odd[1,2,1] = 1.0
    myMPSTensors = Array{T,3}[]
    for i in 1:N
        if(i%2==0)
            push!(myMPSTensors, Ten_even)
        else
            push!(myMPSTensors, Ten_odd)
        end
    end
    return myMPS(myMPSTensors)
end

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

function canonicalize_left_one_site(M::myMPS, site::Int;truncation = false, max_bd = 1024, max_err = 1E-10)
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
        M.TensorList[site+1] = ncon([SVt, M.TensorList[site+1]],[[-1,1],[1,-2,-3]])
    end
    return S, M
end

function canonicalize_right_one_site(M::myMPS, site::Int;truncation = false, max_bd = 1024, max_err = 1E-10)
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
        M.TensorList[site-1] = ncon([US, M.TensorList[site-1]],[[1,-3],[-1,-2,1]])
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

function unitary_evol_two_site(M::myMPS, U::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the pure state by U 
    ## Assuming U is two-site
    ## further assuming the two sites are the center of canonical form if truncation = true (important!)
    ## site is the first site # of the unitary (1 to N-1)
    ## the [site] of the two sites is put into left canonical form (if dir = "l")
    ## the [site+1] of the two sites is put into right canonical form (if dir = "r")
    ## return S and M
    myMPSTensors = M.TensorList
    @assert site<length(M)
    A1 = myMPSTensors[site]
    A2 = myMPSTensors[site+1]
    d1 = size(A1,2)
    d2 = size(A2,2)
    DL = size(A1,1)
    DR = size(A2,3)
    U = reshape(U,(d1,d2,d1,d2)) # site 1 ket, site 2 ket, site 1 bra, site 2 bra
    A_evol = ncon([A1,A2,U],[[-1,1,2],[2,3,-4],[1,3,-2,-3]])
    A_evol_mat = reshape(A_evol,(DL*d1,DR*d2))
    U2 = nothing; S2=nothing; V2=nothing;
    try
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.QRIteration())
    end
    if(truncation == true)
        set_bd = truncate(S2, max_bd, max_err)
        trunc_err = norm(S2[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S2 = S2[1:set_bd]
        U2 = U2[:,1:set_bd]
        V2 = V2[:,1:set_bd]
    end
    if(dir == "l")
        AL = reshape(U2, (DL,d1,length(S2)))
        AR = reshape(diagm(0=>S2)*V2',(length(S2),d2,DR))
    else
        AL = reshape(U2*diagm(0=>S2), (DL,d1,length(S2)))
        AR = reshape(V2',(length(S2),d2,DR))
    end
    M.TensorList[site] = AL
    M.TensorList[site+1] = AR    
    return S2, M
end

apply_TM_l(A::Array{<:Number,3},B::Array{<:Number,3},l::Array{<:Number,2})=ncon([A,conj.(B),l],[[3,2,-2],[1,2,-1],[1,3]])
apply_TM_r(A::Array{<:Number,3},B::Array{<:Number,3},r::Array{<:Number,2})=ncon([A,conj.(B),r],[[-1,2,1],[-2,2,3],[1,3]])

function expect_values_onsite(l::Array{<:Number,2},Ac::Array{<:Number,3},P::Array{<:Matrix{<:Number},1},r::Array{<:Number,2})
    ## Assuming P add up to identity, so the last probability can be computed by 1-others
    vs = zeros(length(P))
    for i in 1:length(P)-1
        prob = ncon([l,Ac,conj.(Ac),P[i],r],[[3,1],[1,2,6],[3,4,7],[2,4],[6,7]])[1]
        vs[i] = real(prob)
    end
    vs[end] = 1.0-sum(vs[1:end-1])
    return vs
end

function measurement_probs_one_site(M::myMPS, Ps::Array{<:Array{<:Matrix{<:Number},1},1}, sites::Array{Int,1})
    ## sites: array of indices for measuring, in ascending order
    ## Ps: array of same sizes as sites, each consisting of a set of projectors acting on sites.
    ## return probs - Array{Array{Float}} probabilities of each projector
    l = diagm(0=>[1.0+0.0im])
    l_envs = Matrix[] #left environment for each measured site
    if(1 in sites)
        push!(l_envs,l)
    end
    myMPSTensors = M.TensorList
    for i in 1:sites[end]-1
        l = apply_TM_l(myMPSTensors[i],myMPSTensors[i],l)
        if(i+1 in sites)
            push!(l_envs,l)
        end
    end
    r = diagm(0=>[1.0+0.0im])
    probs=Array{Float64,1}[]
    for j in length(M):-1:sites[1]
        if(j in sites)
            indP = findall(sites.==j)[1]
            push!(probs,expect_values_onsite(l_envs[end],myMPSTensors[j],Ps[indP],r)) 
            pop!(l_envs)
        end
        if(j>sites[1])
            r = apply_TM_r(myMPSTensors[j],myMPSTensors[j],r)
        end
    end
    probs = probs[end:-1:1]
    return probs
end

function measurement_one_site(M::myMPS,Ps::Array{<:Array{<:Matrix{<:Number},1},1}, sites::Array{Int,1})
    ## Implement the measurement by a random projection chosen according to the probability  
    probs = measurement_probs_one_site(M,Ps,sites)
    for i in 1:length(sites)
        site = sites[i]
        prob = probs[i]
        set_projector = 1
        r = rand()
        for l in 1:length(prob)
            r = r-prob[l]
            if(r<=sqrt(eps(Float64)))
                set_projector = l
                break
            end
        end
        P = Ps[i][set_projector]
        M.TensorList[site] = ncon([M.TensorList[site],P],[[-1,1,-3],[1,-2]])
    end
    return M
end         

function measurement_layer_spin_half(M::myMPS{T}, p::Float64) where T
    ### Apply Z basis measurement at each site with probability p
    sigmaZ = diagm(0=>[1.0,-1.0])
    P_plus = (I+sigmaZ)./2.0
    P_minus = (I-sigmaZ)./2.0 #define the projectors here
    N = length(M)
    rlist = rand(N)
    sites = Int[]
    Ps = Array{Matrix{T},1}[]
    for i in 1:N
        if(rlist[i]<p)
            push!(sites, i)
            push!(Ps, Matrix{T}[P_plus,P_minus])
        end
    end
    if(length(sites)>=1)
        M = measurement_one_site(M,Ps,sites)
    end
    return M
end

function Haar_random_unitary(d::Int)
    ## return a d*d Haar random unitary matrix (complex)
    H = randn(Complex{Float64},d,d)
    Q,R = qr(H)
    fac = diagm(0=>[R[i,i]/abs(R[i,i]) for i in 1:d])
    Q = Q*fac
    return Q
end

function unitary_layer_spin_half(M::myMPS, eo = "o";truncation = false, max_bd = 1024, max_err = 1E-10)
    ## Apply random Haar unitary at each even/odd sites
    ## Note - right canonical form assumed in beginning, and left canonical form in the end
    N = length(M)
    d = phys_dim(M)
    S_list = Vector{Float64}[]
    if(eo == "o")
        site0 = 1
    else
        S, M = canonicalize_left_one_site(M,1,truncation = truncation, max_bd = max_bd, max_err = max_err)
        push!(S_list,S)
        site0 = 2
    end
    for site in site0:2:N-1
        U = Haar_random_unitary(d^2)
        S, M = unitary_evol_two_site(M, U, site, "l", truncation = truncation, max_bd = max_bd, max_err = max_err)
        push!(S_list,S)
        S, M = canonicalize_left_one_site(M,site+1,truncation = truncation, max_bd = max_bd, max_err = max_err)
        push!(S_list,S)
    end
    return S_list,M
end

function entropy(S::Vector{Float64},n::Int=1)
    if(n==1)
        ps = (S.^2).+1E-12 
        EE = -sum(ps.*log.(ps))
    else
        EE = 1/(1-n)*log(sum(S.^(2*n)))
    end
    return EE
end

function expect_value_onsite(Ac::Array{<:Number,3},P::Matrix{<:Number})
    return ncon([Ac,P,conj.(Ac)],[[1,3,2],[3,4],[1,4,2]])[1]
end

function measurement_layer_spin_half_v2(M::myMPS{T}, p::Float64;truncation = false, max_bd = 1024, max_err = 1E-8) where T
    ### Apply Z basis measurement at each site with probability p
    ### Assuming left canonical form as input, and output right canonical form
    N = length(M)
    sigmaZ = diagm(0=>[1.0,-1.0])
    P_plus = (I+sigmaZ)./2.0
    P_minus = (I-sigmaZ)./2.0 #define the projectors here
    r_list = rand(N)
    for i in N:-1:1
        if(r_list[i]<p) #measuring!
            Ac = M.TensorList[i]
            pplus = real(expect_value_onsite(Ac, P_plus))
            pminus = real(expect_value_onsite(Ac, P_minus))
            if(abs(pplus+pminus-1)>1E-7)
                @warn "probability add not equal to one!"
                println(pplus+pminus)
            end
            r = rand()
            if(r<pplus) #projection onto up
                Anew = ncon([Ac,P_plus],[[-1,1,-3],[1,-2]])
            else #projection onto down
                Anew = ncon([Ac,P_minus],[[-1,1,-3],[1,-2]])
            end
            M.TensorList[i] = Anew
        end
        ~, M =canonicalize_right_one_site(M, i, truncation = truncation, max_bd = max_bd, max_err = max_err)
    end
    return M
end

function apply_projector_two_site(M::myMPS, P::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the pure state by U 
    ## Assuming U is two-site
    ## further assuming the two sites are the center of canonical form if truncation = true (important!)
    ## site is the first site # of the unitary (1 to N-1)
    ## the [site] of the two sites is put into left canonical form (if dir = "l")
    ## the [site+1] of the two sites is put into right canonical form (if dir = "r")
    ## return S and M
    myMPSTensors = M.TensorList
    d = phys_dim(M)
    @assert site<length(M)
    @assert size(P,1)==size(P,2)
    @assert size(P,1)==d^2
    A1 = myMPSTensors[site]
    A2 = myMPSTensors[site+1]
    DL = size(A1,1)
    DR = size(A2,3)
    P = reshape(P,(d,d,d,d)) # site 1 ket, site 2 ket, site 1 bra, site 2 bra
    A_evol = ncon([A1,A2,P],[[-1,1,2],[2,3,-4],[1,3,-2,-3]])
    A_evol_mat = reshape(A_evol,(DL*d,DR*d))
    U2 = nothing; S2=nothing; V2=nothing;
    try
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.QRIteration())
    end
    S2 = S2./norm(S2); #normalize the state
    if(truncation == true)
        set_bd = truncate(S2, max_bd, max_err)
        trunc_err = norm(S2[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S2 = S2[1:set_bd]
        U2 = U2[:,1:set_bd]
        V2 = V2[:,1:set_bd]
    end
    if(dir == "l")
        AL = reshape(U2, (DL,d,length(S2)))
        AR = reshape(diagm(0=>S2)*V2',(length(S2),d,DR))
    else
        AL = reshape(U2*diagm(0=>S2), (DL,d,length(S2)))
        AR = reshape(V2',(length(S2),d,DR))
    end
    M.TensorList[site] = AL
    M.TensorList[site+1] = AR    
    return S2, M
end

function svd_trunc(A_evol_mat::Matrix; truncation = false, max_bd = 1024, max_err=1E-10)
    U2 = nothing; S2=nothing; V2=nothing;
    try
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.QRIteration())
    end
    S2 = S2./norm(S2); #normalize the state
    if(truncation == true)
        set_bd = truncate(S2, max_bd, max_err)
        trunc_err = norm(S2[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S2 = S2[1:set_bd]
        U2 = U2[:,1:set_bd]
        V2 = V2[:,1:set_bd]
    end
    return S2,U2,V2
end

function apply_projector_three_site(M::myMPS, P::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the pure state by U 
    ## Assuming U is two-site
    ## further assuming the th433 sites are the center of canonical form if truncation = true (important!)
    ## site is the first site # of the unitary (1 to N-2)
    ## the [site,site+1] of the three sites are put into left canonical form (if dir = "l")
    ## the [site+2,site+1] of the three sites are put into right canonical form (if dir = "r") ! do not support 
    ## return S and M
    myMPSTensors = M.TensorList
    d = phys_dim(M)
    @assert site<length(M)
    @assert size(P,1)==size(P,2)
    @assert size(P,1)==d^3
    A1 = myMPSTensors[site]
    A2 = myMPSTensors[site+1]
    A3 = myMPSTensors[site+2]
    DL = size(A1,1)
    DR = size(A3,3)
    P = reshape(P,(d,d,d,d,d,d)) # site 1 ket, site 2 ket, site 3 ket, site 1 bra, site 2 bra, site 3 bra
    A_evol = ncon([A1,A2,A3,P],[[-1,4,1],[1,5,2],[2,6,-5],[4,5,6,-2,-3,-4]])
    A_evol_mat = reshape(A_evol,(DL*d,DR*d*d))
    S2,U2,V2 = svd_trunc(A_evol_mat)
    AL = reshape(U2, (DL,d,length(S2)))
    M.TensorList[site] = AL
    AR = reshape(diagm(0=>S2)*V2',(length(S2),d*d,DR))
    DL = size(AR,1)
    A_evol_mat = reshape(AR, (d*DL, d*DR));
    S2,U2,V2 = svd_trunc(A_evol_mat)
    AL = reshape(U2, (DL,d,length(S2)))
    AR = reshape(diagm(0=>S2)*V2',(length(S2),d,DR))
    M.TensorList[site+1] = AL    
    M.TensorList[site+2] = AR  
    return M
end

function apply_projector_one_site(M::myMPS, P::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the pure state by U 
    ## Assuming P is one-site
    ## further assuming the site is the center of canonical form if truncation = true (important!)
    ## the [site] is put into left canonical form (if dir = "l")
    ## the [site] is put into right canonical form (if dir = "r")
    ## return M
    myMPSTensors = M.TensorList
    d = phys_dim(M)
    @assert size(P,1)==size(P,2)
    @assert size(P,1)==d
    A1 = myMPSTensors[site]
    DL = size(A1,1)
    DR = size(A1,3)
    P = reshape(P,(d,d)) # site 1 ket, site 2 ket, site 1 bra, site 2 bra
    A_evol = ncon([A1,P],[[-1,1,-3],[1,-2]])
    M.TensorList[site] = A_evol
    if(dir == "l")
        S, M = canonicalize_left_one_site(M, site, truncation = truncation, max_bd = max_bd, max_err = max_err)
    else
        S, M = canonicalize_right_one_site(M, site, truncation = truncation, max_bd = max_bd, max_err = max_err)
    end
    return M
end

function expect_values_twosite(A1::Array{<:Number,3},A2::Array{<:Number,3},P::Vector{<:Matrix{<:Number}})
    ## Compute the expectation value of two site operators (not necessarily normalized) note: central CF assumed
    vs = zeros(length(P))
    d1 = size(A1,2)
    d2 = size(A2,2)
    for i in 1:length(P)
        op = reshape(P[i],(d1,d2,d1,d2))
        prob = ncon([A1,A2,conj.(A1),conj.(A2),op],[[4,2,1],[1,3,8],[4,5,6],[6,7,8],[2,3,5,7]])
        vs[i] = real(prob[1])
    end
    return vs
end

function samp_from_list(ps::Vector{Float64})
    ## Given a set of probability, sample out a single index 
    ## if sum(ps)<1, then there is a prob = 1-sum(ps) to return 0
    r = rand()
    setindex = 0
    for l in 1:length(ps)
        r = r-ps[l]
        if(r<=sqrt(eps(Float64)))
            setindex = l
            break
        end
    end
    return setindex
end

function product_state_init2(T::Type, d::Int, N::Int) 
    ## Initialize a product state |000000...>
    ## T - data type
    ## d - local dimension
    ## N - number of sites
    Ten_even = zeros(T,1,d,1)
    Ten_odd = zeros(T,1,d,1) 
    Ten_even[1,1,1] = 1.0
    Ten_odd[1,1,1] = 1.0
    myMPSTensors = Array{T,3}[]
    for i in 1:N
        if(i%2==0)
            push!(myMPSTensors, Ten_even)
        else
            push!(myMPSTensors, Ten_odd)
        end
    end
    return myMPS(myMPSTensors)
end

function canonicalize!(M::myMPS, dir = "l";truncation = true, max_bd = 1024, max_err = 1E-8)
    ## dir == "l" => output a left canonical form and EEs
    ## die == "r" => output a right canonical form and EEs
    EEs = zeros(length(M)-1)
    if(dir == "l")
        M = canonicalize_right(M)
        for i in 1:length(M)
            S,M = canonicalize_left_one_site(M, i, truncation = true, max_bd = max_bd, max_err = 1E-8)
            if(i<length(M))
                EEs[i] = entropy(S)
            end
        end
    else
        M = canonicalize_left(M)
        for i in length(M):-1:1
            S,M = canonicalize_right_one_site(M, i, truncation = true, max_bd = max_bd, max_err = 1E-8)
            if(i>1)
                EEs[i-1] = entropy(S)
            end
        end
    end
    return M, EEs
end

function apply_projector_two_site(M::myMPS, P::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the pure state by U 
    ## Assuming U is two-site
    ## further assuming the two sites are the center of canonical form if truncation = true (important!)
    ## site is the first site # of the unitary (1 to N-1)
    ## the [site] of the two sites is put into left canonical form (if dir = "l")
    ## the [site+1] of the two sites is put into right canonical form (if dir = "r")
    ## return S and M
    myMPSTensors = M.TensorList
    d = phys_dim(M)
    @assert site<length(M)
    @assert size(P,1)==size(P,2)
    @assert size(P,1)==d^2
    A1 = myMPSTensors[site]
    A2 = myMPSTensors[site+1]
    DL = size(A1,1)
    DR = size(A2,3)
    P = reshape(P,(d,d,d,d)) # site 1 ket, site 2 ket, site 1 bra, site 2 bra
    A_evol = ncon([A1,A2,P],[[-1,1,2],[2,3,-4],[1,3,-2,-3]])
    A_evol_mat = reshape(A_evol,(DL*d,DR*d))
    U2 = nothing; S2=nothing; V2=nothing;
    try
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.QRIteration())
    end
    S2 = S2./norm(S2); #normalize the state
    if(truncation == true)
        set_bd = truncate(S2, max_bd, max_err)
        trunc_err = norm(S2[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S2 = S2[1:set_bd]
        U2 = U2[:,1:set_bd]
        V2 = V2[:,1:set_bd]
    end
    if(dir == "l")
        AL = reshape(U2, (DL,d,length(S2)))
        AR = reshape(diagm(0=>S2)*V2',(length(S2),d,DR))
    else
        AL = reshape(U2*diagm(0=>S2), (DL,d,length(S2)))
        AR = reshape(V2',(length(S2),d,DR))
    end
    M.TensorList[site] = AL
    M.TensorList[site+1] = AR    
    return S2, M
end

function svd_trunc(A_evol_mat::Matrix; truncation = false, max_bd = 1024, max_err=1E-10)
    U2 = nothing; S2=nothing; V2=nothing;
    try
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.DivideAndConquer())
    catch
        U2,S2,V2 = svd(A_evol_mat,alg=LinearAlgebra.QRIteration())
    end
    S2 = S2./norm(S2); #normalize the state
    if(truncation == true)
        set_bd = truncate(S2, max_bd, max_err)
        trunc_err = norm(S2[set_bd+1:end])^2
        if(trunc_err>1E-6)
            println("truncation error:",trunc_err)
        end
        S2 = S2[1:set_bd]
        U2 = U2[:,1:set_bd]
        V2 = V2[:,1:set_bd]
    end
    return S2,U2,V2
end

function apply_projector_three_site(M::myMPS, P::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the pure state by U 
    ## Assuming U is two-site
    ## further assuming the th433 sites are the center of canonical form if truncation = true (important!)
    ## site is the first site # of the unitary (1 to N-2)
    ## the [site,site+1] of the three sites are put into left canonical form (if dir = "l")
    ## the [site+2,site+1] of the three sites are put into right canonical form (if dir = "r") ! do not support 
    ## return S and M
    myMPSTensors = M.TensorList
    d = phys_dim(M)
    @assert site<length(M)
    @assert size(P,1)==size(P,2)
    @assert size(P,1)==d^3
    A1 = myMPSTensors[site]
    A2 = myMPSTensors[site+1]
    A3 = myMPSTensors[site+2]
    DL = size(A1,1)
    DR = size(A3,3)
    P = reshape(P,(d,d,d,d,d,d)) # site 1 ket, site 2 ket, site 3 ket, site 1 bra, site 2 bra, site 3 bra
    A_evol = ncon([A1,A2,A3,P],[[-1,4,1],[1,5,2],[2,6,-5],[4,5,6,-2,-3,-4]])
    A_evol_mat = reshape(A_evol,(DL*d,DR*d*d))
    S2,U2,V2 = svd_trunc(A_evol_mat, truncation = truncation, max_bd = max_bd, max_err = max_err)
    AL = reshape(U2, (DL,d,length(S2)))
    M.TensorList[site] = AL
    AR = reshape(diagm(0=>S2)*V2',(length(S2),d*d,DR))
    DL = size(AR,1)
    A_evol_mat = reshape(AR, (d*DL, d*DR));
    S2,U2,V2 = svd_trunc(A_evol_mat, truncation = truncation, max_bd = max_bd, max_err = max_err)
    AL = reshape(U2, (DL,d,length(S2)))
    AR = reshape(diagm(0=>S2)*V2',(length(S2),d,DR))
    M.TensorList[site+1] = AL    
    M.TensorList[site+2] = AR  
    return M
end

function apply_projector_one_site(M::myMPS, P::Matrix, site::Int, dir = "l"; truncation = false, max_bd = 1024, max_err=1E-10)
    ## Evolve the pure state by U 
    ## Assuming P is one-site
    ## further assuming the site is the center of canonical form if truncation = true (important!)
    ## the [site] is put into left canonical form (if dir = "l")
    ## the [site] is put into right canonical form (if dir = "r")
    ## return M
    myMPSTensors = M.TensorList
    d = phys_dim(M)
    @assert size(P,1)==size(P,2)
    @assert size(P,1)==d
    A1 = myMPSTensors[site]
    DL = size(A1,1)
    DR = size(A1,3)
    P = reshape(P,(d,d)) # site 1 ket, site 2 ket, site 1 bra, site 2 bra
    A_evol = ncon([A1,P],[[-1,1,-3],[1,-2]])
    M.TensorList[site] = A_evol
    if(dir == "l")
        S, M = canonicalize_left_one_site(M, site, truncation = truncation, max_bd = max_bd, max_err = max_err)
    else
        S, M = canonicalize_right_one_site(M, site, truncation = truncation, max_bd = max_bd, max_err = max_err)
    end
    return M
end

function measurement_layer_part_system(M::myMPS, sites::UnitRange{Int};truncation = true, max_bd = 1024, max_err = 1E-8) 
    ## Let sites = siteL:siteR
    ## Assuming the input myMPS as central canonical form centered at siteR !!!!!
    ## perform measurement from siteR to siteL with probability 1
    ## record the site that is measured and entanglement between left and right after each measurement
    N = length(M)
    sigmaZ = diagm(0=>[1.0,-1.0])
    P_plus = (I+sigmaZ)./2.0
    P_minus = (I-sigmaZ)./2.0 #define the projectors here
    r_list = rand(N)
    siteLs = Int[]
    siteRs = Int[]
    ees = Float64[]
    for i in sites[end]:-1:sites[1]
        Ac = M.TensorList[i]
        pplus = real(expect_value_onsite(Ac, P_plus))
        pminus = real(expect_value_onsite(Ac, P_minus))
        if(abs(pplus+pminus-1)>1E-7)
            @warn "probability add not equal to one!"
        end
        r = rand()
        if(r<pplus) #projection onto up
            Anew = ncon([Ac,P_plus],[[-1,1,-3],[1,-2]])
        else #projection onto down
            Anew = ncon([Ac,P_minus],[[-1,1,-3],[1,-2]])
        end
        M.TensorList[i] = Anew
        S, M = canonicalize_right_one_site(M, i, truncation = truncation, max_bd = max_bd, max_err = max_err)
        ee = entropy(S)
        push!(siteRs,sites[end])
        push!(siteLs,i)
        push!(ees,ee)
    end
    return siteLs, siteRs, ees
end

function compute_eta(N, siteL, siteR)
    ## OBC, compute cross ratio [1:siteL-1] - [siteR+1:N]
    x1 = -pi/2+pi*(siteL-1)/N
    x2 = pi/2 - pi*(N-1-siteR)/N
    z1 = sin(x1)
    z2 = sin(x2)
    eta = ((z1+1)*(1-z2))/((1-z1)*(z2+1))
    return eta
end

function localizable_EE_fixed_sites(M::myMPS, sites::UnitRange{Int}; meas_iters::Int = 200)
    N = length(M)
    siteL = sites[1]
    siteR = sites[end]
    M0 = copy(M)
    M0 = canonicalize_left(M0)
    for j in length(M):-1:siteR+1
        ~,M0 = canonicalize_right_one_site(M0,j)
    end
    etas = nothing
    ees_iters = zeros(meas_iters, length(sites))
    for i in 1:meas_iters
        M_cp = copy(M0)
        siteLs,siteRs, ees = measurement_layer_part_system(M_cp,sites,truncation = true, max_bd = max_bd, max_err = max_err);
        etas = [compute_eta(N,siteLs[j],siteRs[j]) for j in 1:length(siteLs)]
        ees_iters[i,:] = ees;
    end
    ys = [mean(ees_iters[:,i]) for i in 1:length(sites)];
    return etas, ys
end