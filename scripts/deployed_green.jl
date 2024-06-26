#refactor DG and the rest of the code to be more modular and grabbed here
# until then make sure you have trained versions of the following networks
# L (G) , bAE <- moregreen.jl
# pNN (pressure network), and upconv <-controller_attempt.jl

using Puffer
include("hnp_nlsolve_strou.jl")


function sample_field(foil::Foil;  dist = 2.0, N=10)
    n,t,l = norms(foil.edge)
    # cent  = mean(foil.foil, dims=2)
    te = foil.foil[:,end]
    θ = atan(t[:,2]...)

    xs = LinRange(-dist, dist, N) 
    ys = LinRange(-0.25, 0.25, N)
    X = repeat(xs, 1, length(ys))[:]
    Y = repeat(ys',length(xs),1)[:]
    
    points = (rotation(-θ) *  [X'; Y']) .+ te
end
function rotate_field(foil::Foil, field)
    _,t,_ = norms(foil.edge)
    # cent  = mean(foil.foil, dims=2)
    te = foil.foil[:,end]
    θ = atan(t[:,2]...)
    rotation(-θ) *  (field.- te) 
end
function stencil_points(points; ϵ=sqrt(eps(Float32)))
    xp = deepcopy(points)
    xp[1,:] .+= ϵ
    xm = deepcopy(points)
    xm[1,:] .-= ϵ
    yp = deepcopy(points)
    yp[2,:] .+= ϵ
    ym = deepcopy(points)
    ym[2,:] .-= ϵ
    hcat(xp, xm, yp, ym)
end

function grab_samp_wake(wake::Wake, foil,flow; nsamps = 75)
    N = size(wake.xy, 2)
    uv_body = b2f(wake.xy,foil, flow)
    if nsamps == size(wake.xy, 2)
        return deepcopy(wake.xy), deepcopy(wake.Γ), deepcopy(wake.uv), deepcopy(uv_body)
    end
    idxs = randperm(N)[1:nsamps]
    xy = wake.xy[:, idxs]
    Γ = wake.Γ[idxs]
    uv = wake.uv[:, idxs]    
    return xy, Γ, uv, uv_body[:,idxs]
end
function b2f(field,foil, flow)
    x1, x2, y = panel_frame(field, foil.foil)
    nw, nb = size(x1)
    lexp = zeros((nw, nb))
    texp = zeros((nw, nb))
    yc = zeros((nw, nb))
    xc = zeros((nw, nb))
    β = atan.(-foil.normals[1, :], foil.normals[2, :])
    β = repeat(β, 1, nw)'
    @. lexp = log((x1^2 + y^2) / (x2^2 + y^2)) / (4π)
    @. texp = (atan(y, x2) - atan(y, x1)) / (2π)
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv = [xc * foil.σs yc * foil.σs]'
    # #cirulatory effects      
    @. lexp = -(y/(x1^2 + y^2) -  y/(x2^2 + y^2))/2π
    @. texp = (x1/(x1^2 + y^2) - x2/(x2^2 + y^2))/2π
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv += [xc * foil.μs yc * foil.μs]'
    uv
end
function b2ff(field,foil, flow)
    x1, x2, y = panel_frame(field, foil.foil)
    nw, nb = size(x1)
    lexp = zeros((nw, nb))
    texp = zeros((nw, nb))
    yc = zeros((nw, nb))
    xc = zeros((nw, nb))
    β = atan.(-foil.normals[1, :], foil.normals[2, :])
    β = repeat(β, 1, nw)'
    @. lexp = log((x1^2 + y^2) / (x2^2 + y^2)) / (4π)
    @. texp = (atan(y, x2) - atan(y, x1)) / (2π)
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv = [xc * foil.σs yc * foil.σs]'
    # #cirulatory effects    
    @. lexp = -(y/(x1^2 + y^2) -  y/(x2^2 + y^2))/2π
    @. texp = (x1/(x1^2 + y^2) - x2/(x2^2 + y^2))/2π
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv += [xc * foil.μs yc * foil.μs]'
    x1, x2, y = panel_frame(field, foil.edge)
    nw, nb = size(x1)
    lexp = zeros((nw, nb))
    texp = zeros((nw, nb))
    yc = zeros((nw, nb))
    xc = zeros((nw, nb))
    tans, ns,_ = norms(foil.edge)
    β = atan.(-ns[1,:], ns[2, :])
    β = repeat(β, 1, nw)'
    @. lexp = -(y/(x1^2 + y^2) -  y/(x2^2 + y^2))/2π
    @. texp = (x1/(x1^2 + y^2) - x2/(x2^2 + y^2))/2π
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv += [xc * foil.μ_edge yc * foil.μ_edge]'
    uv
end

function phi_field(field,foil)
    x1, x2, y = panel_frame(field, foil.foil)    
    phis = zeros(size(x1))
    phid = zeros(size(x1))
    
    @. phis= x1*log(x1^2 + y^2) - x2*log(x2^2 + y^2) + 2*y*(atan(y, x2) - atan(y, x1)) 
    @. phid = (atan(y, x2) - atan(y, x1))

    phis*foil.σs/4π - phid*foil.μs/2π
end

function de_stencil(phi)
    n = size(phi,2)÷5
    [phi[:,n*i+1:n*(i+1)] for i=0:4]
end
output = load("wake_body_vels_epsilon_cbrtF32_$layerstr.jld2")
output = load("wake_body_vels_epsilon_sqrtF32_$layerstr.jld2")
# push!(poss,  points)
# push!(vels,  uv)
# push!(phis,  phi)
# push!(images, image)

isamp = output["body"]
psamp = output["wakepos"]
vsamp = output["vels"]
phisamp = output["phis"]

begin
    T = Float32
    ϵ = sqrt(eps(T))
    strouhals = LinRange{T}(0.2, 0.4, 7)
    td = LinRange{Int}(0, 10, 6)
    θs = deg2rad.(td).|> T    
    hs = LinRange{T}(0.0, 0.25, 6)
    # f,a = fna[(θ, h, strou)]
    num_samps = 8
    vels = []
    poss = []
    images = []
    phis = []
    G = inv(L|>cpu)|>gpu
    begin
        for h in hs[1:end], θ in θs[2:end], st in strouhals[1:end]
            # (θ,h, st) = fna.keys[1]
            f,a = fna[(θ,h, st)]
            test = deepcopy(defaultDict)
            test[:N] = 64
            test[:Nt] = 100
            test[:Ncycles] = 1
            test[:f] = f
            test[:Uinf] = 1
            test[:kine] = :make_heave_pitch

            test[:ψ] = -pi/2
            test[:motion_parameters] = [h, θ]
            foil, flow = init_params(; test...)
            wake = Wake(foil)
            test[:N] = num_samps
            test[:T] = Float32
            foil2, _ = init_params(; test...)
            (foil)(flow)
            (foil2)(flow)
            flow.n -= 1

            ### EXAMPLE OF AN ANIMATION LOOP
            # movie = @animate for i in 1:200
            for i = 1:200
                rhs = time_increment!(flow, foil, wake)
                # Nice steady window for plotting        
                # push!(vels, deepcopy(foil.panel_vel))
                (foil2)(flow)
                flow.n -= 1
                points = zeros(2,100)
                if flow.n > 100
                    sf = sample_field(foil::Foil)                    
                    sf = wake.xy[:, randperm(size(wake.xy, 2))[1:100]]
                    sp = stencil_points(sf;ϵ=ϵ)
                    points = sf #.+ foil.edge[:,end] .+flow.Δt
                    test = hcat(sf,sp)
                    phi = phi_field(test,foil)
                    (ori, xp, xm, yp, ym) = de_stencil(phi')
                    if sum((-4*ori + xp + xm + yp + ym) .> ϵ) >1
                        println("Error in stencil")
                    end
                    uv = b2f(points, foil, flow)
                    
                    get_panel_vels!(foil2,flow)
                    # vortex_to_target(sources, targets, Γs, flow)   
                    pos = deepcopy(foil2.col')
                    zeropos = deepcopy(foil2.col') .- minimum(pos,dims=1).*[1 0]
                    # pos[:,1] .+= minimum(pos[:,1])
                    # @show errorL2(dx,points[1,:]' .- pos[:,1])
                    iv = vortex_to_target(wake.xy, foil2.col, wake.Γ, flow)

                    cont = reshape([zeropos foil2.normals'  iv' foil2.panel_vel' ], (num_samps,2,4,1))

                    β = B_DNN.layers[2](upconv(cont|>gpu))
                    # B = B_DNN.layers[2:3](upconv(cont|>gpu))|>cpu
                    # excont = vcat(cont[end-1:end,:,:,:],cont,cont[1:2,:,:,:])
                    # β1 = B_DNN.layers[1:2](excont|>gpu)
                    # B1 = B_DNN.layers[1:3](excont|>gpu)|>cpu
                    ν = G*β
                
                    # errorL2(μAE[:decoder](ν)|>cpu, foil.μs)
                    cont = reshape([pos foil2.normals'  iv' foil2.panel_vel' ], (num_samps,2,4))
                    image = cat(cont|>mp.dev, cat(ν, β, dims=2), dims=3)

                    # image = cat(cont|>mp.dev, cat(foil.μs, foil.σs   , dims=2), dims=3)
                    push!(poss,  points)
                    push!(vels,  uv)
                    push!(phis,  phi)
                    push!(images, image)
                    # plot(foil, wake)   
                    # plot!(points[1,:], points[2,:], seriestype = :scatter)  
                end
       
            end
            # gif(movie, "./images/full.gif", fps = 30)
            # plot(foil, wake)
        end
    end
end


# begin
#     #NATE: This is working. We are making the flow field strictly as a function of the foil source/doublets
#     # this is not using the edge panel influence 
#     #more over this shows that finding the potential field can be then stenciled for velocity fields.
#     xy,_, uv, uvb = grab_samp_wake(wake,foil,flow; nsamps = 200) 
#     # sf = sample_field(foil::Foil)
#     sp = stencil_points(xy;ϵ=ϵ)
#     points = hcat(xy,sp)
#     phi = phi_field(points,foil)
#     (p0, pxp, pxm, pyp, pym) = de_stencil(phi)
#     # uv1 = b2f(xy, foil, flow)
#     # ϵ = sqrt(eps(Float32))
#     # xp = deepcopy(xy)
#     # xp[1,:] .+= ϵ
#     # xm = deepcopy(xy)
#     # xm[1,:] .-= ϵ
#     # yp = deepcopy(xy)
#     # yp[2,:] .+= ϵ
#     # ym = deepcopy(xy)
#     # ym[2,:] .-= ϵ

#     # p0  = phi_field(xy,foil)
#     # pxp = phi_field(xp,foil)
#     # pxm = phi_field(xm,foil)
#     # pyp = phi_field(yp,foil)
#     # pym = phi_field(ym,foil)
#     Δϕ = (pxp + pym + pxm + pyp - 4*p0)/ϵ^2
#     u = (pxp - pxm)/(2ϵ)
#     v = (pyp - pym)/(2ϵ)
#     errorL2(u, uvb[1,:]), errorL2(v, uvb[2,:]), sum(abs2,Δϕ)/200
# end


#work on body/foil to wake interactions
nsamps = poss |> size|>first 
# dims, nts=  poss[1] |> size
vels |> size
images |> size
images = images|>cpu
psamp = reshape(hcat(poss...), (2, 100, nsamps));
vsamp = reshape(hcat(vels...), (2, 100, nsamps));
phisamp = reshape(hcat(phis...),  (500, nsamps));
# isamp = cat(images..., dims = 4);
isamp = zeros(size(images[2])..., nsamps)
isamp|>size
for (i,im) in enumerate(images)
    isamp[:,:,:,i] .= im
end


# datar = DataLoader((wakepos = poss, body = images, vels = vels), batchsize = 16, shuffle = true)
#save all of the values for the dataloader into a DataFrame and then save it to a JLD2 file

jldsave("wake_body_vels_epsilon_sqrtF32_$layerstr.jld2"; wakepos = psamp, body = isamp, vels = vsamp, phis = phisamp)

datar = DataLoader((wakepos = psamp  .|>Float32, 
                    body    = isamp  .|>Float32, 
                    vels    = vsamp  .|>Float32,
                    phis    = phisamp.|>Float32), batchsize = 4096*2, shuffle = true)


# load_AEs(;μ=prefix*"μAE_L$(layerstr).jld2", bdnn=prefix*"B_DNN_L$(layerstr).jld2")
# mj = Chain(
#     Join( +,
#         Chain(Dense(mp.layers[end],mp.layers[1],Flux.tanh),Dense(64,64)),  
#         Chain(Dense(mp.layers[end],mp.layers[1],Flux.tanh),Dense(64,64)),
#         x->μAE[:decoder](x),
#         x->B_DNN.layers[:dec](x)),
#     Dense(mp.layers[1],mp.layers[1],Flux.tanh),
#     Dense(mp.layers[1],1))|>gpu


# mj(map(gpu,(dxs,dys,nus,bes)))                              
# plot()
# for i = 1:128
#     plot!(wake[1,:,i], wake[2,:,i],label="", st=:scatter)
# end
# plot!()
function l₂loss(𝐲̂, 𝐲)
    feature_dims = 1 #2:(ndims(𝐲) - 1)

    loss = sum(.√(sum(abs2, 𝐲̂ - 𝐲, dims = feature_dims)))
    y_norm = sum(.√(sum(abs2, 𝐲, dims = feature_dims)))

    return loss / y_norm
end
begin
   # Load sample data and push thru to make sure sizes are good
    (wake, body, vels,phis)= datar|>first
    wake|>size
    phis = phis|>gpu
    # wx = wake[1,:,:]
    # wy = wake[2,:,:]
    wake = wake[:,:,1]
    wy = wake[2,:,1]

    nus = body[:,1,5,:]
    bes = body[:,2,5,:]

    # xxx = branch(vcat(nus,bes)|>gpu)
    # yyy = trunk(wake|>gpu)

    # tester = Onet(vcat(nus,bes)|>gpu,xyphi|>gpu)#|>size
    # (ϕo, ϕpx, ϕmx, ϕpy, ϕmy) = de_stencil(tester')
    # ϕpx = m(vcat(nus,bes)|>gpu, wpx)
    # ϕmx = m(vcat(nus,bes)|>gpu, wmx)
    # ϕpy = m(vcat(nus,bes)|>gpu, wpy)
    # ϕmy = m(vcat(nus,bes)|>gpu, wmy)

    Δϕ = (ϕpx + ϕmy + ϕmx + ϕpy - 4*ϕo)/ϵ^2
    u  = (ϕpx - ϕmx)/2ϵ
    v  = (ϕpy - ϕmy)/2ϵ
    @show typeof(tester),typeof(phis)
    l₂loss(tester, phis|>gpu), errorL2(tester,phis|>gpu)
end
# onet(xs,ys) = hcat([sum(x*y',dims=1)' for (x,y) in zip(eachcol(xs),eachcol(ys))]...)

begin
    inlayer = layer[end]*2
    hidden  = 64
    out     = 64
    branch = Chain(Dense(inlayer, hidden, Flux.tanh; init=Flux.glorot_normal),
                    Dense(hidden, hidden, Flux.tanh; init=Flux.glorot_normal),                
                    Dense(hidden, hidden, Flux.tanh; init=Flux.glorot_normal),                
                    Dense(hidden, out,    Flux.tanh; init=Flux.glorot_normal))|>gpu
    trunk  = Chain(Dense(2,       hidden, Flux.tanh; init=Flux.glorot_normal),
                    Dense(hidden, hidden, Flux.tanh; init=Flux.glorot_normal),
                    Dense(hidden, hidden, Flux.tanh; init=Flux.glorot_normal),                
                    Dense(hidden, out,    Flux.tanh; init=Flux.glorot_normal))|>gpu
    onet(xs,ys) = permutedims((xs'*ys),(2,1))
    # onet(branch(vcat(nus,bes)|>gpu),trunk(wake|>gpu))
    Onet = Parallel( onet, branch, trunk)|>gpu
    # Onet(vcat(nus,bes)|>gpu,wake|>gpu)|>size
    # Onet(vcat(nus,bes)|>gpu,xyphi|>gpu)|>size

    ostate = Flux.setup(Adam(0.001), Onet)

    (θ,h, st) = fna.keys[1]
    f,a = fna[(θ,h, st)]
    test = deepcopy(defaultDict)
    test[:N] = 64
    test[:Nt] = 100
    test[:Ncycles] = 1
    test[:f] = f
    test[:Uinf] = 1
    test[:kine] = :make_heave_pitch

    test[:ψ] = -pi/2
    test[:motion_parameters] = [h, θ]
    foil, flow = init_params(; test...)
    
    
    ϵ  = sqrt(eps(Float32))
    wake = sample_field(foil::Foil) #.+ [1.0,0 ]#.+2.3*flow.Δt #make sure the foil isn't rotated
    # wpx = (wake .+ [ϵ, 0])|>gpu
    # wmx = (wake .- [ϵ, 0])|>gpu
    # wpy = (wake .+ [0, ϵ])|>gpu
    # wmy = (wake .- [0, ϵ])|>gpu
    wake = wake|>gpu
    xyphi = hcat(wake, stencil_points(wake;ϵ=ϵ))

    # _ϵ = inv(ϵ)
   
end
begin
    ϕlosses = []
    for i=1:500*8
        for (_, body, vels, phis) in datar
            # wake = wake |> gpu
            body = body |> gpu
            vels = vels |> gpu
            phis = phis |> gpu
            # nb = size(wake,3)        

            nus = body[:,1,5,:]
            bes = body[:,2,5,:]

            loss = 0.0
            grads = Flux.gradient(Onet) do m
                
                ϕa  = m(vcat(nus,bes)|>gpu, xyphi)
                (ϕo, ϕpx, ϕmx, ϕpy, ϕmy) = de_stencil(ϕa')
                # ϕpx = m(vcat(nus,bes)|>gpu, wpx)
                # ϕmx = m(vcat(nus,bes)|>gpu, wmx)
                # ϕpy = m(vcat(nus,bes)|>gpu, wpy)
                # ϕmy = m(vcat(nus,bes)|>gpu, wmy)

                Δϕ = (ϕpx + ϕmy + ϕmx + ϕpy - 4*ϕo)/ϵ^2
                u  = (ϕpx - ϕmx)/2ϵ
                v  = (ϕpy - ϕmy)/2ϵ
                loss  = l₂loss(ϕa, phis)*100
                loss += Flux.mse(Δϕ, 0.0)
                # loss += sum(abs.(Δϕ))
                loss += l₂loss(u', vels[1,:,:])*100
                loss += l₂loss(v', vels[2,:,:])*100

                # ϕls += errorL2(atan.(v,u), atan.(vels[2,:]',vels[1,:]'))
            end
            Flux.update!(ostate, Onet, grads[1])
        
            push!(ϕlosses, loss)
        end
        if i%10 == 0
            println("Epoch: $i, Loss: $(ϕlosses[end])")
        end
        
    end 
end

# save the Onet 
stamp = "sqrtF32_DELTA"
o_state = Flux.state(Onet)|>cpu;
path = joinpath("data", "Onet_L$(layerstr)_$(stamp).jld2")
jldsave(path; o_state)
# Load the Onet
path = joinpath("data", "Onet_L$(layerstr)_$(stamp).jld2")
# path = joinpath("data", perf)
perf_state = JLD2.load(path,"o_state")
Flux.loadmodel!(Onet, perf_state)

function nabla_phi(phi) 
    x0,xp,xm,yp,ym = de_stencil(phi)

    (-4.0*x0 + xp + xm + yp + ym)/ϵ^2
end

begin
    which = rand(1:size(datar.data[1],3))
    
    # (wake, body, vels, phis) in datar

    field = datar.data.wakepos[:,:,which] |> gpu
    body  = datar.data.body[:,:,:,which] |> gpu
    vels  = datar.data.vels[:,:,which] |> gpu
    phis  = datar.data.phis[:,which] |> gpu
    # wx    = field[1,:,1]
    # wy    = field[2,:,1]
    wake = sample_field(foil::Foil)
    wx = wake[1,:]
    wy = wake[2,:]
    wpx = (wake .+ [ϵ, 0])|>gpu
    wmx = (wake .- [ϵ, 0])|>gpu
    wpy = (wake .+ [0, ϵ])|>gpu
    wmy = (wake .- [0, ϵ])|>gpu
    wake = wake|>gpu

    nus = body[:,1,5,:]
    bes = body[:,2,5,:]
    ϕall= Onet(vcat(nus,bes)|>gpu, xyphi)
    (ϕa, ϕpx, ϕmx, ϕpy, ϕmy) = de_stencil(ϕall')
    # ϕa  = Onet(vcat(nus,bes)|>gpu, wake)
    # ϕpx = Onet(vcat(nus,bes)|>gpu, wpx)
    # ϕmx = Onet(vcat(nus,bes)|>gpu, wmx)
    # ϕpy = Onet(vcat(nus,bes)|>gpu, wpy)
    # ϕmy = Onet(vcat(nus,bes)|>gpu, wmy)
    # ϕall - vcat(ϕa,ϕpx,ϕmx,ϕpy,ϕmy)
    # @show errorL2(ϕall, vcat(ϕa,ϕpx,ϕmx,ϕpy,ϕmy))
    Δϕ = (ϕpx + ϕmy + ϕmx + ϕpy - 4*ϕa)/ϵ^2
    u =  (ϕpx - ϕmx)/2ϵ
    v =  (ϕpy - ϕmy)/2ϵ
    nabla_phi(phis')
    nabla_phi(ϕall')
    
    # ust = (phipx - phimx)/2ϵ
    # vst = (phipy - phimy)/2ϵ
    @show Flux.mse(Δϕ, 0.0), errorL2(ϕall, phis)
    # @show errorL2(ust', vels[1,:]), errorL2(vst', vels[2,:])
    wx = wx|>cpu
    wy = wy|>cpu
    vels = vels|>cpu
    u = u|>cpu
    v = v|>cpu

    @show errorL2(ϕall, phis), Flux.mse(ϕall, phis), l₂loss(ϕall', phis')
    stenerror = errorL2(ϕall, phis)
    x = unique(wx)
    y = unique(wy)
    
    clims = extrema(phis)
    nn = plot(x, y, reshape(ϕa|>cpu,(10,10)), c=:coolwarm,st=:heatmap, label="", aspect_ratio=:equal, clims=clims)
    plot!(nn,foil,c=:black)
    
    an = plot(x, y, reshape(phis[1:100]|>cpu,(10,10)), c=:coolwarm, st=:heatmap, label="", aspect_ratio=:equal, clims=clims)
    plot!(an,foil,c=:black)

    plot!(title="Stencil Error: $stenerror")
    er = plot(x, y, reshape(abs2.(ϕa'-phis[1:100])|>cpu,(10,10)), c=:coolwarm,st=:heatmap, label="", aspect_ratio=:equal)
    
    plot(an, nn, er, layout=(3,1), size=(900,650))    


    scl = [norm(vels[i,:],2) for i=1:2]
    plot(wx, wy, st=:scatter,label="")
    quiver!(wx, wy, quiver=(vels[1,:]./scl[1],vels[2,:]./scl[2]),label="true",color=:blue)
    quiver!(wx, wy, quiver=(u'./scl[1],v'./scl[2]),label="pred", color=:red)
    plot!(xlims = (-1,3), ylims = (-1,1))
    dist = sum(abs2,vels-vcat(u, v)|>cpu)/sum(abs2,vels)
    vfield = title!("Error: $dist")

    plot(an, nn, vfield, layout=(3,1), size=(900,650)) 
end

begin
    using ForwardDiff
    using Zygote

    fivept = xyphi[:,1:100:end]
    @time begin 
        ϕ5pt= Onet(vcat(nus,bes)|>gpu, fivept)|>cpu
        xypt = fivept[:,1:2]|>gpu
        ust = (ϕ5pt[2] - ϕ5pt[3])/2ϵ
        vst = (ϕ5pt[4] - ϕ5pt[5])/2ϵ
    end
    @time ForwardDiff.gradient(x->sum(Onet(vcat(nus,bes)|>gpu, x)), xypt)
    @time ForwardDiff.jacobian(x->Onet(vcat(nus,bes)|>gpu, x), xypt)
    @time Zygote.jacobian(x->sum(Onet(vcat(nus,bes)|>gpu, x)), xypt)
        using Zygote
        using Flux
        using Random

        Random.seed!(1234)

        d = 1
        u = Chain(
        Dense(d => 8, tanh),
        Dense(8 => 8, tanh),
        Dense(8 => 8, tanh),
        Dense(8 => 1)
        )
        ϵ = sqrt(eps(Float32))
        _ϵ = inv(first(ϵ[ϵ .!= zero(ϵ)]))
        ∇u(x) = Zygote.gradient(x -> sum(u(x)),x)[1]
        n∇u(x) = (u(x.+ϵ)-u(x.-ϵ))*_ϵ./2
        x = ones(Float32,d,1)
        @time ∇u(x)
        # -0.15050948
        @time n∇u(x)
end



begin
    num_samps = layer[end]
    G = inv(L|>cpu)|>gpu
    # (θ,h, st) = fna.keys[1]
    # (θ,h, st) = fna.keys[5]
    θ = θs[2]
    h = hs[2]
    st = strouhals[1]
    f,a = fna[(θ,h, st)]
    test = deepcopy(defaultDict)
    test[:N] = num_samps
    test[:Nt] = 32
    test[:Ncycles] = 5
    test[:f] = f
    test[:Uinf] = 1
    test[:kine] = :make_heave_pitch

    test[:ψ] = pi/2
    test[:motion_parameters] = [h, θ]
    foil, flow = init_params(; test...)
    wake = Wake(foil)
    cls = zeros(flow.N*flow.Ncycles)
    cts = zeros(flow.N*flow.Ncycles)
    (foil)(flow)
    movie = @animate for i in 1:flow.N*flow.Ncycles   
    # for i in 1:flow.N*flow.Ncycles   
        if flow.n != 1
            move_wake!(wake, flow)
            release_vortex!(wake, foil)
        end    
        (foil)(flow)                
        get_panel_vels!(foil,flow)
        # vortex_to_target(sources, targets, Γs, flow)   
        pos = deepcopy(foil.col')
        zeropos = deepcopy(foil.col') .- minimum(pos,dims=1).*[1 0]
        iv = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        cont = reshape([zeropos foil.normals'  iv' foil.panel_vel' ], (num_samps,2,4,1))

        β = B_DNN.layers[2](upconv(cont|>gpu))
        ν = G*β            
        
        # cont = reshape([pos foil.normals'  iv' foil.panel_vel' ], (num_samps,2,4,1))
        image = cat(cont|>mp.dev, cat(ν, β, dims=2), dims=3)
        cl,ct,Γ = pNN(image)|>cpu
        cls[i] = cl
        cts[i] = ct
        # push!(cls, cl)
        # push!(ct, ct)
        #TODO: CLEAN UP 
        # Γ = -Onet(vcat(ν,β), [0 1; 0 0]|>gpu)|>cpu|>first 
        # @show oG, Γ
        # Γ /= -2π
        
        # add_vortex!(wake, foil, -Γ, foil.edge[:,1])
        # cancel_buffer_Γ!(wake, foil)
        n,t,l = norms(foil.edge)
        # cent  = mean(foil.foil, dims=2)
        te = foil.foil[:,end]
        θedge = atan(t[1,2],t[2,2])
        # wake.xy[:,1] = foil.foil[:,1]
        # wake.xy = [wake.xy foil.foil[:,1]]
        wake.xy[:,1]  = foil.foil[:,1]
        sten = hcat(rotation(θedge)*(wake.xy.-te), 
            stencil_points(rotation(θedge)*(wake.xy.-te );ϵ=ϵ)).-[ϵ 0]'
        c,px,mx,py,my= de_stencil(Onet(vcat(ν,β), sten|>gpu)')        
        nΓ = (c|>cpu)[1]
        foil.μ_edge[2] = foil.μ_edge[1]
        foil.μ_edge[1] = -nΓ
        # wake.Γ = [wake.Γ..., nΓ]
        # @show nΓ, Γk
        # wake.uv = vcat((px - mx)/(2*ϵ),  (py - my)/(2*ϵ))
        # wake.uv += vortex_to_target(foil.edge[:,1:2], wake.xy,  -[nΓ -nΓ], flow)
        wake_self_vel!(wake, flow)
        # add_vortex!(wake, foil, -(c|>cpu)[1], foil.edge[:,1])
        plot(foil,wake)
    end
    gif(movie, "./images/full.gif", fps = test[:Nt]>30 ? 30 : test[:Nt])
    # plot(foil, wake)
    # plot(cls)
    # plot!(cts)
end
plot(cls)
plot!(cts)
function add_vortex!(wake::Wake, foil::Foil, Γ, pos = nothing, v = [0,0])
    if isnothing(pos)
        pos = (foil.col[:, 1] + foil.col[:, end])/2.0
    end        
    wake.xy = [wake.xy pos]
    wake.Γ = [wake.Γ..., Γ]
    # Set all back to zero for the next time step
    wake.uv = [wake.uv v]
    nothing
end

n,t,l = norms(foil.edge)
te  = foil.foil[:,end]
θ = atan(t[:,2]...)

xs = foil.foil[1,:]
ys = foil.foil[2,:] 
X = repeat(xs, 1, length(ys))[:]
Y = repeat(ys',length(xs),1)[:]

points = (rotation(θ) *  ([xs'; ys'].- te) )
plot(points[1,:], points[2,:], seriestype = :scatter)



plot(foil)
plot!(foil,st=:scatter)
be = 7#num_samps - 1 
stride  = 7
plot!(foil2.foil[1,pts], foil2.foil[2,pts], seriestype = :scatter)
plot!(foil2.foil[1,32+be-1:be:end], foil2.foil[2,32+be-1:be:end], seriestype = :scatter)
plot!(foil2.foil[1,be:be:end], foil2.foil[2,be:be:end], seriestype = :scatter)

pts = collect(vcat(be:stride:32,32+be-1:stride:64))