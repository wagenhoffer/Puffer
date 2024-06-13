#refactor DG and the rest of the code to be more modular and grabbed here
# until then make sure you have trained versions of the following networks
# L (G) , bAE <- moregreen.jl
# pNN (pressure network), and upconv <-controller_attempt.jl

using Puffer
include("hnp_nlsolve_strou.jl")
# θ,h = 0.17453292f0, 0.05f0
# f,a = fna[(θ,h, 0.2f0)]
test = deepcopy(defaultDict)
test[:N] = 64
test[:Nt] = 100
test[:Ncycles] = 1
test[:f] = 1.0
test[:Uinf] = 1
test[:kine] = :make_heave_pitch

test[:ψ] = -pi/2
test[:motion_parameters] = [0.05, 0.0]

function sample_field(foil::Foil;  dist = 2.0, N=10)
    n,t,l = norms(foil.edge)
    θ = atan(t[:,2]...)

    xs = LinRange(0.0, dist, N)             
    ys = LinRange(-0.25, 0.25, N)
    X = repeat(xs, 1, length(ys))[:]
    Y = repeat(ys',length(xs),1)[:]
    
    points = rotation(-θ) *  [X'; Y']
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
    fg, eg = Puffer.get_circulations(foil)
    Γs = [fg... (eg.*0.0)...]
    ps = [foil.foil foil.edge]
    uv += vortex_to_target(ps, field, Γs, flow)
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
    n = size(phi,1)÷5
    [phi[n*i+1:n*(i+1),:] for i=0:4]
end

begin
    foil.foil[:,1] = [-0.1  0.]'
    foil.foil[:,2] = [ 0.1  0.]'
    foil.μs   .= 0.0
    foil.σs   .= 0.0
    foil.μs[1] = 1.0
    foil.σs[1] = .0
    xs = LinRange(-1.0, 1.0, 100)
    ys = LinRange(-1.0, 1.0, 100)
    X = repeat(xs,1,100)[:]
    Y = repeat(ys',100,1)[:]
    xy = [X Y]'
    out1 = phi_field(xy,foil)    
    out1
    contourf(xs,ys,reshape(out1,100,100),levels=20, colormap = :coolwarm)
end

begin
    T = Float32
    ϵ = sqrt(eps(T))
    strouhals = LinRange{T}(0.2, 0.4, 7)
    td = LinRange{Int}(0, 10, 6)
    θs = deg2rad.(td).|> T    
    hs = LinRange{T}(0.0, 0.25, 6)
    # f,a = fna[(θ, h, strou)]
    num_samps = 64
    vels = []
    poss = []
    images = []
    phis = []
    G = inv(L|>cpu)|>gpu
    begin
        for h in hs[2:end-2], θ in θs[2:end-2], st in strouhals[2:end-2]
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
                if flow.n > 101
                    sf = sample_field(foil::Foil)
                    # sp = stencil_points(sf;ϵ=ϵ)
                    points = sf .+ foil.edge[:,end] .+flow.Δt
                    phi = phi_field(points,foil)
                    # (ori, xp, xm, yp, ym) = de_stencil(phi)
                    uv = b2f(points, foil, flow)
                    
                    get_panel_vels!(foil2,flow)
                    # vortex_to_target(sources, targets, Γs, flow)   
                    pos = deepcopy(foil.col')
                    zeropos = deepcopy(foil.col') .- minimum(pos,dims=1).*[1 0]
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
                    # ν = μAE[:encoder](foil.μs|>gpu)
                    # errorL2(μAE[:decoder](ν)|>cpu, foil.μs)
                    cont = reshape([pos foil2.normals'  iv' foil2.panel_vel' ], (num_samps,2,4,1))
                    # image = cat(cont|>mp.dev, cat(ν, β, dims=2), dims=3)

                    image = cat(cont|>mp.dev, cat(foil.μs, foil.σs   , dims=2), dims=3)
                    push!(poss,  points)
                    push!(vels,  uv)
                    push!(phis, phi)
                    push!(images, image)
                    plot(foil, wake)   
                    plot!(points[1,:], points[2,:], seriestype = :scatter)  
                end
       
            end
            # gif(movie, "./images/full.gif", fps = 30)
            # plot(foil, wake)
        end
    end
end


begin
    #NATE: This is working. We are making the flow field strictly as a function of the foil source/doublets
    # this is not using the edge panel influence 
    #more over this shows that finding the potential field can be then stenciled for velocity fields.
    xy,_, uv, uvb = grab_samp_wake(wake,foil,flow; nsamps = 200) 
    # sf = sample_field(foil::Foil)
    sp = stencil_points(xy;ϵ=ϵ)
    points = hcat(xy,sp)
    phi = phi_field(points,foil)
    (p0, pxp, pxm, pyp, pym) = de_stencil(phi)
    # uv1 = b2f(xy, foil, flow)
    # ϵ = sqrt(eps(Float32))
    # xp = deepcopy(xy)
    # xp[1,:] .+= ϵ
    # xm = deepcopy(xy)
    # xm[1,:] .-= ϵ
    # yp = deepcopy(xy)
    # yp[2,:] .+= ϵ
    # ym = deepcopy(xy)
    # ym[2,:] .-= ϵ

    # p0  = phi_field(xy,foil)
    # pxp = phi_field(xp,foil)
    # pxm = phi_field(xm,foil)
    # pyp = phi_field(yp,foil)
    # pym = phi_field(ym,foil)
    Δϕ = (pxp + pym + pxm + pyp - 4*p0)/ϵ^2
    u = (pxp - pxm)/(2ϵ)
    v = (pyp - pym)/(2ϵ)
    errorL2(u, uvb[1,:]), errorL2(v, uvb[2,:]), sum(abs2,Δϕ)/200
end


#work on body/foil to wake interactions
nsamps = poss |> size|>first 
dims, nts=  poss[1] |> size
vels |> size
images |> size

psamp = reshape(hcat(poss...), (2, 100, nsamps));
vsamp = reshape(hcat(vels...), (2, 100, nsamps));
phisamp = reshape(hcat(phis...),  (100, nsamps));
isamp = cat(images..., dims = 4);


# datar = DataLoader((wakepos = poss, body = images, vels = vels), batchsize = 16, shuffle = true)
datar = DataLoader((wakepos = psamp.|>Float32, 
                    body = isamp.|>Float32, 
                    vels = vsamp.|>Float32,
                    phis = phisamp), batchsize = 128, shuffle = true)


ϕ = Chain(Dense(64*4,64*2,Flux.tanh),        
          Dense(64*2,64,Flux.tanh),          
          Dense(64,1))|>gpu



# custom join layer
struct Join{T, F}
    combine::F
    paths::T
end
  
# allow Join(op, m1, m2, ...) as a constructor
Join(combine, paths...) = Join(combine, paths)
# Flux.@layer Join
Flux.@layer Join
(m::Join)(xs::Tuple) = m.combine(map((f, x) -> f(x), m.paths, xs)...)
(m::Join)(xs...) = m(xs)
Flux.trainable(m::Join) = (m.paths[1][2:end], m.paths[2][2:end])
# mj = Chain(
#         Join( +,
#                 x->Dense(8,64,Flux.tanh)(x[1:8,:]), 
#                 x->Dense(8,64,Flux.tanh)(x[9:16,:]),
#                 x->μAE[:decoder](x[17:24,:]),
#                 x->B_DNN.layers[:dec](x[25:32,:])),
#         Dense(64,64,Flux.tanh),          
#         Dense(64,1))|>gpu

# code for using latent space
branch = Chain(Join( +,
            Chain(x->μAE[:decoder](x), Dense(64,64,Flux.tanh), Dense(64,64,Flux.tanh)),
            Chain(x->B_DNN.layers[:dec](x),Dense(64,64,Flux.tanh), Dense(64,64,Flux.tanh))))|>gpu
x = branch((nus,bes).|>gpu)
#skip the DG values and just make sure it works
branch = Chain(Parallel(+, 
            Chain(Dense(64,64,Flux.tanh),Dense(64,64,Flux.tanh)),
            Chain(Dense(64,64,Flux.tanh),Dense(64,64,Flux.tanh))),
            Dense(64,64,Flux.tanh),
            Dense(64,64))|>gpu
trunk  = Chain(Dense(2,64,Flux.tanh),
                Dense(64,64, Flux.tanh),
                Dense(64,64))|>gpu
y = trunk(wx|>gpu)

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
plot()
for i = 1:128
    plot!(wake[1,:,i], wake[2,:,i],label="", st=:scatter)
end
plot!()
(wake, body, vels,phis)= datar|>first
wake|>size
# wx = wake[1,:,:]
# wy = wake[2,:,:]
wake = wake[:,:,1]
wy = wake[2,:,1]

nus = body[:,1,5,:]
bes = body[:,2,5,:]

xxx = branch(map(gpu, (nus,bes)))
yyy = trunk(wake|>gpu)
# onet(xs,ys) = hcat([sum(x*y',dims=1)' for (x,y) in zip(eachcol(xs),eachcol(ys))]...)
begin

    branch = Chain(Parallel(+, 
                Chain(Dense(64,64,Flux.tanh),Dense(64,64,Flux.tanh)),
                Chain(Dense(64,64,Flux.tanh),Dense(64,64,Flux.tanh))),
                Dense(64,64,Flux.tanh),
                Dense(64,64))|>gpu
    trunk  = Chain(Dense(2,64,Flux.tanh),
                    Dense(64,64, Flux.tanh),
                    Dense(64,64))|>gpu
    onet(xs,ys) = permutedims((xs'*ys),(2,1))
    onet(branch(map(gpu, (nus,bes))),trunk(wake|>gpu))
    Onet = Parallel( onet, branch, trunk)|>gpu
    Onet(map(gpu,(nus,bes)),wake|>gpu)|>size

    ostate = Flux.setup(Adam(0.001), Onet)

    wake = sample_field(foil::Foil) .- [1.0,0 ]#.+2.3*flow.Δt #make sure the foil isn't rotated
    wpx = (wake .+ [ϵ, 0])|>gpu
    wmx = (wake .- [ϵ, 0])|>gpu
    wpy = (wake .+ [0, ϵ])|>gpu
    wmy = (wake .- [0, ϵ])|>gpu
    wake = wake|>gpu
    ϵ  = cbrt(eps(Float32))
    _ϵ = inv(ϵ)
    ϕlosses = []
    for i=1:1000
        for (_, body, vels, phis) in datar
            # wake = wake |> gpu
            body = body |> gpu
            vels = vels |> gpu
            phis = phis |> gpu
            nb = size(wake,3)        

            nus = body[:,1,5,:]
            bes = body[:,2,5,:]

            ϕloss = 0.0
            grads = Flux.gradient(Onet) do m
                
                ϕa  = m(map(gpu,(nus,bes)),wake)
                ϕpx = m(map(gpu,(nus,bes)),wpx)
                ϕmx = m(map(gpu,(nus,bes)),wmx)
                ϕpy = m(map(gpu,(nus,bes)),wpy)
                ϕmy = m(map(gpu,(nus,bes)),wmy)

                Δϕ = (ϕpx + ϕmy + ϕmx + ϕpy - 4*ϕa)*_ϵ^2
                u  = (ϕpx - ϕmx)*2_ϵ
                v  = (ϕpy - ϕmy)*2_ϵ
                ϕloss  = errorL2(ϕa, phis)
                ϕloss += Flux.mse(Δϕ, 0.0)
                ϕloss = errorL2(u, vels[1,:,:])
                ϕloss += errorL2(v, vels[2,:,:])
                # ϕls += errorL2(atan.(v,u), atan.(vels[2,:]',vels[1,:]'))
            end
            Flux.update!(ostate, Onet, grads[1])
        
            push!(ϕlosses, ϕloss)
        end
        if i%10 == 0
            println("Epoch: $i, Loss: $(ϕlosses[end])")
        end
        
    end    
end


 begin
    which = rand(1:size(datar.data[1],3))
    
    # (wake, body, vels, phis) in datar

    field = datar.data.wakepos[:,:,which] |> gpu
    body  = datar.data.body[:,:,:,which] |> gpu
    vels  = datar.data.vels[:,:,which] |> gpu
    phis  = datar.data.phis[:,which] |> gpu
    wx    = field[1,:,1]
    wy    = field[2,:,1]

    nus = body[:,1,5,:]
    bes = body[:,2,5,:]
    
    ϕa  = Onet(map(gpu,(nus,bes)), wake)
    ϕpx = Onet(map(gpu,(nus,bes)), wpx)
    ϕmx = Onet(map(gpu,(nus,bes)), wmx)
    ϕpy = Onet(map(gpu,(nus,bes)), wpy)
    ϕmy = Onet(map(gpu,(nus,bes)), wmy)

    Δϕ = (ϕpx + ϕmy + ϕmx + ϕpy - 4*ϕa)*_ϵ^2
    u =  (ϕpx - ϕmx)*2_ϵ
    v =  (ϕpy - ϕmy)*2_ϵ
    @show errorL2(u, vels[1,:,:]), errorL2(v, vels[2,:,:]), sum(abs2,Δϕ), errorL2(ϕa, phis)
    wx = wx|>cpu
    wy = wy|>cpu
    vels = vels|>cpu
    u = u|>cpu
    v = v|>cpu

    plot(wx, wy, st=:scatter,label="")
    quiver!(wx, wy, quiver=(vels[1,:],vels[2,:]),label="true",color=:blue)
    quiver!(wx, wy, quiver=(u,v),label="pred",color=:red)
    # dist = sum(abs2,vels-[u; v]|>cpu)/sum(abs2,vels)
    # title!("Error: $dist")

end




































nw = size(wake,2)
body|>size
nb = size(body,4)
vels|>size
posx = body[:,1,1,:]
posy = body[:,2,1,:]


wake = wake|>gpu
wxs = hcat([wake[1,:,:]]...)
dxs = hcat([wake[1,:,bn]' .- posx[:,bn] for bn=1:nb]...)
dxp = hcat([(wake[1,:,bn].+ ϵ)' .- posx[:,bn]  for bn=1:nb]...)
dxm = hcat([(wake[1,:,bn].- ϵ)' .- posx[:,bn]  for bn=1:nb]...)
dys = hcat([wake[2,:,bn]' .- posy[:,bn] for bn=1:nb]...)
dyp = hcat([(wake[2,:,bn].+ ϵ)' .- posy[:,bn]  for bn=1:nb]...)
dym = hcat([(wake[2,:,bn].- ϵ)' .- posy[:,bn]  for bn=1:nb]...)
nus = [repeat(body[:,1,5,i],1, nw) for i=1:nb]|>gpu
bes = [repeat(body[:,2,5,i],1, nw) for i=1:nb]|>gpu
nus = hcat(nus...)
bes = hcat(bes...)
phis[:][101]
phis[:,1]|>size
hcat([p for p in eachcol(phis)]'...)[101]
# ϕ(vcat(dxs,dys,nus,bes)|>gpu)
# tv = reshape(vels,(2,reduce(*,size(vels)[2:3])))

ϵ = sqrt(eps(Float32))
# ϵ = 0.25
_ϵ = inv(ϵ)
ϕstate = Flux.setup(Adam(0.01), ϕ)
mjstate = Flux.setup(Adam(0.001), mj)
# mstate = Flux.setup(Adam(0.001), model)
ϕlosses =[]
for i=1:500
    for (wake, body, vels, phis) in datar
        wake = wake |> gpu
        body = body |> gpu
        vels = vels |> gpu
        phis = phis |> gpu
        vels = reshape(vels,(2,reduce(*,size(vels)[2:3])))    
        nw = size(wake,2)        
        nb = size(body,4)
        
        posx = body[:,1,1,:]
        posy = body[:,2,1,:]
        dxs = repeat(hcat([ wake[1,:,bn]' .- posx[:,bn] for bn=1:1]...), 1, nb)
        dys = repeat(hcat([ wake[2,:,bn]' .- posy[:,bn] for bn=1:1]...), 1, nb)
        #differentiation
        dxp = repeat(hcat([(wake[1,:,bn].+ ϵ)' .- posx[:,bn]  for bn=1:1]...), 1, nb)
        dxm = repeat(hcat([(wake[1,:,bn].- ϵ)' .- posx[:,bn]  for bn=1:1]...), 1, nb)
        dyp = repeat(hcat([(wake[2,:,bn].+ ϵ)' .- posy[:,bn]  for bn=1:1]...), 1, nb)
        dym = repeat(hcat([(wake[2,:,bn].- ϵ)' .- posy[:,bn]  for bn=1:1]...), 1, nb)
        # dxp = hcat([(wake[1,:,bn].+ ϵ)' .- posx[:,bn]  for bn=1:nb]...)
        # dxm = hcat([(wake[1,:,bn].- ϵ)' .- posx[:,bn]  for bn=1:nb]...)
        # dyp = hcat([(wake[2,:,bn].+ ϵ)' .- posy[:,bn]  for bn=1:nb]...)
        # dym = hcat([(wake[2,:,bn].- ϵ)' .- posy[:,bn]  for bn=1:nb]...)
        nus = [repeat(body[:,1,5,i],1, nw) for i=1:nb]
        bes = [repeat(body[:,2,5,i],1, nw) for i=1:nb]
        nus = hcat(nus...)
        bes = hcat(bes...)

        ϕloss = 0.0
        grads = Flux.gradient(mj) do m
        # grads = Flux.gradient(ϕ) do m
            # ϕa  = m(vcat(dxs,dys,nus,bes))
            # ϕpx = m(vcat(dxp,dys,nus,bes))
            # ϕmx = m(vcat(dxm,dys,nus,bes))
            # ϕpy = m(vcat(dxs,dyp,nus,bes))
            # ϕmy = m(vcat(dxs,dym,nus,bes))
            
            ϕa  = m((dxs,dys,nus,bes))
            ϕpx = m((dxp,dys,nus,bes))
            ϕmx = m((dxm,dys,nus,bes))
            ϕpy = m((dxs,dyp,nus,bes))
            ϕmy = m((dxs,dym,nus,bes))

            Δϕ = (ϕpx + ϕmy + ϕmx + ϕpy - 4*ϕa)*_ϵ^2
            u =  (ϕpx - ϕmx)*2_ϵ
            v =  (ϕpy - ϕmy)*2_ϵ
            
            ϕloss == errorL2(ϕa, phis[:]')
            ϕloss += Flux.mse(Δϕ, 0.0)
            ϕloss += errorL2(u, vels[1,:]')
            ϕloss += errorL2(v, vels[2,:]')
            # ϕls += errorL2(atan.(v,u), atan.(vels[2,:]',vels[1,:]'))
        end
        Flux.update!(mjstate, mj, grads[1])
        # Flux.update!(ϕstate, ϕ, grads[1])
        push!(ϕlosses, ϕloss)
    end
    if i%10 == 0
        println("Epoch: $i, Loss: $(ϕlosses[end])")
    end
    
end

begin
    bn = rand(1:size(datar.data.body,4))
    body = datar.data.body[:,:,:,bn]
    posx = body[:,1,1]
    posy = body[:,2,1]
    wx = datar.data.wakepos[1,:,bn]|>mp.dev
    wy = datar.data.wakepos[2,:,bn]|>mp.dev
    dxs = wx' .- posx
    dys = wy' .- posy
    dxp = (wx .+ ϵ)' .- posx
    dxm = (wx .- ϵ)' .- posx
    dyp = (wy .+ ϵ)' .- posy
    dym = (wy .- ϵ)' .- posy
    nus = repeat(body[:,1,5,:],1, nw)
    bes = repeat(body[:,2,5,:],1, nw) 
    # xp = ϕ(vcat(dxp,dys,nus,bes)|>gpu)
    # xm = ϕ(vcat(dxm,dys,nus,bes)|>gpu)
    # yp = ϕ(vcat(dxs,dyp,nus,bes)|>gpu)
    # ym = ϕ(vcat(dxs,dym,nus,bes)|>gpu)
    # Δϕ = (xp + ym + xm + yp - 4*ϕ(vcat(dxs,dys,nus,bes)|>gpu))/(ϵ^2)
    xp = mj((dxp,dys,nus,bes)|>gpu)
    xm = mj((dxm,dys,nus,bes)|>gpu)
    yp = mj((dxs,dyp,nus,bes)|>gpu)
    ym = mj((dxs,dym,nus,bes)|>gpu)
    Δϕ = (xp + ym + xm + yp - 4*mj((dxs,dys,nus,bes)|>gpu))/(ϵ^2)

    u = (xp-xm)/(2ϵ)|>cpu
    v = (yp-yp)/(2ϵ)|>cpu
    # out = mj((dxs,dys,nus,bes)|>gpu)
    tu = datar.data.vels[:,:,bn]
    
    # plot(tu[1,:],label="true u",st=:scatter)
    # plot!(mu[1,:],label="pred u",st=:scatter)
    wx = wx|>cpu
    wy = wy|>cpu
    plot(wx, wy, st=:scatter,label="")
    quiver!(wx, wy, quiver=(tu[1,:],tu[2,:]),label="true",color=:blue)
    quiver!(wx, wy, quiver=(u',v'),label="pred",color=:red)
    dist = sum(abs2,tu-[u; v]|>cpu)/sum(abs2,tu)
    title!("Error: $dist")


end

bs = body[:,:,1,4:4:end]

plot()
for i= 1:10
    plot!(bs[:,1,i],bs[:,2,i])
end
plot!()
###don't pre dx,dy
model = Chain(Dense(34,34, Flux.tanh),
              Dense(34,34,Flux.tanh),
              Dense(34,2))|>gpu

(wake, body, vels)= datar|>first
wake|>size
nw = size(wake,2)
body|>size
nb = size(body,1)
vels|>size
posx = [repeat(b[:,1,1,:],1,nw) for b in body]
posy = [repeat(b[:,2,1,:],1,nw) for b in body]
wxs = vcat([wake[1,:,bn] for bn=1:nb]...)
wys = vcat([wake[2,:,bn] for bn=1:nb]...)
nus = [repeat(b[:,1,5,:],1, nw) for b in body]
bes = [repeat(b[:,2,5,:],1, nw) for b in body]
posx = hcat(posx...)
posy = hcat(posy...)
nus = hcat(nus...)
bes = hcat(bes...)
model(vcat(posx,posy,wxs',wys',nus,bes)|>gpu)
tv = reshape(vels,(2,reduce(*,size(vels)[2:3])))


mstate = Flux.setup(Adam(0.01), model)

losses =[]
for i=1:1000
    for (wake, body, vels) in datar
        wake = wake |> gpu
        nw = size(wake,2)
        body = body |> gpu
        nb = size(body,1)
        vels = reshape(vels,(2,reduce(*,size(vels)[2:3]))) |> gpu
        posx = [repeat(b[:,1,1,:],1,nw) for b in body]
        posy = [repeat(b[:,2,1,:],1,nw) for b in body]
        wxs = vcat([wake[1,:,bn] for bn=1:nb]...)
        wys = vcat([wake[2,:,bn] for bn=1:nb]...)
        nus = [repeat(b[:,1,5,:],1, nw) for b in body]
        bes = [repeat(b[:,2,5,:],1, nw) for b in body]
        posx = hcat(posx...)
        posy = hcat(posy...)
        nus = hcat(nus...)
        bes = hcat(bes...)
        ols = 0.0f0
        grads = Flux.gradient(model) do m
            # pred = m(vcat(posx,posy,wxs',wys',nus,bes))
            ϕxp = ϕ(vcat(posx,posy,wxs',wys',nus,bes))
            ols = errorL2(pred[1,:], vels[1,:])
            ols += errorL2(pred[2,:], vels[2,:])
        end
        Flux.update!(mstate, model, grads[1])
        push!(losses, ols)
    end
    if i%100 == 0
        println("Epoch: $i, Loss: $(losses[end])")
    end
    
end

begin
    bn = rand(1:100)
    body = datar.data.body[:,:,:,bn]

    wx = datar.data.wakepos[1,:,bn]
    wy = datar.data.wakepos[2,:,bn]
    posx = repeat(body[:,1,1,:],1,nw)
    posy = repeat(body[:,2,1,:],1,nw)
    nus = repeat(body[:,1,5,:],1, nw)
    bes = repeat(body[:,2,5,:],1, nw) 

    mu = model(vcat(posx,posy,wx',wy',nus,bes)|>gpu)
    tu = datar.data.vels[:,:,bn]
    
    plot(tu[1,:],label="true u",st=:scatter)
    plot!(mu[1,:],label="pred u",st=:scatter)

    plot(wx, wy, st=:scatter,label="")
    quiver!(wx, wy, quiver=(tu[1,:],tu[2,:]),label="true")
    quiver!(wx, wy, quiver=(mu[1,:],mu[2,:]),label="pred")


end
vels|>size
"""
    add_vortex!(wake::Wake, foil::Foil)

Reduced order model for adding a vortex to the wake
"""
function add_vortex!(wake::Wake, foil::Foil, Γ, pos = nothing)
    if isnothing(pos)
        pos = (foil2.col[:, 1] + foil2.col[:, end])/2.0 + [0.05f0, 0]
    end        
    wake.xy = [wake.xy pos]
    wake.Γ = [wake.Γ..., Γ]
    # Set all back to zero for the next time step
    wake.uv = [wake.uv .* 0.0 [0.0, 0.0]]
    nothing
end
function cancel_buffer_Γ_rom!(wake::Wake, foil::Foil, Γ)
    shedtan = (foil2.tangents[:,end]-foil2.tangents[:,1])/2.0
    shedxy  = (foil2.foil[:,end]-foil2.foil[:,1])/2.0

    ϵ = 1e-3
    wake.xy[:, 1] = mean(foil.col, dims = 2) 
    # wake.xy[:, 2] = mean(foil.col, dims = 2) + [0, -ϵ]
    wake.Γ[1] = -sum(wake.Γ)-Γ
    # wake.Γ[1] = -Γ/2.0
    nothing
end


# begin
#     # images = []
#     G = inv(L|>cpu)|>gpu
#     num_samps = 8
#     stride = 64 ÷ num_samps
#     test[:N] = num_samps
#     test[:T] = Float32
#     test[:motion_parameters] = [0.0, 0.0]
#     foil2, _ = init_params(; test...)
#     # foil2.col[1,:] .-= 1.0
#     # foil2.foil[1,:] .-= 1.0
#     wake = Wake(foil2)
#     # add_vortex!(wake, foil2, 0.0)
#     (foil2)(flow)
#     # vels = []
#     ### EXAMPLE OF AN ANIMATION LOOP
#     movie = @animate for i in 1:(flow.Ncycles * flow.N)
    
#         (foil2)(flow)
#         get_panel_vels!(foil2,flow)
#         # vortex_to_target(sources, targets, Γs, flow)   
#         pos = deepcopy(foil2.col')
#         mx = minimum(pos[:,1])
#         mx = mx < 0.0 ? mx : -mx
#         pos[:,1] .+= mx
#         iv = vortex_to_target(wake.xy, foil2.col, wake.Γ, flow)

#         cont = reshape([pos foil2.panel_vel' foil2.normals' iv' ], (num_samps,2,4,1))

#         β = bAE[:encoder](upconv(cont|>gpu))
#         ν = G*β
#         image = cat(cont, cat(ν, β, dims=2), dims=3)
#         if i > 100
#             push!(images, image)
#         end
#         cl, ct, Γ = pNN(image|>gpu)
#         # setσ!(foil2, flow)
#         # add_vortex!(wake, foil2, Γ)
#         foil2.μ_edge[2] = foil2.μ_edge[1]
#         foil2.μ_edge[1] = Γ
#         # cancel_buffer_Γ_rom!(wake, foil2, Γ)
#         # body_to_wake!(wake, foil, flow)
#         wake_self_vel!(wake, flow)
#         eΓ = [-foil.μ_edge[1] foil.μ_edge[1]-foil.μ_edge[2] foil.μ_edge[2]]
#         wake.uv += vortex_to_target(foil2.edge, wake.xy, eΓ, flow)
#         dxs = wake.xy[1,:]' .- foil2.col[1,:]
#         dys = wake.xy[2,:]' .- foil2.col[2,:]

#         nw = size(wake.xy, 2)
#         wake.uv += model(vcat(dxs,dys,repeat(ν,1,nw), repeat(β,1,nw))|>gpu)|>cpu
#         move_wake!(wake, flow)
#         release_vortex!(wake, foil2)
#         plot(foil2, wake)            
#     end
#     gif(movie, "./images/samp.gif", fps = 30)
# end


function b2wNN(wake, foil, normals, bs, νs)
    x1, x2, y = panel_frame(wake, foil)
    nw, nb = size(x1)
    lexp = zeros((nw, nb))
    texp = zeros((nw, nb))
    yc = zeros((nw, nb))
    xc = zeros((nw, nb))
    β = atan.(-normals[1, :], normals[2, :])
    β = repeat(β, 1, nw)'
    @. lexp = log((x1^2 + y^2) / (x2^2 + y^2)) / (4π)
    @. texp = (atan(y, x2) - atan(y, x1)) / (2π)
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv = [xc * bs yc *bs ]'
    #doublets
    @. lexp =  (y/(x1^2  + y^2)  - y/(x2^2 + y^2))/2π
    @. texp = -(x1/(x1^2 + y^2) - x2/(x2^2 + y^2))/2π
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv += [xc * νs  yc * νs ]'
    uv
end
function b2wNN(wake, image)
    pos = image[:,:,1,:][:,:]'
    pos = hcat(pos, pos[:,1])
    b2wNN(wake.xy, pos, image[:,:,3,:][:,:]', image[:,2,5,:][:], image[:,1,5,:][:])
end
uv = b2w(wake,foil)
setσ!(foil2, flow)
foil.wake_ind_vel = vortex_to_target(wake.xy, foil2.col, wake.Γ, flow)
uv2 = b2w(wake,foil2)

begin
    scale = 1/pi
    plot(uv[1,:])
    a = plot!(uv2[1,:]./scale)
    plot(uv[2,:])
    b= plot!(uv2[2,:]./scale)
    plot(a,b)
end

