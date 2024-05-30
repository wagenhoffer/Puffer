#refactor DG and the rest of the code to be more modular and grabbed here
# until then make sure you have trained versions of the following networks
# L (G) , bAE <- moregreen.jl
# pNN (pressure network), and upconv <-controller_attempt.jl

using Puffer
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

function grab_samp_wake(wake::Wake; nsamps = 100)
    N = size(wake.xy, 2)
    idxs = rand(1:N, nsamps)
    xy = wake.xy[:, idxs]
    Γ = wake.Γ[idxs]
    uv = wake.uv[:, idxs]
    xy, Γ, uv
end
strouhals = LinRange{T}(0.2, 0.4, 7)
td = LinRange{Int}(0, 10, 6)
θs = deg2rad.(td).|> T    
hs = LinRange{T}(0.0, 0.25, 6)
# f,a = fna[(θ, h, strou)]
vels = []
poss = []
images = []
begin
    for h in hs[2:end-2], t in θs[2:end-2], st in strouhals[2:end-2]
        f,a = fna[(θ,h, st)]
        test = deepcopy(defaultDict)
        test[:N] = 64
        test[:Nt] = 100
        test[:Ncycles] = 1
        test[:f] = f
        test[:Uinf] = 1
        test[:kine] = :make_heave_pitch

        test[:ψ] = -pi/2
        test[:motion_parameters] = [h, t]
        foil, flow = init_params(; test...)
        wake = Wake(foil)
        test[:N] = num_samps
        test[:T] = Float32
        foil2, _ = init_params(; test...)
        (foil)(flow)
        (foil2)(flow)
        flow.n -= 1

        ### EXAMPLE OF AN ANIMATION LOOP
        movie = @animate for i in 1:200
            time_increment!(flow, foil, wake)
            # Nice steady window for plotting        
            # push!(vels, deepcopy(foil.panel_vel))
            (foil2)(flow)
            flow.n -= 1
        
            if i > 100
                xy,_, uv = grab_samp_wake(wake) 
                push!(poss, xy)
                push!(vels, uv)
                get_panel_vels!(foil2,flow)
                # vortex_to_target(sources, targets, Γs, flow)   
                pos = deepcopy(foil2.col')
                pos[:,1] .+= minimum(pos[:,1])
                iv = vortex_to_target(wake.xy, foil2.col, wake.Γ, flow)

                cont = reshape([pos foil2.panel_vel' foil2.normals' iv' ], (num_samps,2,4,1))

                β = bAE[:encoder](upconv(cont|>gpu))
                ν = G*β
                image = cat(cont, cat(ν, β, dims=2), dims=3)
                push!(images, image)
            end
            # plot(foil, wake)            
        end
        # gif(movie, "./images/full.gif", fps = 30)
        # plot(foil, wake)
    end
end

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

begin
    # images = []
    G = inv(L|>cpu)|>gpu
    num_samps = 8
    stride = 64 ÷ num_samps
    test[:N] = num_samps
    test[:T] = Float32
    test[:motion_parameters] = [0.0, 0.0]
    foil2, _ = init_params(; test...)
    # foil2.col[1,:] .-= 1.0
    # foil2.foil[1,:] .-= 1.0
    wake = Wake(foil2)
    # add_vortex!(wake, foil2, 0.0)
    (foil2)(flow)
    # vels = []
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:(flow.Ncycles * flow.N)
    
        (foil2)(flow)
        get_panel_vels!(foil2,flow)
        # vortex_to_target(sources, targets, Γs, flow)   
        pos = deepcopy(foil2.col')
        mx = minimum(pos[:,1])
        mx = mx < 0.0 ? mx : -mx
        pos[:,1] .+= mx
        iv = vortex_to_target(wake.xy, foil2.col, wake.Γ, flow)

        cont = reshape([pos foil2.panel_vel' foil2.normals' iv' ], (num_samps,2,4,1))

        β = bAE[:encoder](upconv(cont|>gpu))
        ν = G*β
        image = cat(cont, cat(ν, β, dims=2), dims=3)
        if i > 100
            push!(images, image)
        end
        cl, ct, Γ = pNN(image|>gpu)
        # setσ!(foil2, flow)
        # add_vortex!(wake, foil2, Γ)
        foil2.μ_edge[2] = foil2.μ_edge[1]
        foil2.μ_edge[1] = Γ
        # cancel_buffer_Γ_rom!(wake, foil2, Γ)
        # body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        eΓ = [-foil.μ_edge[1] foil.μ_edge[1]-foil.μ_edge[2] foil.μ_edge[2]]
        wake.uv += vortex_to_target(foil2.edge, wake.xy, eΓ, flow)
        dxs = wake.xy[1,:]' .- foil2.col[1,:]
        dys = wake.xy[2,:]' .- foil2.col[2,:]

        nw = size(wake.xy, 2)
        wake.uv += model(vcat(dxs,dys,repeat(ν,1,nw), repeat(β,1,nw))|>gpu)|>cpu
        move_wake!(wake, flow)
        release_vortex!(wake, foil2)
        plot(foil2, wake)            
    end
    gif(movie, "./images/samp.gif", fps = 30)
end

function b2w(wake,foil)
    x1, x2, y = panel_frame(wake.xy, foil.foil)
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
    # uv = [xc * foil.σs yc * foil.σs]'
    # #cirulatory effects    
    fg, eg = Puffer.get_circulations(foil)
    Γs = [fg... eg...]
    ps = [foil.foil foil.edge]
    uv = vortex_to_target(ps, wake.xy, Γs, flow)
    uv
end

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



#work on body/foil to wake interactions
nsamps = poss |> size|>first 
dims, nts=  poss[1] |> size
vels |> size
images |> size

psamp = reshape(hcat(poss...), (2, 100, nsamps))
poss[:,1:5,2]
# poss[2]
vsamp = reshape(hcat(vels...), (2,100,nsamps))
isamp = cat(images..., dims = 4)


# datar = DataLoader((wakepos = poss, body = images, vels = vels), batchsize = 16, shuffle = true)
datar = DataLoader((wakepos = psamp, 
                    body = isamp, 
                    vels = vsamp), batchsize = 128, shuffle = true)


model = Chain(Dense(32,32,Flux.tanh),
             Dense(32,32,Flux.tanh),
             Dense(32,32,Flux.tanh),
             Dense(32,2))|>gpu

model = Chain(Dense(32,32,tanh),                          
              Dense(32,64,tanh),             
              SkipConnection(x->μAE[:decoder](x[17:24]), .+), 
              Dense(64,64, tanh), 
              Dense(64,2))|>gpu

(wake, body, vels)= datar|>first
wake|>size
nw = size(wake,2)
body|>size
nb = size(body,4)
vels|>size
posx = body[:,1,1,:]
posy = body[:,2,1,:]
dxs = hcat([wake[1,:,bn]' .- posx[:,bn] for bn=1:nb]...)
dys = hcat([wake[2,:,bn]' .- posy[:,bn] for bn=1:nb]...)
nus = [repeat(body[:,1,5,i],1, nw) for i=1:nb]
bes = [repeat(body[:,2,5,i],1, nw) for i=1:nb]
nus = hcat(nus...)
bes = hcat(bes...)

model(vcat(dxs,dys,nus,bes)|>gpu)
tv = reshape(vels,(2,reduce(*,size(vels)[2:3])))


mstate = Flux.setup(Adam(0.001), model)
losses =[]
for i=1:500
    for (wake, body, vels) in datar
        wake = wake |> gpu
        body = body |> gpu
        vels = vels |> gpu
        vels = reshape(vels,(2,reduce(*,size(vels)[2:3])))
        wake|>size
        nw = size(wake,2)
        body|>size
        nb = size(body,4)
        vels|>size
        posx = body[:,1,1,:]
        posy = body[:,2,1,:]
        dxs = hcat([wake[1,:,bn]' .- posx[:,bn] for bn=1:nb]...)
        dys = hcat([wake[2,:,bn]' .- posy[:,bn] for bn=1:nb]...)
        nus = [repeat(body[:,1,5,i],1, nw) for i=1:nb]
        bes = [repeat(body[:,2,5,i],1, nw) for i=1:nb]
        nus = hcat(nus...)
        bes = hcat(bes...)
        # μs = AE[:decoder](nus)
        ols = 0.0
        grads = Flux.gradient(model) do m
            pred = m(vcat(dxs,dys,nus,bes))
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
    bn = rand(1:size(datar.data.body,4))
    body = datar.data.body[:,:,:,bn]
    posx = body[:,1,1]
    posy = body[:,2,1]
    wx = datar.data.wakepos[1,:,bn]
    wy = datar.data.wakepos[2,:,bn]
    dxs = wx' .- posx
    dys = wy' .- posy
    nus = repeat(body[:,1,5,:],1, nw)
    bes = repeat(body[:,2,5,:],1, nw) 

    mu = model(vcat(dxs,dys,nus,bes)|>gpu)|>cpu
    tu = datar.data.vels[:,:,bn]
    
    # plot(tu[1,:],label="true u",st=:scatter)
    # plot!(mu[1,:],label="pred u",st=:scatter)

    plot(wx, wy, st=:scatter,label="")
    quiver!(wx, wy, quiver=(tu[1,:],tu[2,:]),label="true")
    quiver!(wx, wy, quiver=(mu[1,:],mu[2,:]),label="pred")
    dist = sum(abs2,tu-mu|>cpu)/sum(abs2,tu)
    title!("Error: $dist")


end

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
            pred = m(vcat(posx,posy,wxs',wys',nus,bes))
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
    body = datar.data.body[bn]

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