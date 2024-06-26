using Puffer 
using Flux
using Serialization
using DataFrames

"""data_file="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
coeffs_file= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
# load/ prep  training data
data = deserialize(joinpath("data", data_file))
coeffs = deserialize(joinpath("data", coeffs_file))

RHS = hcat(data[:, :RHS]...).|>Float32
μs  = hcat(data[:, :μs]...).|>Float32
P   = vcat(data[:, :pressure]...).|>Float32
low = -10000
high = -low
these = low .< (data[:,:reduced_freq] ./ data[:,:k] ) .< high
cnt = sum(these)


N = 64
inputs = zeros(Float32, N , 2, 4, cnt)

i = 1
for row in eachrow(data)
    if low < row.reduced_freq / row.k < high
        position = row.position  
        position[1,:] .-= minimum(position[1,:])
        x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]            
        x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
        inputs[:,:,:,i] = x_in
        i += 1
    end
end

#force, lift, thrust, power 
# perfs = zeros(4, size(data,1))
perfs = zeros(4, cnt)
ns = coeffs[1,:coeffs]|>size|>last
i = 1
motions = []
for r in eachrow(coeffs)
    # perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2/r.k)        
    # perfs[:,ns*(i-1)+1:ns*i] = perf
    if low < r.reduced_freq / r.k < high
        perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
        push!(motions, (r.wave,r.reduced_freq,r.k))
        i += 1
    end
    # perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
end

# dataloader = DataLoader((inputs=inputs, RHS=RHS[:,findall(these)], μs=μs[:,findall(these)], perfs=perfs, P=P[findall(these), :]'), batchsize=mp.batchsize, shuffle=true)    
"""

# test ways to sample the data for an approximator
num_samps = 8
nT = 500
samps = zeros(Float32, num_samps, 2, 4,  length(motions)*nT);

start = num_samps ÷ 2 # number of sample/2
stride = 64 ÷ num_samps
#traveling wave function

for (j,motion) in enumerate(motions)    
    sampler =  deepcopy(defaultDict)
    sampler[:Nt] = 100
    sampler[:N]  = num_samps  
    sampler[:kine] = motion[1]
    sampler[:f] = motion[2]    
    if motion[1] == :make_heave_pitch        
        sampler[:motion_parameters] = [motion[3], motion[4]]        
    else        
        sampler[:motion_parameters] = [0.1]        
        sampler[:k] = motion[3]
        sampler[:wave] = motion[1]
    end
    sampler[:ψ] = -π/2
    foil, flow = init_params(; sampler...)
    # (foil)(flow)
    image = []
    slice = ns*(j-1)+1:ns*j 
    ivs = inputs[stride:stride:end,:,3,slice]    
    te = zeros(Float32, nT)
    for i = 1:nT
        (foil)(flow)
        get_panel_vels!(foil, flow)
        te[i] = foil.foil[2,1]
        foil.col[1,:]  = foil.col[1,:] .- minimum(foil.foil[1,:])

        push!(image, reshape([foil.col' foil.panel_vel' foil.normals' ivs[:,:,i] ], (num_samps,2,4)))
    end
    @show motion
    samps[:,:,:,slice] = cat(image..., dims=4)
end


# plot(foil,marker=:circle,lw=0, label="sample")
# plot!(inputs[:,1,1,i], inputs[:,2,1,i], label="panels")
# plot!(inputs[2:4:end,1,1,i], inputs[2:4:end,2,1,i],marker=:square,lw=0, label="panels")
# plot!(ylims=(-0.3,0.3))

# plot(inputs[stride:stride:end,1,1,i], label="panels")
# plot!(foil.col[1,:], marker=:circle,lw=0, label="sample")


dl = DataLoader((approx = samps, RHS = RHS, μs = latentdata.data.μs,
                νs = latentdata.data.νs, βs = latentdata.data.βs,
                perfs = latentdata.data.perfs ), batchsize=1024*4, shuffle=true)
sd = dl|>first
W,H,C,N = size(samps)
upconv = Chain(Conv((4,2), C=>C, Flux.tanh, pad=SamePad()),                            
              Flux.flatten,             
              Dense(W*H*C, 64)) |> gpu
          

upconv(sd.approx|>gpu)

state = Flux.setup(ADAM(0.01), upconv)

losses = []
for epoch = 1:500
    for (x, y, _, _, _, _) in dl  
        x = x |> mp.dev
        y = y |> mp.dev
        ls = 0.0
        grads = Flux.gradient(upconv) do m
            # Evaluate model and loss inside gradient context:                
            ls = errorL2(m(x), y)              
        end
        Flux.update!(state, upconv, grads[1])
        push!(losses, ls)  # logging, outside gradient context
        
    end
    if epoch %  10 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end
out = upconv(samps[:,:,1:4,:]|>gpu) |>cpu
@show errorL2(out, RHS)
@show Flux.mse(out, RHS)

ers = norm.([x for x in eachcol(out .- RHS)], 2)./norm.([x for x in eachcol(RHS)], 2)
val, this = findmax(ers)
val, this = findmin(ers)
# this -= 1
this = rand(1:size(samps,4))
plot(RHS[:,this], label="RHS")
plot!(upconv(samps[:,:,1:4,this:this]|>gpu) |>cpu, seriestype=:scatter, label="model")


outo = B_DNN.layers[1](dataloader.data.inputs|>gpu)|>cpu
@show errorL2(outo, dataloader.data.RHS)
erso = norm.([x for x in eachcol(outo .- RHS)], 2)./norm.([x for x in eachcol(RHS)], 2)
val, this = findmax(erso)
val, this = findmin(erso)
# this -= 1
this = rand(1:size(images,4))
plot(RHS[:,this], label="RHS")
plot!(B_DNN.layers[1](dataloader.data.inputs[:,:,:,this:this]|>gpu)|>cpu, seriestype=:scatter, label="model")



pNN = Chain(Conv((4,2), 5=>5, Flux.tanh, pad = SamePad()),            
            Conv((2,2), 5=>2,  pad = SamePad()),
            Flux.flatten,   
            Dense(32,mp.layers[1],Flux.tanh),
            SkipConnection(Chain(Dense(mp.layers[1],mp.layers[1],Flux.tanh),
                                 Dense(mp.layers[1],mp.layers[1])),+),
            Dense(mp.layers[1],mp.layers[end],Flux.tanh),
            Dense(mp.layers[end],3))|>mp.dev  

pNNstate = Flux.setup(ADAM(0.001), pNN)
plosses = []


#these add the latent spaces to the image/sampled space
if size(samps,3) == 4
    latim = permutedims(cat(latentdata.data.νs, latentdata.data.βs, dims=3), (1,3,2));
    latim = reshape(latim, (size(latim)[1:2]...,1, size(latim,3)));
    samps = cat(samps, latim, dims=3);
end

pNN[1:4](samps[:,:,:,1:4]|>mp.dev)
pndata = DataLoader((images = samps, perfs = perfs, μs = μs, P = P ), batchsize=1024*4, shuffle=true)

for epoch = 1:1000
    for (ims, perf, μs, P) in pndata
        ims = ims |> mp.dev 
        P = P |> mp.dev       
        lift = perf[2,:] |> mp.dev
        thrust = perf[3,:] |> mp.dev
        Γs = μs |> mp.dev
        Γs = Γs[1,:] - Γs[end,:]
        pls = 0.0
        grads = Flux.gradient(pNN) do m
            phat = m[1:4](ims)
            pls = errorL2(m[5].layers(phat), P) #match pressure inside the network
            y = m(ims)
            # ls = errorL2(y, vcat(lift', thrust', Γs')) #lift
            pls += errorL2(y[2,:], thrust)#thrust
            pls += errorL2(y[1,:], lift) #lift
            pls += errorL2(y[3,:], Γs)
            # ls = errorL2(y[:], Γs)             
            
        end
        Flux.update!(pNNstate, pNN, grads[1])
        push!(plosses, pls)  # logging, outside gradient context
    end
    if epoch %  (mp.epochs /10) == 0
        println("Epoch: $epoch, Loss: $(plosses[end])")
    end
end
plosses
 

function find_worst(latentdata, dataloader, pNN;ns = 500)    
    perfs = latentdata.data.perfs
    lift = perfs[2,:] 
    thrust = perfs[3,:] 
    pres  = pNN[5].layers(pNN[1:4](pndata.data.images|>gpu))|>cpu
    Γs = latentdata.data.μs 
    Γs = Γs[1,:] - Γs[end,:]
    P = dataloader.data.P
        
    y = pNN(pndata.data.images|>gpu)|>cpu

    le = []
    te = []
    ge = []
    pe = []
    for i = 1:200
        slice = ns*(i-1)+1:ns*i
        push!(le, errorL2(y[1,slice], lift[slice]))
        push!(te, errorL2(y[2,slice], thrust[slice]))
        push!(ge, errorL2(y[3,slice], Γs[slice]) )
        push!(pe, errorL2(pres[:,slice], P[:,slice]))
    end
    (wl,wt,wg,wp) = [argmax(v) for v in [le, te, ge, pe]]
    wl,wt,wg,wp, y, pres
end

function plot_controller(y, lift, thrust, Γs, P, pres;prefix="",wh=nothing, name="", nsims=200, ns=500)
    
    wh= isnothing(wh) ? rand(1:nsims) : wh
    @show string(motions[wh])
    slice = ns*(wh-1)+1:ns*(wh)

    plot(lift[slice], lw=4,label="lift")
    a = plot!(y[1,slice], lw = 0, marker=:circle, label="model")
    title!(string(motions[wh]))
    # title!("h: $(round(h, digits=2)), θ: $(round(θ|>rad2deg, digits=1)|>Int), St: $(round(St, digits=2))")
    plot(thrust[slice], lw=4,label="thrust")
    b = plot!(y[2,slice], lw = 0, marker=:circle, label="model")
    plot(Γs[slice], lw=4,label="Γ")
    c = plot!(y[3,slice], lw = 0, marker=:circle,
     label="model")
    d = plot(P[:,slice], label="pressure",st=:contourf)
    title!("P")
    e = plot(pres[:,slice], label="model",st=:contourf)
    title!("model")
    f = plot(d,e, layout = (1,2))
    
    g = plot(a,b,c,f, layout = (4,1), size = (1200,1000))

    dir_name = prefix*join(["$(layer)->" for layer in mp.layers])[1:end-2]
    dir_path = joinpath("images", dir_name)
    savefig(joinpath(dir_path,name*"controller"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))    
end

(wl,wt,wg,wp, approx, pres) = find_worst(latentdata, pndata, pNN)
Γs = latentdata.data.μs 
Γs = Γs[1,:] - Γs[end,:]
lift = latentdata.data.perfs[2,:]|>cpu
thrust = latentdata.data.perfs[3,:]|>cpu
plot_controller(approx,lift, thrust,Γs, P, pres ;wh=wl, prefix=prefix, name="worst_lift")
plot_controller(approx,lift, thrust,Γs, P, pres ;wh=wt, prefix=prefix, name="worst_thrust")
plot_controller(approx,lift, thrust,Γs, P, pres ;wh=wg, prefix=prefix, name="worst_Γ")
plot_controller(approx,lift, thrust,Γs, P, pres ;wh=wp, prefix=prefix, name="worst_pressure")
plot_controller(approx,lift, thrust,Γs, P, pres ;wh=nothing, prefix=prefix, name="Random")

l2norm(x,y) = norm(x-y,2)
plot(le, label="lift")
plot!(te, label="thrust")
plot!(ge, label="Γ")
plot!(pe, label="pressure")

coeffs[[argmax(le), argmax(te), argmax(ge), argmax(pe)], [:wave, :k, :reduced_freq]]
coeffs[[argmin(le), argmin(te), argmin(ge), argmin(pe)], 
[:wave, :k, :reduced_freq]]


#can I make a network to use mu to perfs?
sP = dataloader.data.P |>size
sC = dataloader.data.perfs|>size

comp = Chain(Dense(sP[1],sP[1], Flux.tanh),
            #  Dense(sP[1],sP[1], Flux.tanh),
             Dense(sP[1], 2,))|>mp.dev
compstate = Flux.setup(ADAM(0.01), comp)
losses = []
for epoch = 1:1000
    for (_,_,_, perfs, P) in dataloader
        P = P |> mp.dev
        y = perfs[2:3,:] |> mp.dev
        ls = 0.0
        grads = Flux.gradient(comp) do m
            # Evaluate model and loss inside gradient context:                
            out = m(P)
            ls = errorL2(out[1,:], y[1,:])             
            ls += errorL2(out[2,:], y[2,:]) 
        end
        Flux.update!(compstate, comp, grads[1])
        push!(losses, ls)  # logging, outside gradient context
        
    end
    if epoch %  10 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end

begin
    wh= rand(0:199)
    ns = 501
    lift = perfs[2,:]
    thrust = perfs[3,:]
    Γs = μs
    Γs = Γs[1,:] - Γs[end,:]
    y = comp(dataloader.data.P|>gpu)|>cpu

    slice = ns*wh+1:ns*(wh+1)
    @show wave,k,r = coeffs[wh, [:wave, :k,:reduced_freq]]
    @show errorL2(y[:,slice], perfs[2:3,slice])
    @show Flux.mse(y[:,slice], perfs[2:3,slice])
    plot(perfs[2,slice], label="lift")
    a = plot!(y[1,slice], label="model")
    plot(perfs[3,slice], label="thrust")
    b = plot!(y[2,slice], lw = 1, marker=:circle, label="model")
    plot(a,b, layout = (2,1), size = (1200,800))
    # plot(Γs[slice], label="Γ")  
    # c = plot!(y[slice], label="model")
end

data_file="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
coeffs_file= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
hp_file ="single_swimmer_thetas_0.0_0.17453292_h_0.0_0.25_fs_0.25_4.0_h_p.jls"
hp_coeff = "single_swimmer_coeffs_thetas_0.0_0.17453292_h_0.0_0.25_fs_0.25_4.0_h_p.jls"
hp_file = "a0_single_swimmer_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
hp_coeff = "a0_single_swimmer_coeffs_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
data = deserialize(joinpath("data", data_file))
coeffs = deserialize(joinpath("data", coeffs_file))
h_p = deserialize(joinpath("data", hp_file))
h_p_coeffs = deserialize(joinpath("data", hp_coeff))

RHS = hcat(data[:, :RHS]..., h_p[:,:RHS]...).|>Float32
μs  = hcat(data[:, :μs]...,  h_p[:,:μs]...).|>Float32
P   = vcat(data[:, :pressure]...,h_p[:,:pressure]...)'.|>Float32

RHS = hcat(h_p[:,:RHS]...).|>Float32
μs  = hcat( h_p[:,:μs]...).|>Float32
P   = vcat(h_p[:,:pressure]...)'.|>Float32
cnt = size(RHS,2)
N = 64# mp.layers[1]

pad = 2
inputs = zeros(Float32, N + pad*2, 2, 4, cnt);

i = 1
# for row in eachrow(data)    
#     position = row.position .- minimum(row.position, dims=2) 

#     x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]    
#     x_in = vcat(x_in[end-pad+1:end,:,:,:],x_in,x_in[1:pad,:,:,:])
#     x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
#     inputs[:,:,:,i] = x_in
#     i += 1
# end
for row in eachrow(h_p)    
    position = row.position 
    position[1,:] .-= minimum(position[1,:])
    x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]    
    x_in = vcat(x_in[end-pad+1:end,:,:,:],x_in,x_in[1:pad,:,:,:])
    x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
    inputs[:,:,:,i] = x_in
    i += 1
end


#force, lift, thrust, power 
# perfs = zeros(4, size(data,1))
perfs = zeros(4, cnt)
# ns = coeffs[1,:coeffs]|>size|>last
ns = h_p_coeffs[1,:coeffs]|>size|>last
i = 1
motions = []
# for r in eachrow(coeffs)
#     # perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2/r.k)        
#     # perfs[:,ns*(i-1)+1:ns*i] = perf
#     if low < r.reduced_freq / r.k < high
#         perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
#         push!(motions, (r.wave,r.reduced_freq,r.k))
#         i += 1
#     end
#     # perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
# end
for r in eachrow(h_p_coeffs)
    # perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2/r.k)        
    # perfs[:,ns*(i-1)+1:ns*i] = perf
    @show r.θ, r.h
    perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
    f,a = fna[(r.θ, r.h, r.Strouhal)]
    push!(motions, (:make_heave_pitch, f, r.h, r.θ , r.Strouhal))
    i += 1

    # perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
end
dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs, P=P), batchsize=mp.batchsize, shuffle=true)    
inputs|>size
RHS|>size
μs|>size
perfs|>size
P|>size




  #these add the latent spaces to the image/sampled space
if size(controller,3) == 4
    latim = permutedims(cat(latentdata.data.νs, latentdata.data.βs, dims=3), (1,3,2));
    latim = reshape(latim, (size(latim)[1:2]...,1, size(latim,3)));
    latim = cat(controller|>mp.dev, latim, dims=3);
end

pndata = DataLoader((images = latim, perfs = latentdata.data.perfs, μs = latentdata.data.μs, P =dataloader.data.P ), batchsize=mp.batchsize*2, shuffle=true)
pNN = Chain(Conv((4,2), 5=>5, Flux.tanh, pad = SamePad()),            
            Conv((2,2), 5=>2,  pad = SamePad()),
            Flux.flatten,   
            #TODO: 32 is a magic number!
            Dense(mp.layers[end]*4,mp.layers[1],Flux.tanh),
            SkipConnection(Chain(Dense(mp.layers[1],mp.layers[1],Flux.tanh),
                                Dense(mp.layers[1],mp.layers[1])),+),
            Dense(mp.layers[1],mp.layers[end],Flux.tanh),
            Dense(mp.layers[end],3))|>mp.dev  

pNNstate = Flux.setup(ADAM(mp.η), pNN)

plosses = []
for epoch = 1:mp.epochs
    for (ims, perf, μs, P) in pndata
        ims = ims |> mp.dev 
        P = P |> mp.dev       
        lift = perf[2,:] |> mp.dev
        thrust = perf[3,:] |> mp.dev
        Γs = μs |> mp.dev
        Γs = Γs[1,:] - Γs[end,:]
        pls = 0.0
        grads = Flux.gradient(pNN) do m
            phat = m[1:4](ims)
            pls = errorL2(m[5].layers(phat), P) #match pressure inside the network
            y = m(ims)

            pls += errorL2(y[2,:], thrust)#thrust
            pls += errorL2(y[1,:], lift) #lift
            pls += errorL2(y[3,:], Γs)
            # ls = errorL2(y[:], Γs)             
            
        end
        Flux.update!(pNNstate, pNN, grads[1])
        push!(plosses, pls)  # logging, outside gradient context
    end
    if epoch %  (mp.epochs /10) == 0
        println("Epoch: $epoch, Loss: $(plosses[end])")
    end
end
plosses
begin
    wh= rand(1:252)
    @show string(motions[wh])
    slice = ns*(wh-1)+1:ns*(wh)
    perfs = latentdata.data.perfs
    lift = perfs[2,:] 
    thrust = perfs[3,:] 
    pres  = pNN[5].layers(pNN[1:4](pndata.data.images|>gpu))|>cpu
    Γs = latentdata.data.μs 
    Γs = Γs[1,:] - Γs[end,:]
    P = dataloader.data.P
        
    y = pNN(pndata.data.images|>gpu)|>cpu
    plot(lift[slice], lw=4,label="lift")
    a = plot!(y[1,slice], lw = 0, marker=:circle, label="model")
    title!(string(motions[wh]))
    # title!("h: $(round(h, digits=2)), θ: $(round(θ|>rad2deg, digits=1)|>Int), St: $(round(St, digits=2))")
    plot(thrust[slice], lw=4,label="thrust")
    b = plot!(y[2,slice], lw = 0, marker=:circle, label="model")
    plot(Γs[slice], lw=4,label="Γ")
    c = plot!(y[3,slice], lw = 0, marker=:circle,
    label="model")
    d = plot(P[:,slice], label="pressure",st=:contourf)
    title!("P")
    e = plot(pres[:,slice], label="model",st=:contourf)
    title!("model")
    f = plot(d,e, layout = (1,2))

    g = plot(a,b,c,f, layout = (4,1), size = (1200,1000))
end




#just pressure
pnn = Chain(Conv((4,2), 5=>5, Flux.tanh, pad = SamePad()),            
            Conv((2,2), 5=>4,  pad = SamePad()),            
            Flux.flatten,               
            Dense(mp.layers[end]*8,mp.layers[1],Flux.tanh),
            Dense(mp.layers[1],mp.layers[1],Flux.tanh),
            Dense(mp.layers[1],mp.layers[1]))|>mp.dev  
pnn(ims[:,:,:,1:4]|>mp.dev)
pnnstate = Flux.setup(ADAM(mp.η), pnn)

plosses = []
for epoch = 1:mp.epochs
    for (ims, _,_, P) in pndata
        ims = ims |> mp.dev 
        P = P |> mp.dev       

        pls = 0.0
        grads = Flux.gradient(pnn) do m
            phat = m(ims)
            pls = errorL2(phat, P) #match pressure inside the network            
        end
        Flux.update!(pnnstate, pnn, grads[1])
        push!(plosses, pls)  # logging, outside gradient context
    end
    if epoch %  (mp.epochs /10) == 0
        println("Epoch: $epoch, Loss: $(plosses[end])")
    end
end
plosses
begin
    wh= rand(1:200)
    @show string(motions[wh])
    slice = ns*(wh-1)+1:ns*(wh)
    
    pres  = pnn(pndata.data.images|>gpu)|>cpu
    P = dataloader.data.P[:,slice]
    clims = extrema(P)

    @show errorL2(pres[:,slice], P)    
    d = plot(P, label="pressure",st=:contourf, clims=clims)
    title!("P")
    e = plot(pres[:,slice], label="model",st=:contourf, clims=clims)
    title!("model")
    f = plot(d,e, layout = (1,2),size = (1200,800))

    
    
end