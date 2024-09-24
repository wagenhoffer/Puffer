using Plots
using Puffer
using Flux.Data: DataLoader
using Flux
using Random

using Statistics
using LinearAlgebra
using CUDA
using Serialization, DataFrames
using JLD2


struct ModelParams
    layers::Vector    
    η::Float64
    epochs::Int
    batchsize::Int
    lossfunc::Function
    dev
end

errorL2(x, y) = Flux.mse(x,y)/Flux.mse(y, 0.0) 

function build_ae_layers(layer_sizes, activation = tanh)    
    decoder = Chain([Dense(layer_sizes[i+1], layer_sizes[i], activation) 
                    for i = length(layer_sizes)-1:-1:2]...,
                    Dense(layer_sizes[2], layer_sizes[1]))    
    encoder = Chain([Dense(layer_sizes[i], layer_sizes[i+1], activation)
                    for i = 1:length(layer_sizes)-2]...,
                    Dense(layer_sizes[end-1], layer_sizes[end]))
    Chain(encoder = encoder, decoder = decoder)
end



function build_dataloaders(mp::ModelParams; data_file="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls", coeffs_file= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls")
    # load/ prep  training data
    data = deserialize(joinpath("data", data_file))
    coeffs = deserialize(joinpath("data", coeffs_file))
   
    RHS = hcat(data[:, :RHS]...).|>Float32
    μs  = hcat(data[:, :μs]...).|>Float32
    P   = vcat(data[:, :pressure]...)'.|>Float32
  
    N = mp.layers[1]
    cnt = size(RHS,2)
    pad = 2
    inputs = zeros(Float32, N + pad*2, 2, 4, cnt)
    
    i = 1
    for row in eachrow(data)  
        position        = row.position 
        position[1,:] .-= minimum(position[1,:] , dims=1) 
    
        x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]    
        x_in = vcat(x_in[end-pad+1:end,:,:,:],x_in,x_in[1:pad,:,:,:])
        x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
        inputs[:,:,:,i] = x_in
        i += 1
    end


    perfs = zeros(4, cnt)
    ns = coeffs[1,:coeffs]|>size|>last
    i = 1
    motions = []
    if "Strouhal" in names(coeffs)
        for r in eachrow(coeffs)        
            perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
            f,a = fna[(r.θ, r.h, r.Strouhal)]
            push!(motions, (:make_heave_pitch, f, r.h, r.θ , r.Strouhal))
            i += 1
        end
    else
        for r in eachrow(coeffs)        
            perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
            push!(motions, (r.wave,r.reduced_freq,r.k))
            i += 1        
        end
    end
    
    dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs, P=P), batchsize=mp.batchsize, shuffle=true)    
    dataloader, motions
end
function build_dataloaders_multi(mp::ModelParams; data_file="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls", coeffs_file= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls")
    # load/ prep  training data
    data = deserialize(joinpath("data", data_file))
    coeffs = deserialize(joinpath("data", coeffs_file))
   
    RHS = hcat(data[:, :RHS]...).|>Float32
    μs  = hcat(data[:, :μs]...).|>Float32
    P   = vcat(data[:, :pressure]...)'.|>Float32
  
    N = mp.layers[1]
    cnt = size(RHS,2)
    pad = 2
    inputs = zeros(Float32, N + pad*2, 2, 4, cnt)
    
    i = 1
    for row in eachrow(data)  
        position        = row.position 
        position[1,:] .-= minimum(position[1,:] , dims=1) 
    
        x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]    
        x_in = vcat(x_in[end-pad+1:end,:,:,:],x_in,x_in[1:pad,:,:,:])
        x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
        inputs[:,:,:,i] = x_in
        i += 1
    end


    perfs = zeros(4, cnt)
    ns = coeffs[1,:coeffs]|>size|>last
    i = 1
    motions = []
    if "Strouhal" in names(coeffs)
        for r in eachrow(coeffs)        
            perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
            f,a = fna[(r.θ, r.h, r.Strouhal)]
            push!(motions, (:make_heave_pitch, f, r.h, r.θ , r.Strouhal))
            i += 1
        end
    else
        for r in eachrow(coeffs)        
            perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
            push!(motions, (r.wave,r.reduced_freq,r.k))
            i += 1        
        end
    end
    
    dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs, P=P), batchsize=mp.batchsize, shuffle=true)    
    dataloader, motions
end
# W, H, C, Samples = size(dataloader.data.inputs)
function build_networks(mp; C = 4, N = 64)
    convenc = Chain(Conv((4,2), C=>C, Flux.tanh, pad=SamePad()),                
                    Conv((5,1), C=>C, Flux.tanh),
                    Conv((1,2), C=>1, Flux.tanh), 
                    x->reshape(x,(size(x,1), size(x,4))),
                    Dense(N=>N, Flux.tanh),
                    Dense(N=>N, Flux.tanh),
                    Dense(N=>N)) |>mp.dev

    convdec = Chain(Dense(N=>N, Flux.tanh),
                    Dense(N=>N, Flux.tanh),
                    Dense(N=>N, Flux.tanh),
                    x->reshape(x,(size(x,1), 1, 1, size(x,2))),
                    ConvTranspose((1,2), 1=>C, Flux.tanh),
                    ConvTranspose((5,1), C=>C, Flux.tanh),
                    ConvTranspose((4,2), C=>C,  pad=SamePad())) |> mp.dev                


    bAE = build_ae_layers(mp.layers) |> mp.dev
    μAE = build_ae_layers(mp.layers) |> mp.dev
    convNN = SkipConnection(Chain(convenc, convdec), .+)|>mp.dev
    perfNN = Chain( Conv((2,1), 2=>2, Flux.tanh, pad=SamePad()),
                Conv((2,1), 2=>1, pad = SamePad()),
                Flux.flatten,   
                Dense(mp.layers[end],mp.layers[end],Flux.tanh),
                Dense(mp.layers[end],mp.layers[end],Flux.tanh),
                Dense(mp.layers[end],3))|>mp.dev               

   bAE, μAE, convNN, perfNN
end

function build_states(mp, bAE, μAE, convNN, perfNN)
    μstate       = Flux.setup(Adam(mp.η), μAE)
    bstate       = Flux.setup(Adam(mp.η), bAE)
    convNNstate = Flux.setup(Adam(mp.η), convNN)
    perfNNstate = Flux.setup(Adam(mp.η), perfNN)
    μstate, bstate, convNNstate, perfNNstate
end

function train_AEs(dataloader, convNN, convNNstate, μAE, μstate, bAE, bstate, mp;ϵ=1e-3)
    # try to train the convNN layers for input spaces 
    convNNlosses = []    
    kick = false
    for epoch = 1:mp.epochs
        for (x, y,_,_,_) in dataloader        
            x = x |> mp.dev
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(convNN) do m
                # Evaluate model and loss inside gradient context:                
                ls = errorL2(m(x), x)  
                #TODO: increase the weight on the latent space loss
                # the resnet is not learning the latent space well
                latent = errorL2(m.layers[1](x),y)
                ls += latent
            end
            Flux.update!(convNNstate, convNN, grads[1])
            push!(convNNlosses, ls)  # logging, outside gradient context
            if ls < ϵ
                println("Loss is below the threshold. Stopping training at epoch: $epoch")
                kick = true
            end
        end
        if kick == true
            break
        end
        if epoch %  (mp.epochs /10) == 0
            println("Epoch: $epoch, Loss: $(convNNlosses[end])")
        end
    end
    # Train the solution AE - resultant μ
    μlosses = []
    kick = false
    for epoch = 1:mp.epochs
        for (_,_,y,_,_) in dataloader        
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(μAE) do m
                # Evaluate model and loss inside gradient context:                
                ls = errorL2(m(y), y)               
            end
            if ls < ϵ
                println("Loss is below the threshold. Stopping training at epoch: $epoch")
                kick = true
            end
            Flux.update!(μstate, μAE, grads[1])
            push!(μlosses, ls)  # logging, outside gradient context
        end
        if kick == true
            break
        end
        if epoch %  (mp.epochs /10) == 0
            println("Epoch: $epoch, Loss: $(μlosses[end])")
        end
    end

    blosses = []
    kick = false
    for epoch = 1:mp.epochs
        for (x, y,_,_,_) in dataloader        
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(bAE) do m
                # Evaluate model and loss inside gradient context:
                ls = errorL2(m(y), y) 
            end
            Flux.update!(bstate, bAE, grads[1])
            push!(blosses, ls)  # logging, outside gradient context
            if ls < ϵ
                println("Loss is below the threshold. Stopping training at epoch: $epoch")
                kick = true
            end
        end
        if epoch %  (mp.epochs /10) == 0
            println("Epoch: $epoch, Loss: $(blosses[end])")
        end
        if kick == true
            break
        end
    end
    B_DNN = SkipConnection(Chain(
        convenc = convNN.layers[1],
        enc = bAE[1],
        dec = bAE[2],
        convdec = convNN.layers[2]), .+) |> mp.dev
    B_DNN,(convNNlosses, μlosses, blosses)
end

function build_BDNN(convNN, bAE, mp)
    B_DNN = SkipConnection(Chain(
        convenc = convNN.layers[1],
        enc = bAE[1],
        dec = bAE[2],
        convdec = convNN.layers[2]), .+) |> mp.dev
    B_DNN
end

function build_L_with_data(mp, dataloader, μAE, B_DNN)
    L = rand(Float32, mp.layers[end],mp.layers[end])|>mp.dev
    νs = μAE[:encoder](dataloader.data.μs|>mp.dev)
    βs = B_DNN.layers[:enc](dataloader.data.RHS|>mp.dev)
    latentdata = DataLoader((νs=νs, βs=βs, μs=dataloader.data.μs, Bs=dataloader.data.RHS, perfs=dataloader.data.perfs),
         batchsize=mp.batchsize, shuffle=true)
    Lstate = Flux.setup(Adam(mp.η), L)|>mp.dev
    L, Lstate, latentdata    
end

function train_L(L, Lstate, latentdata, mp, B_DNN, μAE;ϵ=1e-3, linearsamps=128)    
    Glosses = []
    kick = false
    e3 = e2 = e4 = 0.0
    for epoch = 1:mp.epochs
        for (ν, β, μ, B, _) in latentdata
            # (ν, β, μ, B) =  first(latentdata)
            ν = ν |> mp.dev
            β = β |> mp.dev
            μ = μ |> mp.dev
            B = B |> mp.dev
            ls = 0.0
            superβ = deepcopy(β)
            superν = deepcopy(ν)
            superLν = deepcopy(β)
            
            grads = Flux.gradient(L) do m
                # Evaluate model and loss inside gradient context:            
                # @show f |> size
                # Lν = m(ν)
                Lν = m*ν
                
                ls = errorL2(Lν, β) 
                # superposition error            
                e2 = 0.0

                
                for ind = 1:linearsamps      
                    ind = rand(1:size(β,2))
                    superβ = β + circshift(β, (0, ind))
                    superν = ν + circshift(ν, (0, ind))
                    superLν = m*superν
                    e2 += errorL2(superLν, superβ)
                end
                e2 /= linearsamps

                e3 = errorL2(B_DNN.layers[:dec](Lν), B) 
                # # solve for \nu in L\nu = \beta                
                Gb = m\β
                decμ =  μAE[:decoder]((Gb))
                e4 = errorL2(decμ, μ)
                
                ls += e2 + e3 + e4
            end
        
            Flux.update!(Lstate, L, grads[1])
            push!(Glosses, ls)  # logging, outside gradient context
            if ls < ϵ
                println("Loss is below the threshold. Stopping training at epoch: $epoch")
                kick = true
            end
        end
        if epoch % 10 == 0
            println("Epoch: $epoch, Loss: $(Glosses[end])")
        end
        if kick == true
            break
        end
    end
    Glosses
end


function train_perfNN(latentdata, perfNN, perfNNstate, mp)
    perflosses = []
    
    for epoch = 1:mp.epochs
        for (ν,βs,μs,_,perf) in latentdata
            ν = ν |> mp.dev
            βs = βs |> mp.dev
            image = permutedims(cat(ν, βs, dims=3), (1,3,2))
            image = reshape(image, (1, size(image)...))
            lift = perf[2,:] |> mp.dev
            thrust = perf[3,:] |> mp.dev
            Γs = μs |> mp.dev
            Γs = Γs[1,:] - Γs[end,:]
            ls = 0.0
            grads = Flux.gradient(perfNN) do m
                # Evaluate model and loss inside gradient context:
                y = m(image)
                ls = errorL2(y[1,:], lift) #lift
                ls +=  10*errorL2(y[2,:], thrust)#thrust  
                ls += errorL2(y[3,:], Γs)             
             
            end
            Flux.update!(perfNNstate, perfNN, grads[1])
            push!(perflosses, ls)  # logging, outside gradient context
        end
        if epoch %  mp.epochs /10 == 0
            println("Epoch: $epoch, Loss: $(perflosses[end])")
        end
    end
    perflosses
end


function load_AEs(;μ="μAE.jld2", bdnn="B_DNN.jld2")
    path = joinpath("data", μ)
    μ_state = JLD2.load(path,"μ_state")
    Flux.loadmodel!(μAE, μ_state)
    path = joinpath("data", bdnn)
    bdnn_state = JLD2.load(path,"bdnn_state")
    Flux.loadmodel!(B_DNN, bdnn_state)
end

function load_L(;l="L.jld2")
    path = joinpath("data", l)
    l_state = JLD2.load(path,"l_state")
    Flux.loadmodel!(L, l_state)
end

function load_perfNN(;perf="perf_L64.jld2")
    path = joinpath("data", perf)
    perf_state = JLD2.load(path,"perf_state")
    Flux.loadmodel!(perfNN, perf_state)
end

function save_state(mp;prefix="")
    layer = join(["$(layer)->" for layer in mp.layers])[1:end-2]
    μ = prefix*"μAE_L$(layer).jld2"
    bdnn = prefix*"B_DNN_L$(layer).jld2"
    l = prefix*"L_L$(layer).jld2"
    perf = prefix*"perf_L$(layer).jld2"
    μ_state = Flux.state(μAE)|>cpu;
    bdnn_state = Flux.state(B_DNN)|>cpu;
    l_state = Flux.state(L)|>cpu;
    perf_state = Flux.state(perfNN)|>cpu;
    path = joinpath("data", μ)
    jldsave(path; μ_state)
    path = joinpath("data", bdnn)
    jldsave(path; bdnn_state)
    path = joinpath("data", l)
    jldsave(path; l_state)
end

function make_plots_and_save(mp, dataloader, latentdata, convNN, μAE, B_DNN, L, Glosses, μlosses, blosses, convNNlosses; wh = nothing, prefix = "")

    dir_name = prefix*join(["$(layer)->" for layer in mp.layers])[1:end-2]
    dir_path = joinpath("images", dir_name)

    if !isdir(dir_path)
        mkdir(dir_path)
    end
    wh = isnothing(wh) ? rand(1:size(dataloader.data[1],4)) : wh
    rhs = dataloader.data.RHS[:,wh]
    rhsa = B_DNN.layers[:convenc](dataloader.data.inputs[:,:,:,wh:wh]|>mp.dev)|>cpu

    plot(rhs[:], label = "Truth",lw=4,c=:red)
    plot!(rhsa[:], label="Approx", marker=:circle, ms=4,lw=0.25)
    title!("RHS : error = $(round(errorL2(rhsa|>cpu, rhs|>cpu) , digits=4))")
    
    # savefig(joinpath(dir_path,"conv_RHS"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))


    μ = dataloader.data.μs[:,wh]
    μa = μAE(μ|>mp.dev) |> cpu
    plot(μ, label = "Truth",lw=4,c=:red)
    a = plot!(μa,marker=:circle,lw=0, label="Approx",  ms=4)
    title!("μ : error = $(round(errorL2(μa|>cpu, μ|>cpu), digits=4))")
    savefig(joinpath(dir_path,"μAE"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))

    B = dataloader.data.RHS[:,wh]
    Ba = B_DNN.layers[2:3](B|>mp.dev)|>cpu
    plot(B, label="Truth", lw=4,c=:red)
    plot!(Ba, lw=0,label="Approx",marker=:circle, ms=4)
    title!("B : error = $(round(errorL2(Ba|>cpu, B|>cpu), digits=5)) ")
    savefig(joinpath(dir_path,"B_DNN"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))

                    
    x1 = reshape(B_DNN(dataloader.data.inputs[:,:,:,wh:wh]|>mp.dev)[:,:,1:4,:], 
                     (68,8))|>cpu         
    x2 = reshape(dataloader.data.inputs[:,:,1:4,wh:wh], (68,8))

    pos = plot(x1[:,1:2], label = "", lw=0.25, marker=:circle)
    plot!(pos, x2[:,1:2], label = "" ,lw=3)
    title!("Position")
    normals = plot(x1[:,3:4], label = "", lw=0.25, marker=:circle)
    plot!(normals, x2[:,3:4], label = "",lw=3)
    title!("Normals")
    wakevel = plot(x1[:,5:6], label = "", lw=0.25, marker=:circle)
    plot!(wakevel, x2[:,5:6], label = "",lw=3)
    title!("Wake Velocity")
    panelvel = plot(x1[:,7:8], label = "", lw=0.25, marker=:circle)
    plot!(panelvel, x2[:,7:8], label = "",lw=3)
    title!("Panel Velocity")

    b_DNN = B_DNN.layers[1](dataloader.data.inputs[:,:,:,wh:wh]|>mp.dev)|>cpu
    b_AE  = B_DNN.layers[1:3](dataloader.data.inputs[:,:,:,wh:wh]|>mp.dev)|>cpu
    b_True = dataloader.data.RHS[:,wh:wh]
    rhsPlot = plot(b_DNN, label="Convolution", lw=0.25, marker=:circle)
    plot!(rhsPlot, b_AE, label="AE", lw=0.25, marker=:circle)
    plot!(rhsPlot, b_True, label = "Truth", lw=3)
    x = plot(pos, normals, wakevel, panelvel, layout = (2,2), size =(1000, 800)) 
    plot(x, rhsPlot, layout = (2,1), size = (1200,800))
    savefig(joinpath(dir_path,"field_recon_and_RHS"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))


    G = inv(L|>cpu)
    recon_ν = G*(B_DNN.layers[:enc](latentdata.data.Bs[:,wh:wh]|>mp.dev)|>cpu)
    plot(latentdata.data.νs[:,wh]|>cpu, label="latent signal",lw=1,marker=:circle)
    a = plot!(recon_ν, label="Approx", marker=:circle, ms=1,lw=1)
    title!("ν : error = $(round(errorL2(recon_ν|>cpu, latentdata.data.νs[:,wh]|>cpu), digits=4))")
    plot(latentdata.data.μs[:,wh], label="μ Truth")
    b = plot!(μAE.layers.decoder(recon_ν|>mp.dev)|>cpu, label="μ DG ",lw=0, marker=:circle, ms=4)
    title!("μs : error = $(round(errorL2(μAE.layers.decoder(recon_ν|>mp.dev)|>cpu, latentdata.data.μs[:,wh]|>cpu), digits=4))")
    plot(a,b, size=(1200,800))
    savefig(joinpath(dir_path,"L_recon"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))



    e = plot(convNNlosses, yscale=:log10, title="ConvNN")
    a = plot(μlosses,      yscale=:log10, title="μAE")
    b = plot(blosses,      yscale=:log10, title="B_DNN")
    c = plot(Glosses,      yscale=:log10, title="L")

    plot(a,b,c,e, size=(1200,800))
    savefig(joinpath(dir_path,"losses"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))
end

function sampled_motion(motions, inputs; num_samps=8, nT=500)
    # test ways to sample the data for an approximator        
    samps = zeros(Float32, num_samps, 2, 4,  length(motions)*nT);

    stride = 64 ÷ num_samps    

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
        slice = nT*(j-1)+1:nT*j 
        #induced_velocities from input images
        #TODO: this is a hack to get the induced velocities;only works for        
        # ivs = inputs[stride÷2+1:stride:end,:,3,slice]    
        be = stride -1
        pts = collect(vcat(be:be:32,32+be-1:be:64))
        ivs = inputs[pts,:,3,slice]  
        
        for i = 1:nT
            (foil)(flow)
            get_panel_vels!(foil, flow)
            
            pos = deepcopy(foil.col)
            pos[1,:] .-= minimum(foil.foil[1,:])            

            push!(image, reshape([pos' foil.normals' ivs[:,:,i] foil.panel_vel'  ], (num_samps,2,4)))
        end
    
        samps[:,:,:,slice] = cat(image..., dims=4)
    end
    samps
end

function upconverter_pressure_networks(latentdata, controller, P, RHS, mp)
    dl = DataLoader((approx = controller, RHS = RHS ), batchsize=mp.batchsize, shuffle=true)
    
    W,H,C,N = size(controller)
    upconv = Chain(Conv((4,2), C=>C, Flux.tanh, pad=SamePad()),                            
                Flux.flatten,             
                Dense(W*H*C, 64)) |> mp.dev                

    upstate = Flux.setup(ADAM(mp.η), upconv)

    losses = []
    for epoch = 1:mp.epochs
        for (x, y,) in dl  
            x = x |> mp.dev
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(upconv) do m
                # Evaluate model and loss inside gradient context:                
                ls = errorL2(m(x), y)              
            end
            Flux.update!(upstate, upconv, grads[1])
            push!(losses, ls)  # logging, outside gradient context
            
        end
        if epoch %  10 == 0
            println("Epoch: $epoch, Loss: $(losses[end])")
        end
    end

    #these add the latent spaces to the image/sampled space
    if size(controller,3) == 4
        latim = permutedims(cat(latentdata.data.νs, latentdata.data.βs, dims=3), (1,3,2));
        latim = reshape(latim, (size(latim)[1:2]...,1, size(latim,3)));
        latim = cat(controller|>mp.dev, latim, dims=3);
    end

    pndata = DataLoader((images = latim, perfs = latentdata.data.perfs, μs = latentdata.data.μs, P = dataloader.data.P ), batchsize=mp.batchsize, shuffle=true)
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
    for epoch = 1:mp.epochs*4
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
    
                pls += errorL2(y[2,:], thrust)*10#thrust
                pls += errorL2(y[1,:], lift)*10 #lift
                pls += errorL2(y[3,:], Γs)
                
                
            end
            Flux.update!(pNNstate, pNN, grads[1])
            push!(plosses, pls)  # logging, outside gradient context
        end
        if epoch %  (mp.epochs /10) == 0
            println("Epoch: $epoch, Loss: $(plosses[end])")
        end
    end
    plosses
    upconv, upstate, pNN, pNNstate, losses, plosses, pndata
end

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
end

function plot_controller(;prefix="",wh=nothing, name="", nsims=200, ns=500)
    
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

function save_up_and_p(mp;prefix="")
    layer = join(["$(layer)->" for layer in mp.layers])[1:end-2]
    upconv_file = prefix*"upconv_L$(layer).jld2"
    pNN_file    = prefix*"pNN_L$(layer).jld2"
    
    up_state  = Flux.state(upconv)|>cpu;
    pNN_state = Flux.state(pNN)|>cpu;
    
    path = joinpath("data", upconv_file)
    jldsave(path; up_state)
    path = joinpath("data", pNN_file)
    jldsave(path; pNN_state)    
end