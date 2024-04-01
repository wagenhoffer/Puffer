using Plots
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

errorL2(x, y) = Flux.mse(x,y)/Flux.mse(y,0.0) 

function build_ae_layers(layer_sizes, activation = tanh)    
    decoder = Chain([Dense(layer_sizes[i+1], layer_sizes[i], activation) 
                    for i = length(layer_sizes)-1:-1:1]...)    
    encoder = Chain([Dense(layer_sizes[i], layer_sizes[i+1], activation)
                    for i = 1:length(layer_sizes)-1]...)
    Chain(encoder = encoder, decoder = decoder)
end



function build_dataloaders(mp::ModelParams; data_file="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls", coeffs_file= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls")
    # load/ prep  training data
    data = deserialize(joinpath("data", data_file))
    coeffs = deserialize(joinpath("data", coeffs_file))
   
    RHS = hcat(data[:, :RHS]...).|>Float32
    μs  = hcat(data[:, :μs]...).|>Float32
    P   = vcat(data[:, :pressure]...).|>Float32

    N = mp.layers[1]
    
    pad = 2
    inputs = zeros(Float32, N + pad*2, 2, 4, size(data,1))
    
    for (i,row) in enumerate(eachrow(data))       
        position = row.position .- minimum(row.position, dims=2) 
    
        x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]    
        x_in = vcat(x_in[end-pad+1:end,:,:,:],x_in,x_in[1:pad,:,:,:])
        x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
        inputs[:,:,:,i] = x_in
    end

    #force, lift, thrust, power 
    perfs = zeros(4, size(data,1))
    ns = coeffs[1,:coeffs]|>size|>last
    for (i,r) in enumerate(eachrow(coeffs))
        # perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2/r.k)
        # perfs[:,ns*(i-1)+1:ns*i] = perf
        perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
    end

    dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs, P=P'), batchsize=mp.batchsize, shuffle=true)    
end

# W, H, C, Samples = size(dataloader.data.inputs)
function build_networks(mp; C = 4, N = 64)
    convenc = Chain(Conv((4,2), C=>C, Flux.tanh, pad=SamePad()),                
                    Conv((5,1), C=>C, Flux.tanh),
                    Conv((1,2), C=>1, Flux.tanh), 
                    x->reshape(x,(size(x,1), size(x,4))),
                    Dense(N=>N, Flux.tanh),
                    Dense(N=>N, Flux.tanh)) |>mp.dev

    convdec = Chain(Dense(N=>N, Flux.tanh),
                    Dense(N=>N, Flux.tanh),
                    x->reshape(x,(size(x,1), 1, 1, size(x,2))),
                    ConvTranspose((1,2), 1=>C, Flux.tanh),
                    ConvTranspose((5,1), C=>C, Flux.tanh),
                    ConvTranspose((4,2), C=>C, Flux.tanh, pad=SamePad())) |> mp.dev                


    bAE = build_ae_layers(mp.layers) |> mp.dev
    μAE = build_ae_layers(mp.layers) |> mp.dev
    convNN = SkipConnection(Chain(convenc, convdec), .+)|>mp.dev
    perfNN = Chain(Dense(mp.layers[end],mp.layers[end],Flux.tanh),
                Dense(mp.layers[end],mp.layers[end],Flux.tanh),
                Dense(mp.layers[end],3,Flux.tanh))|>mp.dev               

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
        if epoch %  mp.epochs /10 == 0
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
        if epoch %  mp.epochs /10 == 0
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
        if epoch %  mp.epochs /10 == 0
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

function train_L(L, Lstate, latentdata, mp, B_DNN, μAE;ϵ=1e-3, linearsamps=100)    
    Glosses = []
    kick = false
    e3 = e2 = e4 = 0.0
    for epoch = 1:mp.epochs
        for (ν, β, μ, B,_) in latentdata
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

                
                for _ = 1:linearsamps      
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
        for (ν,_,μs,_,perf) in latentdata
            ν = ν |> mp.dev
            lift = perf[2,:] |> mp.dev
            thrust = perf[3,:] |> mp.dev
            Γs = μs |> mp.dev
            Γs = Γs[1,:] - Γs[end,:]
            ls = 0.0
            grads = Flux.gradient(perfNN) do m
                # Evaluate model and loss inside gradient context:
                y = m(ν)
                ls = errorL2(y[1,:], lift) #lift
                ls +=  errorL2(y[2,:], thrust)#thrust  
                ls += errorL2(y[3,:], Γs)             
                # ls = errorL2(y,[lift'; thrust'; Γs'])
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

function save_state(mp)
    layer = join(["$(layer)->" for layer in mp.layers])[1:end-2]
    μ = "μAE_L$(layer).jld2"
    bdnn = "B_DNN_L$(layer).jld2"
    l = "L_L$(layer).jld2"
    perf = "perf_L$(layer).jld2"
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
    path = joinpath("data", perf)
    jldsave(path; perf_state)
end

function make_plots_and_save(mp, dataloader, latentdata, convNN, μAE, B_DNN, L, perfNN, Glosses, μlosses, blosses, convNNlosses)

    dir_name = join(["$(layer)->" for layer in mp.layers])[1:end-2]
    dir_path = joinpath("images", dir_name)

    if !isdir(dir_path)
        mkdir(dir_path)
    end
    which = rand(1:size(dataloader.data[1],4))
    rhs = dataloader.data.RHS[:,which]
    rhsa = B_DNN.layers[:convenc](dataloader.data.inputs[:,:,:,which:which]|>mp.dev)
    plot(rhs, label = "Truth",lw=4,c=:red)
    plot!(rhsa, label="Approx", marker=:circle, ms=1,lw=0.25)
    title!("RHS : error = $(errorL2(rhsa|>cpu, rhs|>cpu))")
    # savefig(joinpath(dir_path,"conv_RHS"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))


    μ = dataloader.data.μs[:,which]
    μa = μAE(μ|>mp.dev)
    μa = μAE(μ|>mp.dev)
    plot(μ, label = "Truth",lw=4,c=:red)
    a = plot!(μa,marker=:circle,lw=0, label="Approx",  ms=1)
    title!("μ : error = $(errorL2(μa|>cpu, μ|>cpu))")
    savefig(joinpath(dir_path,"μAE"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))

    B = dataloader.data.RHS[:,which]
    Ba = B_DNN.layers[2:3](B|>mp.dev)
    plot(B, label="Truth")
    a = plot!(Ba, marker=:circle,lw=0,label="Approx")
    title!("B : error = $(errorL2(Ba|>cpu, B|>cpu))")
    savefig(joinpath(dir_path,"B_DNN"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))

                    
    x1 = reshape(B_DNN(dataloader.data.inputs[:,:,:,which:which]|>mp.dev)[:,:,1:4,:], 
                     (68,8))|>cpu         
    x2 = reshape(dataloader.data.inputs[:,:,1:4,which:which], (68,8))

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
    b_DNN = B_DNN.layers[1](dataloader.data[1][:,:,:,which:which]|>mp.dev)
    b_AE  = B_DNN.layers[1:3](dataloader.data[1][:,:,:,which:which]|>mp.dev)
    b_True = dataloader.data[2][:,which:which]
    rhsPlot = plot(b_DNN, label="Convolution", lw=0.25, marker=:circle)
    plot!(rhsPlot, b_AE, label="AE", lw=0.25, marker=:circle)
    plot!(rhsPlot, b_True, label = "Truth", lw=3)
    x = plot(pos, normals, wakevel, panelvel, layout = (2,2), size =(1000, 800)) 
    plot(x, rhsPlot, layout = (2,1), size = (1200,800))
    savefig(joinpath(dir_path,"field_recon_and_RHS"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))


    G = inv(L|>cpu)
    recon_ν = G*(B_DNN.layers[:enc](latentdata.data.Bs[:,which:which]|>mp.dev)|>cpu)
    plot(latentdata.data.νs[:,which], label="latent signal",lw=1,marker=:circle)
    a = plot!(recon_ν, label="Approx", marker=:circle, ms=1,lw=1)
    title!("ν : error = $(errorL2(recon_ν|>cpu, latentdata.data.νs[:,which]|>cpu))")
    plot(latentdata.data.μs[:,which], label="μ Truth")
    b = plot!(μAE.layers.decoder(recon_ν|>mp.dev), label="μ DG ")
    title!("μs : error = $(errorL2(μAE.layers.decoder(recon_ν|>mp.dev)|>cpu, latentdata.data.μs[:,which]|>cpu))")
    plot(a,b, size=(1200,800))
    savefig(joinpath(dir_path,"L_recon"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))

    which = rand(1:199)
    ns = 501
    c = plot(latentdata.data.νs[:,ns*which+1:ns*(which+1)], st=:contour, label="input")
    plot!(c, colorbar=:false)
    title!("Latent")
    plot(latentdata.data.perfs[1,ns*which+1:ns*(which+1)], label="Truth")
    a = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[2,:], label="Approx", marker=:circle, ms=1)
    title!("Lift")
    plot(latentdata.data.perfs[2,ns*which+1:ns*(which+1)], label="Truth")
    b = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[3,:], label="Approx", marker=:circle, ms=1)
    title!("Thrust")
    plot(c,a,b, layout = (3,1), size = (1200,800))
    savefig(joinpath(dir_path,"perfNN"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))

    e = plot(convNNlosses, yscale=:log10, title="ConvNN")
    a = plot(μlosses,      yscale=:log10, title="μAE")
    b = plot(blosses,      yscale=:log10, title="B_DNN")
    c = plot(Glosses,      yscale=:log10, title="L")
    d = plot(perflosses,   yscale=:log10, title="perfNN")
    plot(a,b,c,d,e, size=(1200,800))
    savefig(joinpath(dir_path,"losses"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))
end

layers = [[64,32,16,8]]
          [64,64,64],
          [64,64,32],
          [64,32,16],
          [64,64,16],          
          [64,32,16,8,4],
          [64,32,16,8,4,2]]


for layer in layers            
    @show layer
    mp = ModelParams(layer,  0.01, 50, 4096, errorL2, gpu)
    bAE, μAE, convNN, perfNN = build_networks(mp)
    μstate, bstate, convNNstate, perfNNstate = build_states(mp, bAE, μAE, convNN, perfNN)
    dataloader = build_dataloaders(mp)
    @show "Train Conv and AEs"
    B_DNN,(convNNlosses, μlosses, blosses) = train_AEs(dataloader, convNN, convNNstate, μAE, μstate, bAE, bstate, mp; ϵ=1e-4)
    @show "Train L"
    L, Lstate, latentdata = build_L_with_data(mp, dataloader, μAE, B_DNN)
    Glosses = train_L(L, Lstate, latentdata, mp, B_DNN, μAE)
    @show "Train perfNN"
    perflosses = train_perfNN(latentdata, perfNN, perfNNstate, mp)
    @show "Save State and Make Plots $layer"
    save_state(mp)
    make_plots_and_save(mp, dataloader, latentdata, convNN, μAE, B_DNN, L, perfNN, Glosses, μlosses, blosses, convNNlosses)
end
begin
    layer = layers|>first
    mp = ModelParams(layer,  0.01, 50, 4096, errorL2, gpu)
    layerstr = join(["$(layer)->" for layer in mp.layers])[1:end-2]
    
    bAE, μAE, convNN, perfNN = build_networks(mp)
    μstate, bstate, convNNstate, perfNNstate = build_states(mp, bAE, μAE, convNN, perfNN)
    B_DNN = build_BDNN(convNN, bAE, mp)
    dataloader = build_dataloaders(mp)
    #load previous state
    load_AEs(;μ="μAE_L$(layerstr).jld2", bdnn="B_DNN_L$(layerstr).jld2")
    L, Lstate, latentdata = build_L_with_data(mp, dataloader, μAE, B_DNN)
    load_L(;l="L_L$(layerstr).jld2")
    @show "Train perfNN"
    # perflosses = train_perfNN(latentdata, perfNN, perfNNstate, mp)
end

# plot(latentdata.data.perfs[2,:])
# plot!(dataloader.data.perfs[2,:])
# plot!(perfs[2,:])
begin
    which = rand(1:199)
    ns = 501
    c = plot(latentdata.data.νs[:,ns*which+1:ns*(which+1)]|>Array, st=:heatmap, label="input")
    plot!(c, colorbar=:false)
    title!("Latent")
    plot(latentdata.data.perfs[2,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    a = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[1,:], label="Approx", marker=:circle, ms=1)
    title!("Lift")
    plot(latentdata.data.perfs[3,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    b = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[2,:], label="Approx", marker=:circle, ms=1)
    title!("Thrust")
    Γ = latentdata.data.μs[1,ns*which+1:ns*(which+1)] - latentdata.data.μs[end,ns*which+1:ns*(which+1)]
    d = plot(Γ, label="Truth",lw=4)
    plot!(d, perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[3,:], label="Approx", marker=:circle, ms=1)
    title!("Γ strength")
    plot(c,a,b,d, layout = (4,1), size = (1200,800))
end


b2perf = Chain(Dense(64,2,  Flux.tanh),
                Dense(64,64, Flux.tanh),
                Dense(64,64, Flux.tanh  ),
                Dense(64,2))|>mp.dev
b2perfstate = Flux.setup(Adam(0.001), b2perf)
b2perflosses = []
for epoch = 1:50
    for (inp,RHS,μs,perf,_) in dataloader
        RHS = RHS |>mp.dev
        μs = μs |>mp.dev
        # inp = reshape(inp,(*(size(inp)[1:3]...),size(inp,4))) |>mp.dev
        # μs = vcat(μs, circshift(μs, (0,-1)))
        thrust = perf[2:3,:] |>mp.dev
        ls = 0.0
        grads = Flux.gradient(b2perf) do m
            # Evaluate model and loss inside gradient context:
            ŷ = m(μs)
            # ls = errorL2(ŷ[1,:], thrust[1,:])
            ls += errorL2(ŷ[2,:], thrust[2,:])
        end
        Flux.update!(b2perfstate, b2perf, grads[1])
        push!(b2perflosses, ls)  # logging, outside gradient context
    end
    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(b2perflosses[end])")
    end
end

begin
    which = rand(1:200)
    ns = 501
    xs = 1:ns
    plot(xs, dataloader.data.perfs[2:3,ns*which+1:ns*(which+1)][2,:], label="Truth")
    inp = dataloader.data.μs[:,ns*which+1:ns*(which+1)]
    timed = inp
    b = plot!(1:2:ns, b2perf(timed|>gpu)[2,1:2:ns], label="Approx", marker=:circle, ms=4,lw=1)
    title!("Thrust")
end


# make an embedding of the latent space with x,y,t components
begin 
    # adhoc 
    data_file="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
    coeffs_file= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
    data = deserialize(joinpath("data", data_file))
    coeffs = deserialize(joinpath("data", coeffs_file))
end
forces = coeffs[:,:coeffs]
which = rand(1:200)
ns = 501
full_sim = ns*which+1:ns*(which+1)
# latentdata.data.νs[:,full_sim]
clts = zeros(2,ns-1,200)
νembs = zeros(mp.layers[end]+1,ns-1,200)|>cpu
allnus = B_DNN.layers[1:2](inputs[:,:,:,:]|>mp.dev)|>cpu
pos = zeros(Float32, 64,2, (ns -1), 200)
μtrim = zeros(Float32, 64, (ns-1),200)
for i=0:199
    full_sim = ns*i+1:ns*(i+1) - 1
    times = data[full_sim,:t]
    # times = times .- times[1]
    times = times .% times[101]
    # poss = data[full_sim,:position]
    normals = data[full_sim,:normals]
    nx = map(normal->mean(normal[1,:]), normals)
    ny = map(normal->mean(normal[2,33:end]), normals)
    νs = allnus[:,full_sim]
    clts[:,:,i+1] = forces[i+1][2:3,1:end-1]
    νembs[:,:,i+1] .=  vcat(νs, times')
    pos[:,:,:,i+1] .= inputs[3:end-2,:,1,full_sim]
    μtrim[:,:,i+1] = μs[:,full_sim]
    # νembs[:,:,i+1] .=  νs
end
νembs = reshape(νembs, (mp.layers[end]+1,200*(ns-1 )))
clts = reshape(clts, (2,200*(ns-1)))
pos = reshape(pos, (size(pos)[1:2]...,200*(ns-1)))
μtrim = reshape(μtrim, (size(μtrim)[1]...,200*(ns-1)))
pinndata = DataLoader((νs=νembs.|>Float32, clts=clts.|>Float32, pos = pos.|>Float32, μs = μtrim.|>Float32), batchsize=100, shuffle=false)

test = collect(Iterators.take(pinndata,50))

plot()
for i = 15:30
    plot!(test[i].clts[2,:]  , label="")
end
plot!()




nusize = 8 #size(allnus,1)
nuext = nusize + 3 #x,y,t
# inputs ν[8x1],x,y,t ->  ̂ν[2,1], P_ish
# PINN = Chain(Dense(nuext, nuext, tanh),
#              Dense(nuext, nuext, tanh),             
#              Dense(nuext, 3))|>mp.dev
PINN = Chain(RNN(nuext, nuext, tanh),
            Dense(nuext, nuext, tanh),             
            Dense(nuext, 3))|>mp.dev                         

ts = 55
trunk = Chain(Conv((4,), 2=>2, Flux.tanh),
              Conv((4,), 2=>2, Flux.tanh),
              Conv((4,), 2=>1, Flux.tanh),
              Flux.flatten,
             Dense(ts, ts, tanh),
             Dense(ts, 2)) |>mp.dev
trunk(test[1].pos)
# pipe out puts 4x1 of latent space nu P x y
# pipe = Parallel(vcat, PINN, trunk)
pipe = PairwiseFusion(vcat, trunk, PINN)
pipe(pos[:,:,1:1], νembs[:,1:1])
pinnstate = Flux.setup(Adam(0.01), pipe)


function set_dir(ndirs, whichdir)
    dir = zeros(Float32, ndirs)
    dir[whichdir] = 1.0
    dir
end
ϵ = eps(Float32)^0.5
dx = set_dir(nuext,1)*ϵ
dy = set_dir(nuext,2)*ϵ
dt = set_dir(nuext, nuext)*ϵ



pinnlosses = []
for epoch = 1:1
    @show epoch
    for (ν,lt,pxy,mus) in pinndata
        ν = ν |> mp.dev
        lift  = lt[1,:] |> mp.dev
        thrst = lt[2,:] |> mp.dev
        pxy = pxy |> mp.dev
        mus = mus |>mp.dev
        Γ = mus[1,:] - mus[end,:]
        global ls = 0.0
        Flux.reset!(pipe)
        grads = Flux.gradient(pipe) do m
           # Evaluate model and loss inside gradient context:

            xy, out = m(pxy,ν)
            # me = Flux.mse(y.^2, 0.0)
            pinn = m.layers[2]
            lat = vcat(xy,ν)
            # ddt = (m(ν .+ dt) - m(ν))/ϵ
            ddx = (pinn(lat .+ dx) - pinn(lat.- dx))/(2ϵ)
            ddy = (pinn(lat .+ dy) - pinn(lat.-dy))/(2ϵ)
            # # # #approximate the Unsteady Bernoulli's equation
            # # # ∂ν/∂t + ν ∇⋅(ν) + |∇ν|^2 + P_ish = 0
            bern = sum(ddx[1,:].^2 .+ ddy[1,:].^2, dims=2) + out[2,:]
            # bern = y[1,:].^2/2.0 + y[2,:]
            be = Flux.mse(bern, 0.0)            
            # # # then forces are equal to 
            # # # 0 = -Pish⋅n⋅dS
            ct = errorL2(-xy[1,:].*out[2,:], thrst)
            cl = errorL2(-xy[2,:].*out[2,:], lift)
            Γe = errorL2(out[3,:], Γ)
            
            ls = cl + ct + be + Γe
            # @show ls 
            ls                    
        end
        Flux.update!(pinnstate, pipe, grads[1])
        push!(pinnlosses, ls)  # logging, outside gradient context
    end

    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(pinnlosses[end])")
    end
end


# begin
#     which = rand(1:199)
#     ns = 501
#     samp = pinndata.data.νs[:,ns*which+1:ns*(which+1)]
#     x  = samp[9,:]
#     y  = samp[10,:]
#     c = plot(samp|>Array, st=:heatmap, label="input")
#     plot!(c, colorbar=:false)
#     title!("Latent")
#     plot(pinndata.data.clts[1,ns*which+1:ns*(which+1)], label="Truth",lw=4)
#     a = plot!(-x.*PINN(samp)[2,:], label="Approx", marker=:circle, ms=1)
#     title!("Lift")
#     plot(pinndata.data.clts[2,ns*which+1:ns*(which+1)], label="Truth",lw=4)
#     b = plot!(-y.*PINN(samp)[2,:], label="Approx", marker=:circle, ms=1)
#     title!("Thrust")
#     c = plot(-y.*PINN(samp)[2,:], label="y", marker=:circle, ms=1)
#         plot!(c, x.*PINN(samp)[2,:], label="x", marker=:circle, ms=1)

#     # Γ = latentdata.data.μs[1,ns*which+1:ns*(which+1)] - latentdata.data.μs[end,ns*which+1:ns*(which+1)]
#     # d = plot(Γ, label="Truth",lw=4)
#     # plot!(d, perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[3,:], label="Approx", marker=:circle, ms=1)
#     # title!("Γ strength")
#     plot(a,b,c,  layout = (3,1), size = (1200,800))
# end


begin
    which = rand(1:199)
    ns = 501
    slice = ns*which+1:ns*(which+1)
    samp = pinndata.data.νs[:,slice]
    spxy = pinndata.data.pos[:,:,slice]
    Γs = pinndata.data.μs[1,slice] - pinndata.data.μs[end,slice]
    xy, out = pipe(spxy,samp)
    nuh = out[1,:]
    ph  = out[2,:]
    x = xy[1,:]
    y = xy[2,:]
    plot(pinndata.data.clts[1,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    a = plot!(-ph.*y, label="Approx", marker=:circle, ms=1)
    title!("Lift")
    plot(pinndata.data.clts[2,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    b = plot!(-ph.*x, label="Approx", marker=:circle, ms=1)
    title!("Thrust")
    plot(x, label="x")
    c = plot!(y, label="y", lw= 0 , marker=:circle, ms=2)
    plot(Γs, label="Γ Truth",lw=4)
    d = plot!(out[3,:], label="Approx", marker=:circle, ms=1)
    
    plot(a,b,c,d, layout = (4,1), size = (1200,800))

end