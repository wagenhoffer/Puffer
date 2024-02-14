using Plots
using Flux.Data: DataLoader
using Flux

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



function build_dataloaders(mp::ModelParams; data="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls", coeffs= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls")
    # load/ prep  training data
    data = deserialize(joinpath("data", data))
    coeffs = deserialize(joinpath("data", coeffs))
    using Plots
    using Flux.Data: DataLoader
    using Flux
    
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
    
    
    
    function build_dataloaders(mp::ModelParams; data="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls", coeffs= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls")
        # load/ prep  training data
        data = deserialize(joinpath("data", data))
        coeffs = deserialize(joinpath("data", coeffs))
    
        RHS = hcat(data[:, :RHS]...).|>Float32
        μs = hcat(data[:, :μs]...).|>Float32
    
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
            perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2)
            perfs[:,ns*(i-1)+1:ns*i] = perf
        end
    
        dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs), batchsize=mp.batchsize, shuffle=true)    
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
                    Dense(mp.layers[end],mp.layers[end]÷2,Flux.tanh),
                    Dense(mp.layers[end]÷2,2,Flux.tanh))|>mp.dev               
    
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
            for (x, y,_,_) in dataloader        
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
            for (_,_,y,_) in dataloader        
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
            for (x, y,_,_) in dataloader        
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
        for epoch = 1:mp.epochs/10
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
            for (ν,_,_,_,perf) in latentdata
                ν = ν |> mp.dev
                perf = perf |> mp.dev
                ls = 0.0
                grads = Flux.gradient(perfNN) do m
                    # Evaluate model and loss inside gradient context:
                    y = m(ν)
                    ls = errorL2(y[1,2:end], perf[1,2:end])
                    ls +=  errorL2(y[2,2:end], perf[2,2:end])                
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
        μ = "μAE_L$(mp.layers[end]).jld2"
        bdnn = "B_DNN_L$(mp.layers[end]).jld2"
        l = "L_L$(mp.layers[end]).jld2"
        perf = "perf_L$(mp.layers[end]).jld2"
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
        μ = dataloader.data.μs[:,which]
        μa = B_DNN.layers[:convenc](dataloader.data[1][:,:,:,which:which]|>mp.dev)
        plot(μ, label = "Truth",lw=4,c=:red)
        plot!(μa, label="Approx",lw=0.25, marker=:circle)
        title!("RHS : error = $(errorL2(μa|>cpu, μ|>cpu))")
        savefig(joinpath(dir_path,"conv_RHS"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))
    
    
        
        μa = μAE(μ|>mp.dev)
        plot(μ, label = "Truth",lw=4,c=:red)
        a = plot!(μa,marker=:circle,lw=0, label="Approx")
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
        a = plot!(recon_ν, label="Approx",lw=1,marker=:circle)
        title!("ν : error = $(errorL2(recon_ν|>cpu, latentdata.data.νs[:,which]|>cpu))")
        plot(latentdata.data.μs[:,which], label="μ Truth")
        b = plot!(μAE.layers.decoder(recon_ν|>mp.dev), label="μ DG ")
        title!("μs : error = $(errorL2(μAE.layers.decoder(recon_ν|>mp.dev)|>cpu, latentdata.data.μs[:,which]|>cpu))")
        plot(a,b, size=(1200,800))
        savefig(joinpath(dir_path,"L_recon"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))
    
        which = rand(1:199)
        ns = 357
        c = plot(latentdata.data.νs[:,ns*which+1:ns*(which+1)], st=:contour, label="input")
        plot!(c, colorbar=:false)
        title!("Latent")
        plot(latentdata.data.perfs[1,ns*which+1:ns*(which+1)], label="Truth")
        a = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[1,:], label="Approx")
        title!("Lift")
        plot(latentdata.data.perfs[2,ns*which+1:ns*(which+1)], label="Truth")
        b = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[2,:], label="Approx")
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
    RHS = hcat(data[:, :RHS]...).|>Float32
    μs = hcat(data[:, :μs]...).|>Float32

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
        perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2)
        perfs[:,ns*(i-1)+1:ns*i] = perf
    end

    dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs), batchsize=mp.batchsize, shuffle=true)    
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
                Dense(mp.layers[end],mp.layers[end]÷2,Flux.tanh),
                Dense(mp.layers[end]÷2,2,Flux.tanh))|>mp.dev               

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
        for (x, y,_,_) in dataloader        
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
        for (_,_,y,_) in dataloader        
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
        for (x, y,_,_) in dataloader        
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
        for (ν,_,_,_,perf) in latentdata
            ν = ν |> mp.dev
            perf = perf |> mp.dev
            ls = 0.0
            grads = Flux.gradient(perfNN) do m
                # Evaluate model and loss inside gradient context:
                y = m(ν)
                ls = errorL2(y[1,2:end], perf[1,2:end])
                ls +=  errorL2(y[2,2:end], perf[2,2:end])                
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
    perf = "perf_L$(layers).jld2"
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
    μ = dataloader.data.μs[:,which]
    μa = B_DNN.layers[:convenc](dataloader.data[1][:,:,:,which:which]|>mp.dev)
    plot(μ, label = "Truth",lw=4,c=:red)
    plot!(μa, label="Approx",lw=0.25, marker=:circle)
    title!("RHS : error = $(errorL2(μa|>cpu, μ|>cpu))")
    savefig(joinpath(dir_path,"conv_RHS"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))


    
    μa = μAE(μ|>mp.dev)
    plot(μ, label = "Truth",lw=4,c=:red)
    a = plot!(μa,marker=:circle,lw=0, label="Approx")
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
    a = plot!(recon_ν, label="Approx",lw=1,marker=:circle)
    title!("ν : error = $(errorL2(recon_ν|>cpu, latentdata.data.νs[:,which]|>cpu))")
    plot(latentdata.data.μs[:,which], label="μ Truth")
    b = plot!(μAE.layers.decoder(recon_ν|>mp.dev), label="μ DG ")
    title!("μs : error = $(errorL2(μAE.layers.decoder(recon_ν|>mp.dev)|>cpu, latentdata.data.μs[:,which]|>cpu))")
    plot(a,b, size=(1200,800))
    savefig(joinpath(dir_path,"L_recon"*join(["$(layer)->" for layer in mp.layers])[1:end-2]*".png"))

    which = rand(1:199)
    ns = 357
    c = plot(latentdata.data.νs[:,ns*which+1:ns*(which+1)], st=:contour, label="input")
    plot!(c, colorbar=:false)
    title!("Latent")
    plot(latentdata.data.perfs[1,ns*which+1:ns*(which+1)], label="Truth")
    a = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[1,:], label="Approx")
    title!("Lift")
    plot(latentdata.data.perfs[2,ns*which+1:ns*(which+1)], label="Truth")
    b = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[2,:], label="Approx")
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

layers = [[64,32,16, 8],
          [64,64,64],
          [64,64,32],
          [64,32,16],          
          [64,32,16,8,4],
          [64,32,16,8,4,2],
          [64,32,16,8,4,2,1]]
for layer in layers            
    mp = ModelParams(layer,  0.01, 50, 2048, errorL2, gpu)
    bAE, μAE, convNN, perfNN = build_networks(mp)
    μstate, bstate, convNNstate, perfNNstate = build_states(mp, bAE, μAE, convNN, perfNN)
    dataloader = build_dataloaders(mp)

    B_DNN,(convNNlosses, μlosses, blosses) = train_AEs(dataloader, convNN, convNNstate, μAE, μstate, bAE, bstate, mp; ϵ=1e-4)

    L, Lstate, latentdata = build_L_with_data(mp, dataloader, μAE, B_DNN)
    Glosses = train_L(L, Lstate, latentdata, mp, B_DNN, μAE)
    perflosses = train_perfNN(latentdata, perfNN, perfNNstate, mp)
    save_state(mp)
    make_plots_and_save(mp, dataloader, latentdata, convNN, μAE, B_DNN, L, perfNN, Glosses, μlosses, blosses, convNNlosses)
end