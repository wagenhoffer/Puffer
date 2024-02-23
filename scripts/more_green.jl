using Plots
using Flux.Data: DataLoader
using Flux

using Statistics
using LinearAlgebra
using CUDA
using Serialization, DataFrames
using JLD2

"""
    Modelparams(layers, η, epochs, batchsize, lossfunc, dev)

Holds parameters for creating and training the neural network.
    
# Parameters 
- **`layers::Vector`** - a vector containing data about the layers of the neural network (how many neurons are in each layer)
- **`η::Float64`** - eta, represents the learning rate hyperparameter (i think)
- **epochs::Int** - determines the number of training cycles
- **`batchsize::Int`** - represents the number of training samples used in a single training cylce. this hyperparameter controls how data is divided (batched)
- **`lossfunc::Function`** - the function that well use to evaluate our models error 
- **`dev`** - takes in the device that will run the model (well use the gpu usually)

# Example 
```julia
mp = ModelParams([64, 64, 32], 0.01, 1_000, 2048, errorL2, gpu)


"""
struct ModelParams
    layers::Vector    
    η::Float64
    epochs::Int
    batchsize::Int
    lossfunc::Function
    dev
end

errorL2(x, y) = Flux.mse(x,y)/Flux.mse(y,0.0) 


"""
    build_ae_layers(layer_sizes, activation)

# Arguments
- `layer_sizes` - determines the number of neurons within each layer 
- `activation` - activation function which outputs values between -1 and 1. (set to **tanh** by default) 

# Returns
a dense (fully connected) neural network. 

# Detailed Description 

**`Chain()`** - a function from Flux. 
Its chains together multiple NN layers into a single model. 
Chain allows us to immediately pass data through all layers of the neural network. 
the output of one network gets sent to the next automatically 

**`Dense()`** - takes in 3 parameters 
Dense(output_dimension, input_dimension, activation_function). 
Dense() is a single layer of neurons within our network. 
Each Neuron in a dense layer wheights the sum of all its inputs, adds a bias, 
and passes this whole thing through an activation function. 
Dense layers are **fully connected** I'm pretty sure. 

**`decoder`** 
the decoder is a chain of dense layers. 
layer_sizes is an array containing the size of each layer at the i-th index with larger layer sizes at the beginning. 
Since the Decoder takes inputs from the hidden layer (of a smaller dimension) and blows them up to the original dimension, 
we start from the end of layer_sizes for input and connect the layer to the next largest dimensioned layer. 
The decoder effectively reverses the dimensionality reduction done by the encoder

**`encoder`** 
just like the decoder, this is a chain of dense layers. 
We construct it in the opposite way of the decoder. 
We start with the input layer (first index of layer_sizes) 
and connect each subsequent layer to the next one with a smaller dimension. 
This occurs until we reach the end of layer_sizes which connects to the hidden layer.  
 
"""
function build_ae_layers(layer_sizes, activation = tanh)    
    decoder = Chain([Dense(layer_sizes[i+1], layer_sizes[i], activation) 
                    for i = length(layer_sizes)-1:-1:1]...)    
    encoder = Chain([Dense(layer_sizes[i], layer_sizes[i+1], activation)
                    for i = 1:length(layer_sizes)-1]...)
    Chain(encoder = encoder, decoder = decoder)
end


"""
    build_dataloaders(mp::ModelParams; data, coeffs)

# Arguments
- `mp`- supplies information about the model's structure. Here we just get the size of the input layer. 
- `data` - the raw data that will be used to train the network (from a simulation?)
- `coefs` - swimmer parameters that we wish to predict? (attributes like force, thurst, velocity etc.)

# Detailed Description 
This function prepares the data for processing by the model. This is done with using the following preprocessing techniques: 

## Desrialization
Raw data is extracted from a file containing the raw simulation data as well as the coefficients of swimmer data such thrust, lift, etc. 
This is done using `deserialize()` which uses `joinpath()` to resolve the local path of the file on your system. Simulation data is loaded into `data`
and the coefficient attributes are loaded into `coeffs`

## Preprocessing (RHS and μ)
Here we prepare the data to train the autoencoders. Both RHS and μ are first reshaped into horizontal arrays with `hcat()` 
and then converted to type Float32. The autoencoder denoted ϕ is then fed the μ array. 
Usually, we'll set the autoencoder hidden layers to some dimension lower than 
that of the input layer. After the μ data has passed through the autencoder, well need to ensure that it matches the data
in RHS, essentially validating the autoencoder. 


## Preparing NN Input 

```julia
    N = mp.layers[1]
    pad = 2
```
Here we set `N` to be the input layer. 
pad defines the amount of zeroes which will pad our data.
this is done to keep consistent array dims between different NNs 
but reduce dimensionality within the array. 

## Creating the Empty input Array 
```julia
    inputs = zeros(Float32, N + pad*2, 2, 4, size(data,1))
```
Here we create an (n-dimension) array of zeroes first. Since well be 
reducing the dimensions of our data in the hidden layers of the AEs, 
well use some zero values to pad the array to keep a consistent input 
array size. 


## Reshaping 
```julia
    for (i,row) in enumerate(eachrow(data))
        position = row.position .- minimum(row.position, dims=2)
```
**Since we are reshaping the array, we will shift the data arround in such a way to conform 
to the new dimensions that are defined in the model (width, number of layers etc.?**
We then add the transposed vectors (denote as `position' row.normals' row.wake_ind_vel' row.panel_vel'`)
the final shape of `x_in` should be a horizontal array? 


## Processing Outs & Evaluating AE Performance 
```julia
    perfs = zeros(4, size(data,1))
    ns = coeffs[1,:coeffs]|>size|>last
    
    for (i,r) in enumerate(eachrow(coeffs))
        perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2)
        perfs[:,ns*(i-1)+1:ns*i] = perf
    end
```
After the autoencoder is trained, we then need to evaluate its ability
to reconstruct the coefficient data from the piped in data after it goes 
through the hidden layers (hidden layers have lower dimensionality). 
`perfs` is a 2D array with 4 rows and n columns where n is the number of rows in data. 
**All elements are initially set to 0. `ns` gets the number of elements for a particular 
column???** of coeffs and gets its size. **Size then gets pumped to last so we
pull the last item element of what `size()` outputs which is the number of columnsfor the 
coefs vector? Please Explain the code in the for loop!!!!!!**

## Saving the Dataloader 
```julia
    dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs), batchsize=mp.batchsize, shuffle=true)    
```



"""
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
"""
```julia
    build_networks(mp; C = Layers, N = inputSize)
```
This function builds the autoencoders and autodecoders for the trained neural networks. 


"""
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
    B_DNN = SkipConnection(Chain(
        convenc = convNN.layers[1],
        enc = bAE[1],
        dec = bAE[2],
        convdec = convNN.layers[2]), .+) |> mp.dev
   bAE, μAE, convNN, perfNN, B_DNN
end

function build_states(mp, bAE, μAE, convNN, perfNN)
    μstate       = Flux.setup(Adam(mp.η), μAE)
    bstate       = Flux.setup(Adam(mp.η), bAE)
    convNNstate = Flux.setup(Adam(mp.η), convNN)
    perfNNstate = Flux.setup(Adam(mp.η), perfNN)
    μstate, bstate, convNNstate, perfNNstate
end

function train_AEs(dataloader, convNN, convNNstate, μAE, μstate, bAE, bstate, mp)
    # try to train the convNN layers for input spaces 
    convNNlosses = []
    ϵ = 1e-3;
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
        if epoch % 100 == 0
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
        if epoch % 100 == 0
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
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(blosses[end])")
        end
        if kick == true
            break
        end
    end
    (convNNlosses, μlosses, blosses)
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
        if epoch % 100 == 0
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
    which = rand(1:size(dataloader.data[1],4))

    μ = dataloader.data.μs[:,which]
    μa = convNN.layers[1](dataloader.data[1][:,:,:,which:which]|>mp.dev)
    plot(μ, label = "Truth",lw=4,c=:red)
    plot!(μa, label="Approx",lw=0.25, marker=:circle)
    title!("RHS : error = $(errorL2(μa, μ))")
    savefig(joinpath("images","conv_RHS.png"))


    
    μa = μAE(μ|>mp.dev)
    plot(μ, label = "Truth",lw=4,c=:red)
    a = plot!(μa, label="Approx")
    title!("μ : error = $(errorL2(μa, μ))")
    savefig(joinpath("images","μAE.png"))

    B = dataloader.data.RHS[:,which]
    Ba = B_DNN.layers[2:3](B|>mp.dev)
    plot(B, label="Truth")
    a = plot!(Ba, label="Approx")
    title!("B : error = $(errorL2(Ba, B))")
    savefig(joinpath("images","B_DNN.png"))

                    
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
    savefig(joinpath("images","field_recon_and_RHS.png"))


    G = inv(L|>cpu)
    recon_ν = G*(B_DNN.layers[:enc](latentdata.data.Bs[:,which:which]|>mp.dev)|>cpu)
    plot(latentdata.data.νs[:,which], label="latent signal",lw=1,marker=:circle)
    a = plot!(recon_ν, label="Approx",lw=1,marker=:circle)
    title!("ν : error = $(errorL2(recon_ν, latentdata.data.νs[:,which]))")
    plot(latentdata.data.μs[:,which], label="μ Truth")
    b = plot!(μAE.layers.decoder(recon_ν|>mp.dev), label="μ DG ")
    title!("μs : error = $(errorL2(μAE.layers.decoder(recon_ν|>mp.dev), latentdata.data.μs[:,which]))")
    plot(a,b, size=(1200,800))
    savefig(joinpath("images","L_recon.png"))

    which = rand(1:199)
    c = plot(perfloader.data.νs[:,ns*which+1:ns*(which+1)], st=:contour, label="input")
    plot!(c, colorbar=:false)
    title!("Latent")
    plot(perfloader.data.perfs[1,ns*which+1:ns*(which+1)], label="Truth")
    a = plot!(perfNN(perfloader.data.νs[:,ns*which+1:ns*(which+1)])[1,:], label="Approx")
    title!("Lift")
    plot(perfloader.data.perfs[2,ns*which+1:ns*(which+1)], label="Truth")
    b = plot!(perfNN(perfloader.data.νs[:,ns*which+1:ns*(which+1)])[2,:], label="Approx")
    title!("Thrust")
    plot(c,a,b, layout = (3,1), size = (1200,800))
    savefig(joinpath("images","perfNN.png"))    

    #make a plot of the losses for all trainings
    e = plot(convNNlosses, yscale=:log10, title="ConvNN")
    a = plot(μlosses,      yscale=:log10, title="μAE")
    b = plot(blosses,      yscale=:log10, title="B_DNN")
    c = plot(Glosses,      yscale=:log10, title="L")
    d = plot(perflosses,   yscale=:log10, title="perfNN")
    plot(a,b,c,d,e, size=(1200,800))
    savefig(joinpath("images","losses.png"))
end

mp = ModelParams([64, 64, 8], 0.01, 1_000, 128, errorL2, gpu)
bAE, μAE, convNN, perfNN, B_DNN = build_networks(mp)
μstate, bstate, convNNstate, perfNNstate = build_states(mp, bAE, μAE, convNN, perfNN)
dataloader = build_dataloaders(mp)

convNNlosses, μlosses, blosses = train_AEs(dataloader, convNN, convNNstate, μAE, μstate, bAE, bstate, mp)

L, Lstate, latentdata = build_L_with_data(mp, dataloader, μAE, B_DNN)
Glosses = train_L(L, Lstate, latentdata, mp, B_DNN, μAE)
perflosses = train_perfNN(latentdata, perfNN, perfNNstate, mp)
save_state(mp)
make_plots_and_save(mp, dataloader, latentdata, convNN, μAE, B_DNN, L, perfNN, Glosses, μlosses, blosses, convNNlosses)