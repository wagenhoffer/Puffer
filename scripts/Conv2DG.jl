using Plots
using Flux.Data: DataLoader
using Flux

using Statistics
using LinearAlgebra
using CUDA
using Serialization, DataFrames


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

mp = ModelParams([64,32,16,8], 0.01, 1_000, 2048, errorL2, gpu)

data = deserialize(joinpath("data", "single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"))
RHS = data[:, :RHS]
μs = data[:, :μs]

N = mp.layers[1]
σs = nothing
pad = 2
inputs = zeros(N + pad*2, 2, 4, size(data,1))
freqs = data[:, :reduced_freq]
ks    = data[:,:k]
for (i,row) in enumerate(eachrow(data))       
    U_inf = ones(N).*row.U_inf
    freaks = ones(N).*row.reduced_freq
    # foil should be in the [-1,1; -1,1] domain
    position = row.position .- minimum(row.position, dims=2) 
    # x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' [U_inf U_inf] [freaks freaks]]
    x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]
    # wrap it around itself for circular convolution
    x_in = vcat(x_in[end-pad+1:end,:,:,:],x_in,x_in[1:pad,:,:,:])
    x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
    inputs[:,:,:,i] = x_in
end

W, H, C, Samples = size(inputs)
N 
# this is for 2d channels
# conv arch - layer reasoning
# 1. mimicry of FFT convolutions
# 2. further mimicry of FFT convolutions, with data truncation
# 3. 2D->1D convolution 
# 4. reshape for testing 
# 5. dense layer account for missing physics/data
# Wrap it all in a ResNet; mess with this in the future for better scaling
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


AEb = build_ae_layers(mp.layers)
μAE = build_ae_layers(mp.layers) |> mp.dev
# B_DNN = SkipConnection(Chain(
#     convenc = convenc,
#     enc = AEb[1],
#     dec = AEb[2],
#     convdec = convdec
# ), .+)|>gpu
B_DNN = Chain(
    convenc = convenc,
    enc = AEb[1],
    dec = AEb[2],
    convdec = convdec
) |> mp.dev

#+++++
begin
    σs = data[:, :σs]
    μs = data[:, :μs]
    lNNloader = DataLoader((hcat(σs...).|>Float32, hcat(RHS...).|>Float32), batchsize=mp.batchsize, shuffle=true)
    lNN = Chain(Dense(64,16, tanh),
                Dense(16,16, tanh),
                Dense(16,64, tanh))

    lNNstate = Flux.setup(Adam(mp.η), lNN)
    # x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]
    # sum(σs .== inputs[3:end-2,:,2,:].*(inputs[3:end-2,:,3,:]+inputs[3:end-2,:,4,:])  )

end
# Train the solution AE
begin
    teError(x,y) = Flux.mse(x[1,:]-x[end,:],y[1,:]-y[end,:])#/Flux.mse(y[end,:]-y[1,:],0.0)
    losses = []
    for epoch = 1:1_000
        for (x, y) in lNNloader        
            
            ls = 0.0
            grads = Flux.gradient(lNN) do m
                # Evaluate model and loss inside gradient context:
                ls = errorL2(m(x),y) +teError(m(x), y)
            end
            Flux.update!(lNNstate, lNN, grads[1])
            push!(losses, ls)  # logging, outside gradient context
        end
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(losses[end])")
        end
    end
    which = rand(1:size(lNNloader.data[1],2))
    plot(lNNloader.data[2][:,which], label="signal")
    a = plot!(lNN(lNNloader.data[1][:,which]), label="reconstructed")
    title!("RHS")
    b = plot(lNNloader.data[1][:,which], label="σ")
    plot!(b, lNNloader.data[2][:,which], label="rhs")
    plot(a,b, layout = (2,1), size = (1200,800))
end

#+++++
DNNstate = Flux.setup(Adam(mp.η), B_DNN)
μstate = Flux.setup(Adam(mp.η), μAE)
convencstate = Flux.setup(Adam(mp.η), convenc)
dataloader = DataLoader((inputs.|>Float32, hcat(RHS...).|>Float32), batchsize=mp.batchsize, shuffle=true)
μloader = DataLoader(hcat(μs...).|>Float32, batchsize=mp.batchsize, shuffle=true)

begin
    # try to train the convenc only
    losses = []
    for epoch = 1:mp.epochs
        for (x, y) in dataloader        
            x = x |> mp.dev
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(convenc) do m
                # Evaluate model and loss inside gradient context:
                ŷ = m(x)
                ls = errorL2(ŷ, y)  
                return ls
            end
            Flux.update!(convencstate, convenc, grads[1])
            push!(losses, ls)  # logging, outside gradient context
        end
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(losses[end])")
        end
    end
end

begin
    which = rand(1:size(dataloader.data[1],4))
    plot(dataloader.data[2][:,which], label = "",lw=4,c=:red)
    plot!(convenc(dataloader.data[1][:,:,:,which:which]|>mp.dev), label = "recon",lw=0.25, marker=:circle)
end
# Train the solution AE
begin
    losses = []
    for epoch = 1:mp.epochs
        for (y) in μloader        
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(μAE) do m
                # Evaluate model and loss inside gradient context:
                ls = errorL2(m(y), y) 
            end
            Flux.update!(μstate, μAE, grads[1])
            push!(losses, ls)  # logging, outside gradient context
        end
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(losses[end])")
        end
    end
    which = rand(1:size(μloader.data,2))
    plot(μs[which], label="signal")
    a = plot!(μAE(μloader.data[:,which]|>gpu), label="reconstructed")
    title!("μ")
    b = plot(losses, label="loss")
    plot(a,b, layout = (2,1), size = (1200,800))
end

begin
    #train the forcing function 
    losses = []
    @time for i=1:mp.epochs
        for (x, y) in dataloader
            x = x |> mp.dev
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(B_DNN) do m
                # Evaluate model and loss inside gradient context:
                # m = model.layers #remove the ResNet
                # converr = errorL2(model(x), x) # error over entire model
                mcx = m[1](x) # model -> conv x
                mcaex = m[2:3](mcx) # model -> conv -> AE x
                encerr = errorL2(mcx, y) # does the conv layer learn the RHS?
                decerr1 = errorL2(mcaex, y) # does the AE layer work for input?
                decerr2 = errorL2(mcaex, mcx) # does the AE layer work for conv input?
                # encdecerr = errorL2(m[4](m[1](x)), x) # just test Conv and ConvTranspose
                ls =  decerr1 
                
            end
            Flux.update!(DNNstate, B_DNN, grads[1])
            push!(losses, ls)  # logging, outside gradient context
        end
        if i % 10 == 0
            println("Epoch: $i, Loss: $(losses[end])")
        end
        
    end
    plot(losses, label="loss")
end

begin
    # this looks at the smaller model on conv layers only
    
    which = rand(1:size(dataloader.data[1],4))
    # @show inputs[1,1,6,which:which]
    plot(dataloader.data[2][:,which], label = "",lw=4,c=:red)
                
    x1 = reshape(B_DNN(dataloader.data[1][:,:,:,which:which]|>mp.dev)[:,:,1:4,:], 
                     (68,8))|>cpu         
    x2 = reshape(dataloader.data[1][:,:,1:4,which:which], (68,8))
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
    bDNN = B_DNN[1](dataloader.data[1][:,:,:,which:which]|>mp.dev)
    bAE  = B_DNN[1:3](dataloader.data[1][:,:,:,which:which]|>mp.dev)
    bTrue = dataloader.data[2][:,which:which]
    rhsPlot = plot(bDNN, label="Convolution", lw=0.25, marker=:circle)
    plot!(rhsPlot, bAE, label="AE", lw=0.25, marker=:circle)
    plot!(rhsPlot, bTrue, label = "Truth", lw=3)
    x = plot(pos, normals, wakevel, panelvel, layout = (2,2), size =(1000, 800)) 
    plot(x, rhsPlot, layout = (2,1), size = (1200,800))

end


# Let's build the Lν = f model
# Define the model
L = Chain(Dense(mp.layers[end], mp.layers[end]; bias=false))|>gpu

νs = μAE.layers.encoder.(μs|>mp.dev)
# βs = B_DNN[1:2](inputs.|>Float32|>mp.dev)
βs = R


latentdata = DataLoader((νs=hcat(νs...), βs=βs, μs = hcat(μs...).|>Float32,
             Bs =inputs.|>Float32), batchsize=mp.batchsize, shuffle=true)

Lopts = Flux.setup(Adam(mp.η), L)


losses = []
@time for epoch = 1:100
    for (ν, β, μ, B) in latentdata
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
            Lν = m(ν)
            
            ls = errorL2(Lν, β) 
            # superposition error            
            e2 = 0.0

            
            # for ind = 1:mp.batchsize       
            #     superβ = β + circshift(β, (0, ind))
            #     superν = ν + circshift(ν, (0, ind))
            #     superLν = m(superν)
            #     e2 += errorL2(superLν, superβ)
            # end
            # e2 /= mp.batchsize 

            e3 = errorL2(B_DNN[3:4](Lν), B) 
            # solve for \nu in L\nu = \beta
            Gb = m[1].weight\β
            e4 = errorL2(μAE.layers.decoder((Gb)), μ)
            
            ls = ls + e2 + e3 + e4 
        end
        Flux.update!(Lopts, L, grads[1])
        push!(losses, ls)  # logging, outside gradient context
    end
    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end


G = inv(L[1].weight|>cpu)
begin
    which = rand(1:size(latentdata.data[1],2))
    recon_ν = G*(B_DNN[1:2](latentdata.data.Bs[:,:,:,which:which]|>gpu)|>cpu)
    plot(latentdata.data.νs[:,which], label="latent signal")
    a = plot!(recon_ν, label="reconstructed")
    title!("ν")
    plot(latentdata.data.μs[:,which], label="μ Truth")
    b = plot!(μAE.layers.decoder(recon_ν|>gpu), label="μ DG ")
    title!("μs")
    plot(a,b, size=(1200,800))
end


begin
    # save all of the weights in Serialization
    # save the model parameters
    using JLD2Flux.tanh
    μ_state = Flux.state(μAE)|>cpu;
    bdnn_state = Flux.state(B_DNN)|>cpu;
    l_state = Flux.state(L)|>cpu;

    path = joinpath("data", "μAE.jld2")
    jldsave(path; μ_state)
    path = joinpath("data", "B_DNN.jld2")
    jldsave(path; bdnn_state)
    path = joinpath("data", "L.jld2")
    jldsave(path; l_state)
end

begin
    #load the weights
    using JLD2
    path = joinpath("data", "μAE.jld2")
    μ_state = JLD2.load(path,"μ_state")
    Flux.loadmodel!(μAE, μ_state)
    path = joinpath("data", "B_DNN.jld2")
    bdnn_state = JLD2.load(path,"bdnn_state")
    Flux.loadmodel!(B_DNN, bdnn_state)
    path = joinpath("data", "L.jld2")
    l_state = JLD2.load(path,"l_state")
    Flux.loadmodel!(L, l_state)
end

begin
    λs, vecs= eigen(G)
    plot(real(λs), imag(λs), seriestype=:scatter, label="Eigenvalues")
    plot(abs.(λs), marker=:circle, label="Eigenvalues")
    
end

begin
    eigenvalue,eigenvector = eigen(G)
    ix = sortperm(abs.(eigenvalue), rev=true)
    scatter(eigenvalue, marker_z = abs.(eigenvalue), 
    markercolor = cgrad(["white", "blue", "red"]), 
    legend=false, cbar=true)
end

plot()
for (λ,v) in zip(λs, eachcol(vecs))
    vals = λ*v
    plot!(real(vals), label="$(abs.(λ))")    
    plot!(imag(vals), label="$(abs.(λ))")   
end
plot!()


x = real.(eigenvalue)
y = imag.(eigenvalue)
z = norm.(eigenvector) 
scatter(x,y,marker_z=z, markercolors=:blues)


#looking at traing the convolution separated from the AE and then combining them
convN = SkipConnection(Chain(convenc, convdec), .+)|>mp.dev

convstate = Flux.setup(Adam(mp.η), convN)
ϵ = 1e-3
begin
    losses = []
    for epoch = 1:mp.epochs
        for (x, y) in dataloader        
            x = x |> mp.dev
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(convN) do m
                # Encode data, ensure it equals the RHS in latent space                
                lat = errorL2(m.layers[1](x), y)  
                #decoder it and make sure we can see the output space
                dec = errorL2(m(x), x)
                ls = lat
                return ls
            end
            Flux.update!(convstate, convN, grads[1])
            push!(losses, ls)  # logging, outside gradient context
        end
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(losses[end])")
        end
        if any(losses .< ϵ)
            @show "we out"
            break
        end
    end
    plot(losses)

end

begin
    which = rand(1:size(dataloader.data[1],4))
    truth = dataloader.data[2][:,which]|>cpu
    recon = convN.layers[1](dataloader.data[1][:,:,:,which:which]|>mp.dev)|>cpu
    err = norm(truth .- recon, 2)/norm(truth,2)
    @show err
    plot(truth, label = "truth",lw=4,c=:red)
    plot!(recon, label = "recon",lw=0.25, marker=:circle)
    plot!(title="Error = $(err)")
end

AEbstate = Flux.setup(Adam(mp.η), AEb)
begin
    losses = []
    for epoch = 1:mp.epochs
        for (x, y) in dataloader        
            # x = x |> mp.dev
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(AEb) do m                 
                ls =  errorL2(m(y), y)
                return ls#*mp.batchsize
            end
            Flux.update!(AEbstate, AEb, grads[1])
            push!(losses, ls)  # logging, outside gradient context
        end
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(losses[end])")
        end
    end
end
begin
    which = rand(1:size(dataloader.data[1],4))
    truth = dataloader.data[2][:,which]|>cpu
    recon = AEb(dataloader.data[2][:,which:which]|>mp.dev)|>cpu
    err = norm(truth .- recon, 2)/norm(truth,2)
    @show err
    plot(truth, label = "truth",lw=4,c=:red)
    plot!(recon, label = "recon",lw=0.25, marker=:circle)
    plot!(title="Error = $(err)")
end

begin 
    # slap together the conv and AE and see what happens
    convAE = Chain(convN.layers[1], 
                    AEb,
                   convenc.layers[2])
    which = rand(1:size(dataloader.data[1],4))
    truth = dataloader.data[2][:,which]|>cpu
    recon = convAE[1](dataloader.data[1][:,:,:,which:which]|>mp.dev)|>cpu
    recon2 =convAE[1:2](dataloader.data[1][:,:,:,which:which]|>mp.dev)|>cpu
    plot(truth, label = "truth",lw=4,c=:red)
    plot!(recon, label = "recon",lw=0.25, marker=:circle)
    plot!(recon2, label = "recon2",lw=0.25, marker=:circle)
end

