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

mp = ModelParams([64,32,16,8], 0.01, 1_000, 2048, errorL2, cpu)

# load/ prep  training data
data = deserialize(joinpath("data", "single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"))
coeffs = deserialize(joinpath("data", "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"))
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

# unroll the coeffs into the same order  as the inputs
#force, lift, thrust, power 
perfs = zeros(4, size(data,1))
ns = coeffs[1,:coeffs]|>size|>last
for (i,r) in enumerate(eachrow(coeffs))
    perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2)
    perfs[:,ns*(i-1)+1:ns*i] = perf
end
# end training data load

#begin build models
# this is for 2d channels
# conv arch - layer reasoning
# 1. mimicry of FFT convolutions
# 2. further mimicry of FFT convolutions, with data truncation
# 3. 2D->1D convolution 
# 4. reshape for testing 
# 5. dense layer account for missing physics/data
# Wrap it all in a ResNet; mess with this in the future for better scaling
W, H, C, Samples = size(inputs)
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
                Dense(mp.layers[end]÷2,2,Flux.tanh))

μstate       = Flux.setup(Adam(mp.η), μAE)
bstate       = Flux.setup(Adam(mp.η), bAE)
convNNstate = Flux.setup(Adam(mp.η), convNN)
perfNNstate = Flux.setup(Adam(mp.η), perfNN)


#Slap bAE into the middle of the convNN and call it B_DNN
B_DNN = SkipConnection(Chain(
    convenc = convNN.layers[1],
    enc = bAE[1],
    dec = bAE[2],
    convdec = convNN.layers[2]), .+) |> mp.dev

L = rand(mp.layers[end],mp.layers[end])


#no state train it as separated parts
#end build models

dataloader = DataLoader((inputs.|>Float32, hcat(RHS...).|>Float32), batchsize=mp.batchsize, shuffle=true)
μloader    = DataLoader(hcat(μs...).|>Float32, batchsize=mp.batchsize, shuffle=true)

begin
    # try to train the convNN layers for input spaces 
    losses = []
    for epoch = 1:mp.epochs
        for (x, y) in dataloader        
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
    plot!(convNN.layers[1](dataloader.data[1][:,:,:,which:which]|>mp.dev), label = "recon",lw=0.25, marker=:circle)
end

# Train the solution AE - resultant μ
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

# Train the solution AE - forcing b
begin
    losses = []
    for epoch = 1:mp.epochs
        for (x, y) in dataloader        
            y = y |> mp.dev
            ls = 0.0
            grads = Flux.gradient(bAE) do m
                # Evaluate model and loss inside gradient context:
                ls = errorL2(m(y), y) 
            end
            Flux.update!(bstate, bAE, grads[1])
            push!(losses, ls)  # logging, outside gradient context
        end
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(losses[end])")
        end
    end
    which = rand(1:size(dataloader.data[2],2))
    plot(RHS[which], label="signal")
    a = plot!(bAE(dataloader.data[2][:,which]|>gpu), label="reconstructed")
    title!("B")
    b = plot(losses, label="loss")
    plot(a,b, layout = (2,1), size = (1200,800))
end


#Slap bAE into the middle of the convNN and call it B_DNN
B_DNN = SkipConnection(Chain(
    convenc = convNN.layers[1],
    enc = bAE[1],
    dec = bAE[2],
    convdec = convNN.layers[2]), .+) |> mp.dev


begin
    # this looks at the smaller model on conv layers only
    
    which = rand(1:size(dataloader.data[1],4))
    # B_DNN(dataloader.data[1][:,:,:,which:which]|>mp.dev)
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
    b_DNN = B_DNN.layers[1](dataloader.data[1][:,:,:,which:which]|>mp.dev)
    b_AE  = B_DNN.layers[1:3](dataloader.data[1][:,:,:,which:which]|>mp.dev)
    b_True = dataloader.data[2][:,which:which]
    rhsPlot = plot(b_DNN, label="Convolution", lw=0.25, marker=:circle)
    plot!(rhsPlot, b_AE, label="AE", lw=0.25, marker=:circle)
    plot!(rhsPlot, b_True, label = "Truth", lw=3)
    x = plot(pos, normals, wakevel, panelvel, layout = (2,2), size =(1000, 800)) 
    plot(x, rhsPlot, layout = (2,1), size = (1200,800))

end


# Let's build the Lν = f model
# Define the model
# L = Chain(Dense(mp.layers[end], mp.layers[end]; bias=false,init=(x,y)->Symmetric(rand(x,y)|>Matrix)))|>mp.dev
L = rand(Float32, mp.layers[end],mp.layers[end])
# L[1].weight
Bs = hcat(RHS...).|>Float32
μs = hcat(μs...).|>Float32
νs = μAE[:encoder](μs|>mp.dev)
βs = bAE[:encoder](Bs|>mp.dev)


latentdata = DataLoader((νs=νs, βs=βs, μs=μs, Bs=Bs), batchsize=mp.batchsize, shuffle=true)

Lopts = Flux.setup(Adam(mp.η), L)

coeffs
losses = []
e3 = e2 = e4 = 0.0
@time for epoch = 1:mp.epochs
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
            # Lν = m(ν)
            Lν = m*ν
            
            ls = errorL2(Lν, β) 
            # superposition error            
            e2 = 0.0

            
            # for ind = 1:mp.batchsize       
            #     superβ = β + circshift(β, (0, ind))
            #     superν = ν + circshift(ν, (0, ind))
            #     superLν = m*superν
            #     e2 += errorL2(superLν, superβ)
            # end
            # e2 /= mp.batchsize 

            e3 = errorL2(B_DNN.layers[:dec](Lν), B) 
            # # solve for \nu in L\nu = \beta
            # Gb = m[1].weight\β
            Gb = m\β
            decμ =  μAE[:decoder]((Gb))
            e4 = errorL2(decμ, μ)
            e2 = errorL2(decμ[1,:]-decμ[end,:] - μ[1,:] + μ[end,:], μ[1,:] - μ[end,:])
            # @show e2, e3, e4
            # println("ls: $ls, e3: $e3, e4: $e4")           
            ls =  ls + e2 + e3 + e4
 
        end
       
        Flux.update!(Lopts, L, grads[1])
        push!(losses, ls)  # logging, outside gradient context
    end
    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")

    end
end

# G = inv(L[1].weight|>cpu)
G = inv(L|>cpu)
begin
    which = rand(1:size(latentdata.data[1],2))
    recon_ν = G*(B_DNN.layers[:enc](latentdata.data.Bs[:,which:which]|>mp.dev)|>cpu)
    plot(latentdata.data.νs[:,which], label="latent signal")
    a = plot!(recon_ν, label="reconstructed")
    title!("ν")
    plot(latentdata.data.μs[:,which], label="μ Truth")
    b = plot!(μAE.layers.decoder(recon_ν|>mp.dev), label="μ DG ")
    title!("μs")
    plot(a,b, size=(1200,800))
end

perfNN = Chain(Dense(mp.layers[end],mp.layers[end], Flux.tanh),                
                Dense(mp.layers[end],2,Flux.tanh),
                Dense(2,2, Flux.tanh))
perfNNstate = Flux.setup(Adam(0.01), perfNN)
# ps = hcat([perfs[p,:]./perfs[1,:] for p in 2:3]...)'
perfloader = DataLoader((νs=νs,perfs=perfs[2:3,:] .|>Float32), batchsize=512, shuffle=true)
begin
    losses = []
    for epoch = 1:mp.epochs/10
        for (ν, perf) in perfloader        
            ν = ν |> mp.dev
            perf = perf |> mp.dev
            ls = 0.0
            grads = Flux.gradient(perfNN) do m
                # Evaluate model and loss inside gradient context:
                ls = errorL2(m(ν), perf) 
            end
            Flux.update!(perfNNstate, perfNN, grads[1])
            push!(losses, ls)  # logging, outside gradient context
        end
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(losses[end])")
        end
    end
end

begin
    which = rand(1:199)
    c = plot(perfloader.data.νs[:,ns*which:ns*(which+1)], st=:contourf, label="input")
    plot!(c, colorbar=:false)
    title!("Latent")
    plot(perfloader.data.perfs[1,ns*which:ns*(which+1)], label="signal")
    a = plot!(perfNN(perfloader.data.νs[:,ns*which:ns*(which+1)])[1,:], label="")
    title!("Lift")
    plot(perfloader.data.perfs[2,ns*which:ns*(which+1)], label="signal")
    b = plot!(perfNN(perfloader.data.νs[:,ns*which:ns*(which+1)])[2,:], label="")
    title!("Thrust")
    plot(c,a,b, layout = (3,1), size = (1200,800))
    
end

begin
    # save all of the weights in Serialization
    # save the model parameters
    using JLD2
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

