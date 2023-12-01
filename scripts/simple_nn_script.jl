"""
This script assumes that you have run the script 'data_collection_script.jl'. We will be assuming that 
data follows the same shape as the data collected in that file. 

"""

using Flux
using DataDrivenDiffEq, DataDrivenSR
using Random, Statistics
using Surrogates, SurrogatesFlux
using SymbolicRegression
using Serialization, DataFrames

### LOAD DATA AND MESS AROUND
# data, header = readdlm(joinpath("data", "starter_data.csv"), ',', header=true)
data = deserialize(joinpath("data", "starter_data.jls"))

N = data[1, :σs] |> length

# Define a AutoEncoder architecture
# TODO: write factories for building these out faster
model = Chain(enc = Chain(Dense(N, N ÷ 2, Flux.relu),
        Dense(N ÷ 2, N ÷ 4, Flux.relu),
        Dense(N ÷ 4, N ÷ 8, Flux.relu)),
    latent = Chain(Dense(8, 8)),
    dec = Chain(Dense(N ÷ 8, N ÷ 4, Flux.relu),
        Dense(N ÷ 4, N ÷ 2, Flux.relu),
        Dense(N ÷ 2, N, Flux.relu)))
model(foil.σs) #does this kick out numbers?
# """ We have implemented named tuples for our model. This means we can easily 
#     process portions of our model. This will be useful when writing new loss functions.
# """
@assert model(foil.σs) == model.layers.dec(model.layers.latent(model.layers.enc(foil.σs)))
# You may often see people write these models in a more function approach, they are equivalent
@assert model(foil.σs) ==
        foil.σs |> model.layers.enc |> model.layers.latent |> model.layers.dec

##### CONVOLUTIONAL LAYERS

# Prepare and shape the data for input/output
σs = nothing
inputs = nothing
freqs = data[:, :reduced_freq]
ks    = data[:,:k]
for row in eachrow(data)
    U_inf = repeat([-row.U_inf, 0.0]', N)
    # DataFrame(St, reduced_freq, wave_number, U_inf,
    # σ , panel_velocity , position,
    # normals, wake_vel , tangents)
    x_in = [row.position' row.normals' row.wake_ind_vel' row.panel_vel' U_inf]
    x_in = reshape(x_in, N, 2, 5, 1) #shape the data to look like channels
    σs = isnothing(σs) ? row.σs : hcat(σs, row.σs) #<-- scale the σs with f*
    inputs = isnothing(inputs) ? x_in : cat(inputs, x_in; dims = 4)
end

BS = 64
dataloader = DataLoader((inputs, σs), batchsize = BS, shuffle = true)
#WHCN -> width height channels number
@show inputs |> size
conv_layers = Chain(
    Conv((3, 3), 5=>1, pad=(1,1), Flux.relu),
    MaxPool((1,2)),
    x-> reshape(x, (size(x,1),size(x, 4))))

conv_layers(inputs[:, :, :, 1:BS]) |> size

# Define the loss function
loss(x, y) = Flux.mse(model(x), y)

""" Now you have to do the hard part. Split up the data and train the network -> Dataloader might be useful. The current network `conv_layer` may not have enough layers,
feel free to expand on that.  """

errmse(x, y) = Flux.mse(x, y; agg = mean) 

opt_state = Flux.setup(Adam(0.01), conv_layers)
plot((σs./maximum(abs.(σs),dims=1))', label="")
losses = []
@time for epoch in 1:1_000
    for (x, y) in dataloader
        x = x 
        y = y 
        ls = 0.0
        grads = Flux.gradient(conv_layers) do m
            ls = errmse(m(x), y) #+ λ.*sum(pen_l2, Flux.params(m))
        end
        Flux.update!(opt_state, conv_layers, grads[1])
        push!(losses, ls)  # logging, outside gradient context
    end
    if epoch % 100 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end


# Attempt again, but with using dense networks
 #reshape inputs to be 2D
w,h,c,n = size(inputs)
dinputs = reshape(inputs, w*h*c, n )

model = Chain(Dense(w*h*c, w*h, Flux.tanh),
    Dense(w*h, w, Flux.tanh))
model(dinputs[:, 1:BS]) |> size

errmse(x, y) = Flux.mse(x, y; agg = mean) 

opt_state = Flux.setup(Adam(0.01), model)
ddataloader = DataLoader((dinputs, σs), batchsize = BS, shuffle = true)
losses = []
@time for epoch in 1:1_000
    for (x, y) in ddataloader
        x = x 
        y = y 
        ls = 0.0
        grads = Flux.gradient(model) do m
            ls = errmse(m(x), y) #+ λ.*sum(pen_l2, Flux.params(m))
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, ls)  # logging, outside gradient context
    end
    if epoch % 100 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end

begin
    #plot the signal and reconstructed signal./row.reduced_freq
    which = rand(1:size(ddataloader.data[1],2))
    @show which
    σ = ddataloader.data[2][:,which]
    plot(σ, label="Truth")
    plot!(model(ddataloader.data[1][:,which]), label="Reconstructed")
    # plot horizontal bars at -1 and 1
    plot!([1, length(σ)], [-1, -1], label="", color=:black)
    plot!([1, length(σ)], [1, 1], label="", color=:black)
end

begin
    this = data[which,:]
    for name in names(this)
        println(name,"   ", maximum(this[name]),"   ", minimum(this[name]))
    end

end



begin

    #shooting for a scaling of σs to be between -1 and 1, but as a function 
    # of input parameters -> f* = fc/U or k*= kc or λ (the compation of the wake)

    # plot((σs./maximum(abs.(σs),dims=1))', label="")

    a = plot()
    for (k,σ) in zip(freqs,eachcol(σs))
        if k < 1.0
            plot!(a, σ.*k, label="")
        else
            plot!(a, σ./k, label="")   
        end     
    end
    a
end