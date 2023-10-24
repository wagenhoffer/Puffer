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
for row in eachrow(data)
    U_inf = repeat([-row.U_inf, 0.0]', N)
    # DataFrame(St, reduced_freq, wave_number, U_inf,
    # σ , panel_velocity , position,
    # normals, wake_vel , tangents)
    x_in = [row.position' row.normals' row.wake_ind_vel' row.panel_vel' U_inf]
    x_in = reshape(x_in, N, 2, 5, 1) #shape the data to look like channels
    σs = isnothing(σs) ? row.σs : vcat(σs, row.σs)
    inputs = isnothing(inputs) ? x_in : cat(inputs, x_in; dims = 4)
end

#WHCN -> width height channels number
@show inputs |> size
conv_layer = Conv((1, 1), 5 => 1)

conv_layer(x_in)

# Define the loss function
loss(x, y) = Flux.mse(model(x), y)

""" Now you have to do the hard part. Split up the data and train the network -> Dataloader might be useful. The current network `conv_layer` may not have enough layers,
feel free to expand on that.  """
