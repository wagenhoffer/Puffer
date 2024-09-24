using Plots
using Flux.Data: DataLoader
using Flux
using Flux: params
using NeuralOperators
using Statistics
using LinearAlgebra
using CUDA
using Serialization, DataFrames
using Random

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

nsamples = 1_000
indices = rand(1:size(RHS,1),nsamples)
xtrain = RHS[indices]
ytrain = μs[indices]

loss(X,Y, sensor) = Flux.mse(m(X,sensor), Y)

opt = Adam(0.01)

m = NOMAD((64, 64), (64, 64), tanh, tanh)
grid = position
Flux.@epochs 10 Flux.train!(loss, params(m), data, opt)