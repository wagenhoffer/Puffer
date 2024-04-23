using Plots
using ForwardDiff
using Flux.Data: DataLoader
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using Statistics
using LinearAlgebra
using CUDA

# GPU config
if CUDA.has_cuda()
    dev = gpu
    @info "Training on GPU"
else
    dev = cpu
    @info "Training on CPU"
end
"""
    sines(x; coeffs = ones(N))

    Compute a signal based on a sum of sinusoids.

    # Arguments
    - `x`: Input vector.
    - `coeffs`: Coefficients for the sinusoids.

    # Returns
    - Computed signal.
"""
function sines(x; coeffs = ones(N))
    a = one(eltype(x))
    result = zero(eltype(x))
    for j = 1:N        
        result += coeffs[j] * sin(a * x)                
        a += one(eltype(x))
    end
    result ./ N
end

function powers(x; coeffs = ones(N))    
    result = zero(eltype(x))
    for j = 1:N        
        result += coeffs[j] *x^(j-1)        
    end
    result
end

function build_ae(input_size, hidden_size = 8, activation = tanh)
    encoder = Chain(Dense(input_size, hidden_size, activation))        
    decoder = Chain(Dense(hidden_size, input_size, activation))  
    Chain(encoder = encoder,decoder= decoder)   
end


function ExpoRange(start, stop) 
    start = log2(start) .|> Int
    stop = log2(stop) .|> Int
    steps = stop - start + 1
    exp2.(LinRange(start, stop, steps)).|>ceil.|>Int
end
ExpoRange(4,128)

function build_ae_layers(layer_sizes, activation = tanh)    
    decoder = Chain([Dense(layer_sizes[i+1], layer_sizes[i], activation) 
                    for i = length(layer_sizes)-1:-1:1]...)    
    encoder = Chain([Dense(layer_sizes[i], layer_sizes[i+1], activation)
                    for i = 1:length(layer_sizes)-1]...)
    Chain(encoder = encoder, decoder = decoder)
end


#COLLECT Data
# Define the number of signals to collect
t = LinRange(-1,1,64)

step = 0.125/2.0
as = -0.25:step:0.25
bs = -0.25:step:0.25
cs = -0.25:step:0.25
ds = -0.25:step:0.25
N =4

ys = zeros(length(as)*length(bs)*length(cs)*length(ds), length(t))
dys = zeros(length(as)*length(bs)*length(cs)*length(ds), length(t))
sine_coeff_dict = Dict()
index = 1
for (i,a) in enumerate(as), (j,b) in enumerate(bs), (k,c) in enumerate(cs), (l,d) in enumerate(ds)
    coeffs = [a,b,c,d]
    sine_coeff_dict[index] = coeffs
    ys[index, :] = sines.(t; coeffs=coeffs)
    dys[index, :] = [ForwardDiff.derivative(_t->sines(_t;coeffs=coeffs),_t) for _t in t]
    index += 1
end
plot(ys', label="")

yps = zeros(length(as)*length(bs)*length(cs)*length(ds), length(t))
dyps = zeros(length(as)*length(bs)*length(cs)*length(ds), length(t))
poly_coeff_dict = Dict()
index = 1 
for (i,a) in enumerate(as), (j,b) in enumerate(bs), (k,c) in enumerate(cs), (l,d) in enumerate(ds)
    coeffs = [a,b,c,d]
    poly_coeff_dict[index] = coeffs
    yps[index, :] = powers.(t;coeffs=coeffs)
    dyps[index, :] = [ForwardDiff.derivative(_t->powers(_t;coeffs=coeffs),_t) for _t in t]
    index += 1
end
coeff_dict = Dict()
coeff_dict["sines"] = sine_coeff_dict
coeff_dict["polynomials"] = poly_coeff_dict

""" 
Put both datasets into a dataloader
"""


dataloader =    DataLoader((y=vcat(ys,yps)'.|>Float32, yp=vcat(dys, dyps)'.|>Float32), batchsize=128, shuffle=true)
# dataloader = DataLoader((yps,dyps), batchsize=64, shuffle=true)
autoencoder = build_ae(length(t), 4 )
autoencoder = build_ae_layers([length(t), length(t)÷2, 4]) |> dev
@assert autoencoder(ys[42,:]) |>size == size(dys[42,:])


λ = 1e-2
errorl2(x, y) = Flux.mse(x, y; agg = mean) 
errorLA(x, y) = norm(x-y,2)/norm(y,2) |> dev
pen_l1(x) = sum(abs2, x)/2
errorLasso(m,x,y) = Flux.mse(x, y) + λ.*sum(pen_l1, Flux.params(m))


@time errorLasso(autoencoder, ys[42,:], dys[42,:])
opt_state = Flux.setup(Adam(0.01), autoencoder)

losses = []
@time for epoch in 1:1_000
    for (x, y) in dataloader
        x = x |> dev
        y = y |> dev
        ls = 0.0
        grads = Flux.gradient(autoencoder) do m
            # Evaluate model and loss inside gradient context:
            ls = errorl2(m(x), y) #+ λ.*sum(pen_l2, Flux.params(m))
            # loss = errorLA(m(x), y) #+ λ.*sum(pen_l2, Flux.params(m))
            # ls = errorLasso(m,x,y)
            # @show grads|>size
        end
        Flux.update!(opt_state, autoencoder, grads[1])
        push!(losses, ls)  # logging, outside gradient context
    end
    if epoch % 100 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end

# Plot the loss function and a sample of reconstructed signals; rerun to see how its doing
begin
    which = rand(1:size(dataloader.data.y,2))
    dataset = which <= size(ys,1) ? "sines" : "polynomials"
    @show which, dataset , coeff_dict[dataset][which % size(ys,2)]
    a = plot(losses; xaxis=(:log10, "iteration"),
        yaxis="loss", label="per batch")    
    b = plot(dataloader.data.yp[:,which], label="signal")
    @show Flux.mse(autoencoder(dataloader.data.y[:,which]|>cpu)|>cpu, dataloader.data.yp[:,which]|>cpu)|>cpu
    @show norm(autoencoder(dataloader.data.y[:,which])- dataloader.data.yp[:,which],2)/norm(dataloader.data.yp[:,which],2)
    title!(b, dataset)
    plot!(b, autoencoder(dataloader.data.y[:,which]), label="reconstructed signal")
    plot(a,b, size=(1200,800))
end

errorLA(autoencoder(ys[42,:]), dys[42,:])
errorl2(autoencoder(ys[42,:]), dys[42,:])
errorLasso(autoencoder, ys[42,:], dys[42,:])