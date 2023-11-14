using Plots
using ForwardDiff
using Flux.Data: DataLoader
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using Statistics

# Define the number of signals
N = 4
# Define the time vector
dt = 0.01
t = LinRange(0,2π,128)

# Initialize the signal
coeffs = 1.0 ./[4, 3, 2, 1]

function sines(x; coeffs = ones(N))
    a = one(eltype(x))
    result = zero(eltype(x))
    for j = 1:N        
        result += coeffs[j] * sin(a * x)                
        a += one(eltype(x))
    end
    result
end
signal = sines.(t;coeffs = coeffs)

# Plot the signal
plot(t, signal ,label="signal")
plot!(t,sines.(t;coeffs=coeffs), label="sines")
# Calculate the derivative of the signal
derivative = [ForwardDiff.derivative(_t->sines(_t;coeffs=coeffs),_t) for _t in t]

# Plot the derivative
plot!(t, derivative ,label="Forward Diff")

dydt = diff(sines.(t;coeffs=coeffs))./dt
plot!(t[1:end-1], dydt, label = "CD")

# We have shown that the CD and FD methods are equivalent. We will use the FD method to calculate the derivative of the signal.
# Now to collect a gang of signals and their derivatives to train the neural network

# Define the number of signals to collect
step = 0.25
as = 0:step:1
bs = 0:step:1
cs = 0:step:1
ds = 0:step:1

ys = zeros(length(as)*length(bs)*length(cs)*length(ds), length(t))
dys = zeros(length(as)*length(bs)*length(cs)*length(ds), length(t))
index = 1
for (i,a) in enumerate(as), (j,b) in enumerate(bs), (k,c) in enumerate(cs), (l,d) in enumerate(ds)
    coeffs = [a,b,c,d]
    ys[index, :] = sines.(t;coeffs=coeffs)
    dys[index, :] = [ForwardDiff.derivative(_t->sines(_t;coeffs=coeffs),_t) for _t in t]
    index += 1
end


"""Write a simple AutoEncoder to learn the signal and then to learn
the derivative of the signal"""


# Encoder
encoder = Chain(
    Dense(length(t), 64, relu),
    Dense(64, 8, relu),
    Dense(8, 4,relu )
)

# Decoder
decoder = Chain(
    Dense(4,8,relu),
    Dense(8, 64, relu),
    Dense(64, length(t), relu)
)

# Autoencoder
autoencoder = Chain(encoder, decoder)

# autoencoder = Dense(length(t), length(t), relu)
#test a signal
@assert autoencoder(ys[42,:]) |>size == size(dys[42,:])
λ = 1e-3
errorl2(x, y) = Flux.mse(autoencoder(x), y) # + λ*sum(abs, Flux.params(autoencoder))


# Assuming `dataset` is your data and `batchsize` is the size of your batches
dataloader = DataLoader((data=ys'.|>Float32, label=dys'.|>Float32), batchsize=32, shuffle=true)
opt_state = Flux.setup(Adam(0.01), autoencoder)

losses = []
for epoch in 1:1_000
    for (x, y) in dataloader
        los, grads = Flux.withgradient(autoencoder) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            errorl2(y_hat, y)
        end
        Flux.update!(opt_state, autoencoder, grads[1])
        push!(losses, los)  # logging, outside gradient context
    end
end

plot(losses; xaxis=(:log10, "iteration"),
    yaxis="loss", label="per batch")

plot(dys[42,:], label="signal")
plot!(autoencoder(ys[42,:]), label="reconstructed signal")