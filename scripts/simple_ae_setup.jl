using Plots
using ForwardDiff
# Define the number of signals
N = 4

# Define the time vector
dt = 0.01
t = 0:dt:2*pi

# Initialize the signal


function sines(x)
    a = one(eltype(x))
    result = zero(eltype(x))
    for j = 1:N
        
        result += sin(a * x)        
        
        a += one(eltype(x))
    end
    result
end
signal = sines.(t)

# Plot the signal
plot(t, signal)
plot!(t,sines.(t))
# Calculate the derivative of the signal
derivative =[]
for _t in t
    push!(derivative, ForwardDiff.derivative(sines,_t))
end
# Plot the derivative
plot!(t, derivative)

dydt = diff(sines.(t))./dt
plot!(t[1:end-1], dydt)


"""Write a simple AutoEncoder to learn the signal and then to learn
the derivative of the signal"""

using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using Statistics

# Reshape the signal data for model compatibility
signal_reshaped = reshape(signal, (1, length(signal)))

# Normalize the data
signal_mean = mean(signal_reshaped)
signal_max = maximum(signal_reshaped)
signal_normalized = (signal_reshaped .- signal_mean) / signal_max

# Encoder
encoder = Chain(
    Dense(length(t), 64, σ),
    Dense(64, 32, σ),
    Dense(32, 16, σ)
)

# Decoder
decoder = Chain(
    Dense(16, 32, σ),
    Dense(32, 64, σ),
    Dense(64, length(t), σ)
)

# Autoencoder
autoencoder = Chain(encoder, decoder)


loss(x, y) = Flux.mse(autoencoder(x), y)
dataset = [(signal_normalized', signal_normalized')]

opt_state = Flux.setup(Adam(), autoencoder)
ps = Flux.params(autoencoder)
for epoch in 1:100
    Flux.train!(autoencoder, dataset, opt_state) do m, x, y
      loss(m(x), y)
    end
end

plot(t, autoencoder(t))

# Assuming you have a model architecture (similar to above) for the derivative learning
derivative_model = Chain( ... )

# Train the derivative model with the derived data
# loss_derivative(x) = Flux.mse(derivative_model(x), expected_derivative)
# dataset_derivative = [(signal_normalized, expected_derivative)]
# train!(loss_derivative, params(derivative_model), dataset_derivative, opt)
