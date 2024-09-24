using Flux, Plots, Statistics
using Base.Iterators

# generate the data
n = 1_000 # sample size
σ = 0.05  # true std. dev. of the shocks
# data = Float32.(AR1(n, σ))
data = Float32.(sin.(LinRange(0, 2π, n))*.25 .+ σ*randn(n) )
plot(data, marker=:circle, label="data")
# set up the training
batchsize = 3  # remember, MA model is only predictable one step out
epochs = 100   # number of training loops through data

# the model: this is just an initial guess
# need to experiment with this
# model = Chain(LSTM(batchsize, 10), Dense(10,2, tanh), Dense(2,batchsize))
model = Chain(
    # RNN(batchsize => 32, tanh),
    GRU(batchsize => 32),
    Dense(32 => batchsize, tanh),    
)

function predict(data, batchsize)
    # n = size(data,1)
    yhat = zeros(Float32, n)
    for t = batchsize+1:n -1   
        Flux.reset!(model)
        yhat[t] = model(X[t])[1]
    end
    yhat
end


# the first element of the batched data is one lag
# of the second element, in chunks of batchsize. So,
# we are doing one-step-ahead forecasting, conditioning
# on batchsize lags
X = [data[i:i+batchsize-1] for i in 1:n-batchsize] #.|> Float32
Y = [data[i+1:i+batchsize] for i in 1:n-batchsize] #.|> Float32

# Flux.@epochs epochs Flux.train!(loss,Flux.params(m), batches, ADAM())

epochs = 100
opt = ADAM()
θ = Flux.params(model) # Keep track of the model parameters
ls = 0.0
losses = []
for epoch ∈ 1:epochs*10 # Training loop
    Flux.reset!(model) # Reset the hidden state of the RNN
    # Compute the gradient of the mean squared error loss
    ∇ = gradient(θ) do
        # ls = Flux.mse.([model(x) for x ∈ X[2:end]], Y[2:end])./Flux.mse.(Y[2:end],0.0) |> mean
        ls = Flux.mse.([model(x) for x ∈ X[2:end]], Y[2:end]) |> mean
    end
    push!(losses, ls)
    Flux.update!(opt, θ, ∇) # Update the parameters
end
# plot(losses)

begin
    # pred = predict(X,batchsize)
    Flux.reset!(model)
    pred  = vcat([model(x)[2] for x in vcat(X,X)]...)
    yflat = vcat([y[1] for y in Y]...)
    error = yflat - pred[length(yflat)+1:end]
    @show sum(abs2,error)/ sum(abs2, yflat)
    println("true std. error of noise: ", σ)
    println("std. error of forecast: ", std(error))
    println("std. error of data: ", std(data))
    plot(1:n, data)
    a = plot!(batchsize+1:n,  pred[length(yflat)+1:end])
    
    # plot!(a, 1:n, error, label="error")
end

half  = length(pred) ÷2
plot(pred[1:half], label="pred")
plot!(pred[half+1:end], label="pred")
plot!(yflat, label="data")
plot!(pred[1:half] - pred[half+1:end], label="error")
