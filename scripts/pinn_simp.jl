
using Optimisers, Zygote, Plots, Random, Distributions
using Flux
using ReverseDiff
using ForwardDiff
SEED = 42
N_collocation_points = 50
HIDDEN_DEPTH = 100
LEARNING_RATE = 1e-3
N_EPOCHS = 20_000
BC_LOSS_WEIGHT = 100.0
rhs_function(x) = sin(π * x)
analytical_solution(x) = sin(π * x) / π^2


rng = MersenneTwister(SEED)
sigmoid(x) = 1.0 / (1.0 + exp(-x))

# Initialize the weights according to the Xavier Glorot initializer
uniform_limit = sqrt(6 / (1 + HIDDEN_DEPTH))
W = rand(
    rng,
    Uniform(-uniform_limit, +uniform_limit),
    HIDDEN_DEPTH,
    1,
)
V = rand(
    rng,
    Uniform(-uniform_limit, +uniform_limit),
    1,
    HIDDEN_DEPTH,
)
b = zeros(HIDDEN_DEPTH)
fN = Chain(Dense(1, HIDDEN_DEPTH, sigmoid), Dense(HIDDEN_DEPTH, 1, sigmoid))
    # Dense(HIDDEN_DEPTH, 1, sigmoid)
# fN.weight .= W'
# fN.bias .= b
parameters = (; W, V, b)
network_forward(x, p) = p.V * sigmoid.(p.W * x .+ p.b)

x_line = reshape(collect(range(0.0, stop=1.0, length=100)), (1, 100))

# Plot initial prediction of the network (together with the analytical solution)
plot(x_line[:], network_forward(x_line, parameters)[:], label="initial prediction")
plot!(x_line[:], analytical_solution.(x_line[:]), label="analytical_solution")

function network_output_and_first_two_derivatives(x, p)
    activated_state = sigmoid.(p.W * x .+ p.b)
    sigmoid_prime = activated_state .* (1.0 .- activated_state)
    sigmoid_double_prime = sigmoid_prime .* (1.0 .- 2.0 .* activated_state)

    output = p.V * activated_state
    first_derivative = (p.V .* p.W') * sigmoid_prime
    second_derivative = (p.V .* p.W' .* p.W') * sigmoid_double_prime

    return output, first_derivative, second_derivative
end

_output, _first_derivative, _second_derivative = network_output_and_first_two_derivatives(x_line, parameters)

_zygote_first_derivative = Zygote.gradient(x -> sum(network_forward(x, parameters)), x_line)[1]
_first_derivative


interior_collocation_points = rand(rng, Uniform(0.0, 1.0), (1, N_collocation_points))
boundary_collocation_points = [0.0 1.0]

function loss_forward(p)
    output, first_derivative, second_derivative = network_output_and_first_two_derivatives(
        interior_collocation_points,
        p,
    )

    interior_residuals = second_derivative .+ rhs_function.(interior_collocation_points)

    interior_loss = 0.5 * mean(interior_residuals.^2)

    boundary_residuals = network_forward(boundary_collocation_points, p) .- 0.0

    boundary_loss = 0.5 * mean(boundary_residuals.^2)

    total_loss = interior_loss + BC_LOSS_WEIGHT * boundary_loss

    return total_loss
end
dur(m, xs) = gradient(x->sum(m(x)), xs)
    # ddur(xs) = ReverseDiff.hessian(x->sum(fN(x)), xs)
dduf(m, xs) = ForwardDiff.hessian(x->sum(m(x)), xs)
function loss_forward_2(;fN=fN)
    u = fN(interior_collocation_points)              
    dudx = dur(fN, interior_collocation_points)
    # @time d2udx2 = ddur(interior_collocation_points)
    # @time begin
    #     d2udx2 = dduf(fN, interior_collocation_points)    
    #     d2udx2 = [d2udx2[i,i] for i=1:length(interior_collocation_points)]        
    # end
    d2udx2 = [dduf(fN, [x])[1] for x in interior_collocation_points]
                
    interior_residuals = d2udx2' .+ rhs_function.(interior_collocation_points)

    interior_loss = 0.5 * mean(interior_residuals.^2)

    boundary_residuals = fN(boundary_collocation_points) .- 0.0

    boundary_loss = 0.5 * mean(boundary_residuals.^2)

    total_loss = interior_loss + BC_LOSS_WEIGHT * boundary_loss

    return total_loss
end
@time loss_forward_2()
function forward_over_reverse_hessian(f,θ)
    ForwardDiff.jacobian(θ) do θ
      Zygote.gradient(x -> f(x), θ)[1]
    end
  end
 
loss_forward(parameters)
loss_forward_2()
# out, back = Zygote.pullback(loss_forward, parameters)
out, back = Zygote.pullback(loss_forward_2)
back(1.0)[1]
opt = Adam(LEARNING_RATE)
opt_state = Optimisers.setup(opt, parameters)
loss_history = []
for i in 1:500
    loss, back = Zygote.pullback(loss_forward, parameters)
    push!(loss_history, loss)
    grad, = back(1.0)
    opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
    if i % 100 == 0
        println("Epoch: $i, Loss: $loss")
    end
end

plot(loss_history, yscale=:log10)
plot(x_line[:], network_forward(x_line, parameters)[:], label="final prediction")
plot!(x_line[:], analytical_solution.(x_line[:]), label="analytical_solution")

dfdx(x) = gradient(fn,x)[1]
using Flux
fNState = Flux.setup(Flux.Adam(0.01), fN)

for epoch = 1:1
    total_loss = 0.0
    grads = Flux.gradient(fN) do m
        # Evaluate model and loss inside gradient context:                
        u = m(interior_collocation_points)        
        dudx = ReverseDiff.gradient(x->sum(m(x)), interior_collocation_points)
        d2udx2 = ReverseDiff.hessian(x->sum(m(x)),interior_collocation_points)
        d2udx2 = [d2udx2[i,i] for i=1:length(interior_collocation_points)]                    
        interior_residuals = d2udx2' .+ rhs_function.(interior_collocation_points)    
        interior_loss = 0.5 * mean(interior_residuals.^2)    
        boundary_residuals = fN(boundary_collocation_points) .- 0.0    
        boundary_loss = 0.5 * mean(boundary_residuals.^2)    
        total_loss = interior_loss + BC_LOSS_WEIGHT * boundary_loss                      
    end
    Flux.update!(b2perfstate, b2perf, grads[1])
    push!(losses, total_loss)  # logging, outside gradient context
end