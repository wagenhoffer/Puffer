using LinearAlgebra
using Plots

# Define the circle
struct Circle
    center::Vector{Float64}
    radius::Float64
end

# Define the boundary element
struct BoundaryElement
    node1::Vector{Float64}
    node2::Vector{Float64}
    circle_index::Int
end

# Generate boundary elements for multiple circles
function generate_boundary_elements(circles::Vector{Circle}, n_elements_per_circle::Int)
    elements = BoundaryElement[]
    for (circle_index, circle) in enumerate(circles)
        for i in 1:n_elements_per_circle
            θ1 = 2π * (i - 1) / n_elements_per_circle
            θ2 = 2π * i / n_elements_per_circle
            node1 = circle.center + circle.radius * [cos(θ1), sin(θ1)]
            node2 = circle.center + circle.radius * [cos(θ2), sin(θ2)]
            push!(elements, BoundaryElement(node1, node2, circle_index))
        end
    end
    return elements
end

# Define the boundary condition (example: cosine variation)
function boundary_condition(θ::Float64, circle_index::Int)
    return cos(θ + circle_index * π/4)  # Different phase for each circle
end

# Compute the influence coefficient for external problem
function influence_coefficient(x::Vector{Float64}, element::BoundaryElement)
    r1 = x - element.node1
    r2 = x - element.node2
    L = norm(element.node2 - element.node1)
    return -1/(2π) * (log(norm(r2)) - log(norm(r1))) * L
end

# Assemble the system matrix and right-hand side vector
function assemble_system(elements::Vector{BoundaryElement}, bc=boundary_condition)
    n = length(elements)
    A = zeros(n, n)
    b = zeros(n)
    
    for i in 1:n
        x_mid = 0.5 * (elements[i].node1 + elements[i].node2)
        θ = atan(x_mid[2] - elements[i].node1[2], x_mid[1] - elements[i].node1[1])
        b[i] = bc(θ, elements[i].circle_index)
        
        for j in 1:n
            if i != j
                A[i, j] = influence_coefficient(x_mid, elements[j])
            else
                A[i, i] = -0.5
            end
        end
    end
    
    return A, b
end

# Solve the boundary integral equation
function solve_laplacian_bem(circles::Vector{Circle}, n_elements_per_circle::Int,bc=boundary_condition)
    elements = generate_boundary_elements(circles, n_elements_per_circle)
    A, b = assemble_system(elements,bc)
    solution = A \ b
    elements, solution, b
end

# Function to compute potential at a point
function compute_potential(x::Vector{Float64}, elements::Vector{BoundaryElement}, solution::Vector{Float64})
    potential = 0.0
    for (i, element) in enumerate(elements)
        potential += solution[i] * influence_coefficient(x, element)
    end
    return potential
end

# Check if a point is inside any circle
function is_inside_any_circle(x::Vector{Float64}, circles::Vector{Circle})
    for circle in circles
        if norm(x - circle.center) < circle.radius
            return true
        end
    end
    false
end

# Interpolate solution on a grid
function interpolate_solution(elements, solution, grid_size, circles)
    x_min, x_max = extrema(hcat(extrema([circle.center[1] .+ [-2, 2].*circle.radius for circle in circles])...))
    y_min, y_max = extrema(hcat(extrema([circle.center[2] .+ [-2, 2].*circle.radius for circle in circles])...))
    
    x = range(x_min, x_max, length=grid_size)
    y = range(y_min, y_max, length=grid_size)
    z = zeros(grid_size, grid_size)

    for i in 1:grid_size
        for j in 1:grid_size
            point = [x[i], y[j]]
            if !is_inside_any_circle(point, circles)
                z[j, i] = compute_potential(point, elements, solution)
            else
                z[j, i] = NaN  # Set points inside any circle to NaN
            end
        end
    end

    x, y, z
end

#EXAMPLE OF SEVERAL CIRCLES
# Define multiple circles
circles = [
    Circle([0.0, 0.0], 1.0),   # Unit circle at origin
    Circle([3.0, 0.0], 0.8),   # Smaller circle to the right
    Circle([-1.5, 2.0], 1.2)   # Larger circle to the top-left
]
n_elements_per_circle = 100
grid_size = 300

elements, solution, b = solve_laplacian_bem(circles, n_elements_per_circle)

# Interpolate solution on a grid
x, y, z = interpolate_solution(elements, solution, grid_size, circles)

# Create the plot
p = contourf(x, y, z, 
                color=:viridis, 
                linewidth=0,
                title="External Laplacian Field around Multiple Circles",
                xlabel="x",
                ylabel="y",
                aspect_ratio=:equal,
                size=(1000, 800))

# Add circle outlines
for (i, circle) in enumerate(circles)
    θ = range(0, 2π, length=100)
    circle_x = circle.center[1] .+ circle.radius * cos.(θ)
    
end
   
using LinearAlgebra
using Plots

# Define the circle
struct Circle
    center::Vector{Float64}
    radius::Float64
end

# Define the boundary element
struct BoundaryElement
    node1::Vector{Float64}
    node2::Vector{Float64}
    circle_index::Int
end

# Generate boundary elements for multiple circles
function generate_boundary_elements(circles::Vector{Circle}, n_elements_per_circle::Int)
    elements = BoundaryElement[]
    for (circle_index, circle) in enumerate(circles)
        for i in 1:n_elements_per_circle
            θ1 = 2π * (i - 1) / n_elements_per_circle
            θ2 = 2π * i / n_elements_per_circle
            node1 = circle.center + circle.radius * [cos(θ1), sin(θ1)]
            node2 = circle.center + circle.radius * [cos(θ2), sin(θ2)]
            push!(elements, BoundaryElement(node1, node2, circle_index))
        end
    end
    return elements
end

# Define the boundary condition (example: cosine variation)
function boundary_condition(θ::Float64, circle_index::Int)
    return cos(θ + circle_index * π/4)  # Different phase for each circle
end

# Compute the influence coefficient for external problem
function influence_coefficient(x::Vector{Float64}, element::BoundaryElement)
    r1 = x - element.node1
    r2 = x - element.node2
    L = norm(element.node2 - element.node1)
    return -1/(2π) * (log(norm(r2)) - log(norm(r1))) * L
end

# Assemble the system matrix and right-hand side vector
function assemble_system(elements::Vector{BoundaryElement}, bc=boundary_condition)
    n = length(elements)
    A = zeros(n, n)
    b = zeros(n)
    
    for i in 1:n
        x_mid = 0.5 * (elements[i].node1 + elements[i].node2)
        θ = atan(x_mid[2] - elements[i].node1[2], x_mid[1] - elements[i].node1[1])
        b[i] = bc(θ, elements[i].circle_index)
        
        for j in 1:n
            if i != j
                A[i, j] = influence_coefficient(x_mid, elements[j])
            else
                A[i, i] = -0.5
            end
        end
    end
    
    return A, b
end

# Solve the boundary integral equation
function solve_laplacian_bem(circles::Vector{Circle}, n_elements_per_circle::Int,bc=boundary_condition)
    elements = generate_boundary_elements(circles, n_elements_per_circle)
    A, b = assemble_system(elements,bc)
    solution = A \ b
    elements, solution, b
end

# Function to compute potential at a point
function compute_potential(x::Vector{Float64}, elements::Vector{BoundaryElement}, solution::Vector{Float64})
    potential = 0.0
    for (i, element) in enumerate(elements)
        potential += solution[i] * influence_coefficient(x, element)
    end
    return potential
end

# Check if a point is inside any circle
function is_inside_any_circle(x::Vector{Float64}, circles::Vector{Circle})
    for circle in circles
        if norm(x - circle.center) < circle.radius
            return true
        end
    end
    false
end

# Interpolate solution on a grid
function interpolate_solution(elements, solution, grid_size, circles)
    x_min, x_max = extrema(hcat(extrema([circle.center[1] .+ [-2, 2].*circle.radius for circle in circles])...))
    y_min, y_max = extrema(hcat(extrema([circle.center[2] .+ [-2, 2].*circle.radius for circle in circles])...))
    
    x = range(x_min, x_max, length=grid_size)
    y = range(y_min, y_max, length=grid_size)
    z = zeros(grid_size, grid_size)

    for i in 1:grid_size
        for j in 1:grid_size
            point = [x[i], y[j]]
            if !is_inside_any_circle(point, circles)
                z[j, i] = compute_potential(point, elements, solution)
            else
                z[j, i] = NaN  # Set points inside any circle to NaN
            end
        end
    end

    x, y, z
end

#EXAMPLE OF SEVERAL CIRCLES
# Define multiple circles
circles = [
    Circle([0.0, 0.0], 1.0),   # Unit circle at origin
    Circle([3.0, 0.0], 0.8),   # Smaller circle to the right
    Circle([-1.5, 2.0], 1.2)   # Larger circle to the top-left
]
n_elements_per_circle = 100
grid_size = 300

elements, solution, b = solve_laplacian_bem(circles, n_elements_per_circle)

# Interpolate solution on a grid
x, y, z = interpolate_solution(elements, solution, grid_size, circles)

# Create the plot
p = contourf(x, y, z, 
                color=:viridis, 
                linewidth=0,
                title="External Laplacian Field around Multiple Circles",
                xlabel="x",
                ylabel="y",
                aspect_ratio=:equal,
                size=(1000, 800))

# Add circle outlines
for (i, circle) in enumerate(circles)
    θ = range(0, 2π, length=100)
    circle_x = circle.center[1] .+ circle.radius * cos.(θ)
    circle_y = circle.center[2] .+ circle.radius * sin.(θ)
    plot!(p, circle_x, circle_y, color=:white, linewidth=2, label="Circle $i")
end

# Show the plot
p

#START PARAMETERIZATION OF A SINGLE CIRCLE WITH A INCREASING SERIES BOUNDARY CONDITION
# Define a single circle
circle = Circle([0.0, 0.0], 1.0)
n_elements_per_circle = 100
grid_size = 300
boundary_condition(θ::Float64) = boundary_condition(θ::Float64, 1) 

macro generate_bc(a, b, c, d)
    quote
        (θ, i) -> $a * sin($b * θ) + $c * cos($d * θ)
    end
end




# Example usage of the macro

ass = collect(0:0.4:2π)[2:end]
css = collect(0:0.4:2π)[2:end]
bss = collect(1:5)
dss = collect(0:5)
total_parameters = length(ass) * length(bss) * length(css) * length(dss)
# Iterate over the boundary conditions and solve the BEM for each

i = 1
xs = zeros(n_elements_per_circle, total_parameters)
bs = zeros(n_elements_per_circle, total_parameters)
@time for a in a_s
    for b in bs
        for c in cs 
            for d in ds
                # @show a, b, c, d
                bc(θ, ii) =   a * sin(b * θ) + c * cos(d * θ)
                elements, solution, B = solve_laplacian_bem([circle], n_elements_per_circle,bc )
                xs[:, i] = solution
                bs[:, i] = B
                i += 1
            end
        end
    end
end

#Setup a simple NN and train it 
using Flux
using Flux: DataLoader
using CUDA 

dataloader = DataLoader((xs = xs.|>Float32, bs = bs.|>Float32), batchsize = 1024, shuffle = true)

x_ae = Chain(enc = Chain(Dense(100,50, tanh), 
             Dense(50, 25, tanh), 
             Dense(25,8)), 
             dec = Chain(Dense(8,25, tanh),
             Dense(25, 50, tanh), 
             Dense(50, 100)))|>gpu
b_ae = deepcopy(x_ae)             

errorL2(x, y) = Flux.mse(x,y)/Flux.mse(y, 0.0) 
losses = []
bstate = Flux.setup(Adam(0.001), b_ae)
for epoch = 1:1_000
    for (x,b) in dataloader        
        b = b |> gpu     
        ls = 0.0
        grads = Flux.gradient(b_ae) do m        
            
            ls = errorL2(m(b), b)              
            
        end
        Flux.update!(bstate, b_ae, grads[1])
        push!(losses, ls)  # logging, outside gradient context        
    end
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $(losses[end])")
    end
end
xstate = Flux.setup(Adam(0.001), x_ae)
for epoch = 1:1_000
    for (x,b) in dataloader        
        x = x |> gpu     
        ls = 0.0
        grads = Flux.gradient(x_ae) do m        
            
            ls = errorL2(m(x), x)              
            
        end
        Flux.update!(xstate, x_ae, grads[1])
        push!(losses, ls)  # logging, outside gradient context        
    end
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $(losses[end])")
    end
end
β = b_ae[1](bs|>gpu)|>gpu
ξ = x_ae[1](xs|>gpu)|>gpu
loader = DataLoader((ξ = ξ.|>Float32, β = β.|>Float32, xs = xs.|>Float32, bs = bs.|>Float32),
                     batchsize = 1024, shuffle = true)
L = rand(Float32, 8,8)|>gpu
Lstate = Flux.setup(Adam(0.001), L)|>gpu
losses = []
for epoch = 1:1_000
    for (ξ, β, xs, bs) in loader
        ξ = ξ |> gpu
        β = β |> gpu
        xs = xs |> gpu
        bs = bs |> gpu
        ls = 0.0
        grads = Flux.gradient(L) do m
            Lξ = m*ξ
            Gβ = m\β
            ls  = errorL2(Lξ, β)
            ls += errorL2(b_ae[2](Lξ), bs)
            ls += errorL2(x_ae[2](Gβ), xs)
             
        end
        Flux.update!(Lstate, L, grads[1])
        push!(losses, ls)  # logging, outside gradient context
    end
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $(losses[end])")
    end
end