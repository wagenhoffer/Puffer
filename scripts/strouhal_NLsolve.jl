using NLsolve

# Define the system of equations
function eq(out, x)        
    @show θ0, h0, St
    f, a = x
    t= 0:1/(100*f):1
    θ = θ0*sin.(2*π*f*t .- π/2)
    h = h0*sin.(2*π*f*t)
    out[1] = St -  a * f 
    out[2] = a - 2*maximum(h + sin.(θ)) 
end

# Initial guess for omega and a
x0 = [1.0, 0.1]
T = Float32 
strouhals = LinRange{T}(0.2, 0.4, 7)
td = LinRange{Int}(0, 10, 6)
θs = deg2rad.(td).|> T

hs = LinRange{T}(0.0, 0.25, 6)

# Given values
# θ0 = 0.0  # Specify the value of theta_0
# St = 0.4       # Specify the value of St
# h0 = 0.1       # Specify the value of h0
# U = 1.0        # Specify the value of U
fna = Dict()
for θ0 in θs
    for h0 in hs
        for St in strouhals
            # Call the solver
            if θ0 == 0 && h0 == 0.0
                continue
            else
                @show θ0, h0, St
                function eq!(out, x)        
                    
                    f, a = x
                    t= 0:1/(100*f):1
                    θ = θ0*sin.(2*π*f*t .- π/2)
                    h = h0*sin.(2*π*f*t)
                    out[1] = St -  a * f 
                    out[2] = a - 2*maximum(h + sin.(θ)) 
                end
                
                sol = nlsolve(eq!, [1.0,1.0])

                # Extract the solution
                f_min, a_min = sol.zero
                fna[(θ0, h0, St)] = (f_min, a_min)
            end
        end
    end
end

# Call the solver
sol = nlsolve(equations!, x0)

# Extract the solution
f_min, a_min = sol.zero


using ModelingToolkit, NonlinearSolve

# Define the variables
@variables f a

# Define the parameters
@parameters θ0 h0 St


eqs = [St - a*f, 
      a - 2*maximum(h0*sin.(2*π*f*t) + sin.(θ0*sin.(2*π*f.*t .- π/2)))]

# Create the nonlinear system
sys = NonlinearSystem(eqs, [f, a], [θ0, h0, St])

# Create the nonlinear solver
solver = NonlinearSolver(sys)

# Initial guess for f and a
u0 = [f => 1.0, a => 0.1]

strouhals = LinRange(0.2, 0.4, 7)
td = LinRange(0, 10, 6)
θs = deg2rad.(td)

hs = LinRange(0.0, 0.25, 6)

# Given values
# θ0 = 0.0  # Specify the value of theta_0
# St = 0.4  # Specify the value of St
# h0 = 0.1  # Specify the value of h0
# U = 1.0   # Specify the value of U
out = Dict()
for θ in θs
    for h in hs
        for s in strouhals
            # Call the solver
            p = [θ0 => θ, h0 => h, St => s]
            sol = solve(solver, u0, p)

            # Extract the solution
            f_min, a_min = sol.u
            out[(θ, h, s)] = (f_min, a_min)
        end
    end
end
out