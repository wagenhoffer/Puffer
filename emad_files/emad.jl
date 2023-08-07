include("../src/BemRom.jl")
using Plots, CSV, DataFrames

ang = deepcopy(defaultDict)
ang[:N] = 64      #number of panels (elements) to discretize the foil
ang[:Nt] = 64     #number of timesteps per period of motion
ang[:Ncycles] = 3 #number of periods to simulate
ang[:f] = 1.0      #frequency of wave motion 
ang[:Uinf] = 1.0    #free stream velocity 
ang[:kine] = :make_ang 
a0 = 0.1 # how much the leading edge heaves up and down wrt the chord(length) of the swimmer
ang[:motion_parameters] = a0


begin
    foil, flow = init_params(;ang...)
    k = foil.f*foil.chord/flow.Uinf
    @show k
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)        
        phi =  get_phi(foil, wake)    # (output space) <-probably not that important                            
        p = panel_pressure(foil, flow,  old_mus, old_phis, phi)    
        # (output space) <- p is a function of μ and we should be able to recreate this     
        old_mus = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] = get_performance(foil, flow, p)
        # the coefficients of PERFROMANCE are important, but are just a scaling of P
        # if we can recreate p correctly, this will be easy to get also (not important at first)
        ps[:,i] = p # storage container of the output, nice!
    end
    t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs[1,start:end], label="Force"  ,lw = 3, marker=:circle)
    b = plot(t[start:end], coeffs[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
    c = plot(t[start:end], coeffs[3,start:end], label="Thrust" ,lw = 3, marker=:circle)
    d = plot(t[start:end], coeffs[4,start:end], label="Power"  ,lw = 3, marker=:circle)
    plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(800,800))
end

begin
    # watch a video of the motion, does it blow up? if so, what went wrong? 
    foil, flow = init_params(;ang...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:flow.Ncycles*flow.N*1.75
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)       
        win=nothing
        f = plot_current(foil, wake;window=win)
        plot!(f, ylims=(-1,1))
        
    end
    gif(movie, "handp.gif", fps=10)
end

function save_data(input_data, output_data, output_dir)
    # Save input data
    input_file = joinpath(output_dir, "input_data.csv")
    CSV.write(input_file, input_data)

    # Save output data
    output_file = joinpath(output_dir, "output_data.csv")
    CSV.write(output_file, output_data)
end





function run_simulations(output_dir)
    # Define parameter ranges
    Strouhal_values = [0.1, 0.2, 0.3]
    reduced_freq_values = [0.1, 0.2, 0.3]
    wave_number_values = [0.1, 0.2, 0.3]

    allin = Vector{DataFrame}()
    allout = Vector{DataFrame}()

    # Nested loops to vary parameters
    for St in Strouhal_values
        for reduced_freq in reduced_freq_values
            for wave_number in wave_number_values
                # Set motion parameters
                ang = deepcopy(defaultDict)
                ang[:N] = 64
                ang[:Nt] = 64
                ang[:Ncycles] = 5 # number of cycles
                ang[:Uinf] = 1.0
                ang[:f] = reduced_freq * ang[:Uinf]
                ang[:kine] = :make_ang
                a0 = 0.1 #St * ang[:Uinf] / ang[:f]
                ang[:motion_parameters] = [a0]

                # Initialize Foil and Flow objects
                foil, fp = init_params(; ang...)
                wake = Wake(foil)
                (foil)(fp)

                # Perform simulations and save results
                old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
                phi = zeros(foil.N)
                input_data = DataFrame(St = Float64[], reduced_freq = Float64[], wave_number = Float64[], σ = Matrix{Float64}, panel_velocity = Matrix{Float64})
                output_data = DataFrame(mu = Float64[], pressure = Float64[])

                for i in 1:fp.Ncycles * foil.N
                    time_increment!(fp, foil, wake)
                    phi = get_phi(foil, wake)
                    p = panel_pressure(foil, fp, old_mus, old_phis, phi)
                    old_mus = [foil.μs'; old_mus[1:2, :]]
                    old_phis = [phi'; old_phis[1:2, :]]

                    if i == 1
                        input_data = DataFrame(St = [St], reduced_freq = [reduced_freq], wave_number = [wave_number], σ = [foil.σs'], panel_velocity = [foil.panel_vel'])
                        output_data = DataFrame(mu = [foil.μs'], pressure = [p'])
                    else
                        append!(input_data, DataFrame(St = [St], reduced_freq = [reduced_freq], wave_number = [wave_number], σ = [foil.σs'], panel_velocity = [foil.panel_vel']))
                        append!(output_data, DataFrame(mu = [foil.μs'], pressure = [p']))
                    end
                end

                push!(allin, input_data)
                push!(allout, output_data)
            end
        end
    end

    input_data = vcat(allin...)
    output_data = vcat(allout...)

    save_data(input_data, output_data, output_dir)
end




function init_flow(Nt)
    # Initialize the flow object with the given parameters
    return flow
end



function create_plot(output_dir)
    # Load output data
    output_file = joinpath(output_dir, "output_data.csv")
    output_data = CSV.read(output_file, DataFrame)

    # Extract relevant columns
    μ_data = output_data.mu
    pressure_data = output_data.pressure

    # Create plot
    t = collect(1:size(μ_data, 2))
    plot(t, μ_data, label = "μ", xlabel = "Time", ylabel = "μ Value", lw = 2)
    plot!(t, pressure_data, label = "Pressure", lw = 2)

    # Save plot
    plot_file = joinpath(output_dir, "simulation_plot.png")
    savefig(plot_file)
end





output_dir = "path/to/output/directory"
run_simulations(output_dir)
input_data_file = joinpath(output_dir, "input_data.csv")
output_data_file = joinpath(output_dir, "output_data.csv")
input_data = CSV.read(input_data_file, DataFrame)
output_data = CSV.read(output_data_file, DataFrame)

#average_per_cycle(output_dir, foil)







using Surrogates
using Flux
using Statistics
using SurrogatesFlux

#bounds are what we can sample from
bounds = (St = (0.1, 0.3), reduced_freq = (0.1, 0.3), wave_number = (0.1, 0.3))
#the sample will pull out a St,f,k tuple
n_samples = 100
rng = Random.default_rng()  # Create a random number generator
x_s = [Tuple(x) for x in Iterators.partition(x_s, length(bounds))]  # Reshape x_s into a list of tuples
# dataFrames
x_train = [x[1] for x in x_s]  # Extract the first element (σs) from each tuple
y_train = [x[2] for x in x_s]  # Extract the second element (μs) from each tuple
# :N <- number of panels for our swimmer
N_panels = 64 #ang[:N]
# Perceptron with one hidden layer of 20 neurons.
N = 10
using Flux

# Define the model architecture
model = Chain(
    Dense(1, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 8),
    Dense(8, 1)
)

# Define the loss function
loss(x, y) = Flux.mse(model(x), y)

# Define the optimizer
optimizer = Descent(0.1)

# Define the number of epochs
n_epochs = 50

# Reshape the input data
x_train = reshape(x_train, (size(x_train, 1), 1))
y_train = reshape(y_train, (size(y_train, 1), 1))

# Train the model
for epoch in 1:n_epochs
    Flux.train!(loss, Flux.params(model), [(x_train, y_train)], optimizer)
end

# Testing the new model
x_test = [sample(1, bounds..., SobolSample())[1] for _ in 1:length(x_train)]
x_test_reshaped = reshape(x_test, (size(x_test, 1), 1))
test_error = mean(abs2, Flux.predict(model, x_test_reshaped) .- f(x_test))
