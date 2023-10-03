using Puffer
using Plots, CSV, DataFrames
using Surrogates
using Flux
using Statistics
using SurrogatesFlux
using Random
using DataDrivenDiffEq
using SymbolicRegression
using DataDrivenSR


ang = deepcopy(defaultDict)
ang[:N]       = 64      #number of panels (elements) to discretize the foil
ang[:Nt]      = 64      #number of timesteps per period of motion
ang[:Ncycles] = 3       #number of periods to simulate
ang[:f]       = 1     #frequency of wave motion 
ang[:Uinf]    = 1     #free stream velocity 
ang[:kine]    = :make_ang  
ang[:T] = Float32 #kinematics generation function
a0 = 0.1                #how much the leading edge heaves up and down wrt the chord(length) of the swimmer
ang[:motion_parameters] = a0


begin
"""
This code snippet implements the above dictionary, ang, and runs a simulation based on the parameters
The main loop of the simulation can be seen in the for loop, though explicit matrix building is done in time_increment!

The code produces plots of the performance metrics for the given setup. 
"""
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
    """This makes a gif of the same swimmer"""
    foil, flow = init_params(;ang...)   
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)       
        win=nothing
        f = plot_current(foil, wake;window=win)
        plot!(f, ylims=(-1,1))
        
    end
    gif(movie, "handp.gif", fps=10)
end



"""
    save_data(input_data, output_data, output_dir)

Not sure who wrote this, probably chatGPT. Looks pretty useless to me
"""
function save_data(input_data, output_data, output_dir)
    # Save input data
    input_file = joinpath(output_dir, "input_data.csv")
    CSV.write(input_file, input_data)

    # Save output data
    output_file = joinpath(output_dir, "output_data.csv")
    CSV.write(output_file, output_data)
end





begin   
    """
    Primitive approach to scraping data needed to train neural networks 
    """
    # Define parameter ranges
    Strouhal_values = 0.1:0.1:0.3
    reduced_freq_values = 0.1:0.1:0.3
    wave_number_values = 0.1:0.1:0.3

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
                a0 = St * ang[:Uinf] / ang[:f]
                ang[:motion_parameters] = [a0]
                T = ang[:T]
                # Initialize Foil and Flow objects
                foil, fp = init_params(; ang...)
                wake = Wake(foil)
                (foil)(fp)

                # Perform simulations and save results
                old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
                phi = zeros(foil.N)
                # Lets grab this data to start
                # [foil.col'  foil.normals' foil.tangents' foil.wake_ind_vel' foil.panel_vel' U_inf]
                input_data = DataFrame(St = T[],       reduced_freq   = T[],       wave_number = T[], U_inf = T[], 
                                        σ = Matrix{T}, panel_velocity = Matrix{T}, position =  Matrix{T},
                                        normals =  Matrix{T}, wake_vel = Matrix{T}, tangents =  Matrix{T})
                output_data = DataFrame(μ = T[], pressure = T[])
                
                for i in 1:fp.Ncycles * foil.N
                    time_increment!(fp, foil, wake)
                    phi = get_phi(foil, wake)
                    p = panel_pressure(foil, fp, old_mus, old_phis, phi)
                    old_mus = [foil.μs'; old_mus[1:2, :]]
                    old_phis = [phi'; old_phis[1:2, :]]

                    if i == 1
                        input_data = DataFrame(St = [St], reduced_freq = [reduced_freq], wave_number = [wave_number], U_inf=[flow.Uinf],
                                                σ = [foil.σs'], panel_velocity = [foil.panel_vel'], position=[foil.col'],
                                                normals = [foil.normals'], wake_vel = [foil.wake_ind_vel'], tangents=[foil.tangents'])
                        output_data = DataFrame(μ = [foil.μs'], pressure = [p'])
                    else
                        append!(input_data, DataFrame(St = [St], reduced_freq = [reduced_freq], wave_number = [wave_number], U_inf=[flow.Uinf],
                                                       σ = [foil.σs'], panel_velocity = [foil.panel_vel'], position=[foil.col'],
                                                       normals = [foil.normals'], wake_vel = [foil.wake_ind_vel'], tangents=[foil.tangents']))
                        append!(output_data, DataFrame(μ = [foil.μs'], pressure = [p']))
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





###############EVERYTHING BELOW IS STRICTLY SCRATCH WORK###############################

### LOAD DATA AND MESS AROUND
output_dir = "./data/"
# run_simulations(output_dir)
input_data_file = joinpath(output_dir, "input_data.csv")
output_data_file = joinpath(output_dir, "output_data.csv")
input_data = CSV.read(input_data_file, DataFrame)
output_data = CSV.read(output_data_file, DataFrame)


# bounds are what we can sample from
bounds = (St = (0.1, 0.3), reduced_freq = (0.1, 0.3), wave_number = (0.1, 0.3))
# the sample will pull out a St,f,k tuple
n_samples = 100
rng = Random.default_rng(38)  # Create a random number generator
x_s = [Tuple(x) for x in Iterators.partition(x_s, length(bounds))]  # Reshape x_s into a list of tuples
# dataFrames
x_train = [x[1] for x in x_s]  # Extract the first element (σs) from each tuple
y_train = [x[2] for x in x_s]  # Extract the second element (μs) from each tuple
x_train = input_data.σ
y_train = output_data.mu
# :N <- number of panels for our swimmer
NP = ang[:N]

# Define the model architecture
# TODO: write factories for building these out faster
model = Chain(
        enc = Chain(Dense(ang[:N], NP÷2, Flux.relu),
           Dense(NP÷2, NP÷4, Flux.relu),
           Dense(NP÷4, NP÷8, Flux.relu)),
        latent = Chain(Dense(8,8)),
        dec = Chain(Dense(NP÷8, NP÷4, Flux.relu),
           Dense(NP÷4, NP÷2, Flux.relu),
           Dense(NP÷2, ang[:N], Flux.relu)),
        )
model(foil.σs) #does this kick out numbers?
# """ We have implemented named tuples for our model. This means we can easily 
#     process portions of our model. This will be useful when writing new loss functions.
# """
@assert model(foil.σs) == model.layers.dec(model.layers.latent(model.layers.enc(foil.σs)))
# You may often see people write these models in a more function approach, they are equivalent
@assert model(foil.σs) == foil.σs |> model.layers.enc |> model.layers.latent |> model.layers.dec


##### CONVOLUTIONAL LAYERS

# Prepare and shape the data for input/output
σs = nothing
inputs = nothing
for row in eachrow(input_data)
    U_inf = repeat([-row.U_inf, 0.0]', foil.N)
    # DataFrame(St, reduced_freq, wave_number, U_inf,
    # σ , panel_velocity , position,
    # normals, wake_vel , tangents)
    x_in = [row.position  row.normals row.wake_vel row.panel_velocity U_inf]
    x_in = reshape(x_in, NP, 2, 5, 1)
    σs = isnothing(σs) ? row.σ : vcat(σs,row.σ)
    inputs = isnothing(inputs) ? x_in : cat(inputs, x_in; dims=4)
end



#WHCN -> width height channels number
@show inputs |> size  
convL = Conv((1,1), 6 => 6)

# Flux.@autosize size(x_in)
convL(x_in)

# Define the loss function
loss(x, y) = Flux.mse(model(x), y)

# Define the optimizer
optimizer = Descent(0.01)

# Define the number of epochs
n_epochs = 50

# Reshape the input data
x_train = reshape(x_train, (size(x_train, 1), 1))
y_train = reshape(y_train, (size(y_train, 1), 1))

##NATE##
lower_bound = [bound[1] for bound in bounds]
upper_bound = [bound[2] for bound in bounds]
neural = NeuralSurrogate(x_train, y_train, lower_bound, upper_bound, model = model, n_echos = 10)
surrogate_optimize(schaffer, SRBF(), lower_bound, upper_bound, neural, SobolSample(), maxiters=20, num_new_samples=10)
##NATE##

# Train the model
for epoch in 1:n_epochs
    Flux.train!(loss, Flux.params(model), [(x_train, y_train)], optimizer)
end

# Testing the new model
x_test = [sample(1, bounds..., SobolSample())[1] for _ in 1:length(x_train)]
x_test_reshaped = reshape(x_test, (size(x_test, 1), 1))
test_error = mean(abs2, Flux.predict(model, x_test_reshaped) .- f(x_test))
sample(10, lower_bound, upper_bound, SobolSample())

input_data[St=0.1,:]


#fix k, St, and f
# then sample on that data, train, test. 
#after that, roll back to allow for mutation on k, St, f

id = input_data
σs =  id[(id.St .== 0.1) .& (id.reduced_freq .== 0.1) .& (id.wave_number .== 0.1), :σ]
#output isn't labeled?
μs =  output_data[(id.St .== 0.1) .& (id.reduced_freq .== 0.1) .& (id.wave_number .== 0.1), :mu]
model.(σs')

bounds = (1, 320)
#inline updates the model, should be a !
neural = NeuralSurrogate(σs, μs, bounds[1], bounds[2], model = model, n_echos = 10)
#work on how to make a proxy function for input to output 
getμ(x) = μs[x]
surrogate_optimize(getμ, SRBF(), bounds[1],  bounds[2], neural, SobolSample(), maxiters=20, num_new_samples=10)


using DataDrivenDMD