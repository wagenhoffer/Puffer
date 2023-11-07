using Puffer
using Plots, DataFrames, Serialization

ang = deepcopy(defaultDict)
ang[:N] = 64      #number of panels (elements) to discretize the foil
ang[:Nt] = 64      #number of timesteps per period of motion
ang[:Ncycles] = 3       #number of periods to simulate
ang[:f] = 1     #frequency of wave motion 
ang[:Uinf] = 1     #free stream velocity 
ang[:kine] = :make_ang
ang[:T] = Float32 #kinematics generation function
a0 = 0.1                #how much the leading edge heaves up and down wrt the chord(length) of the swimmer
ang[:motion_parameters] = a0

begin
    """
    This code snippet implements the above dictionary, ang, and runs a simulation based on the parameters
    The main loop of the simulation can be seen in the for loop, though explicit matrix building is done in time_increment!

    The code produces plots of the performance metrics for the given setup. 

    THIS IS JUST TO VISUALIZE AND ENSURE YOU'RE BUILDING THE CORRECT SYTEMS!
    """
    foil, flow = init_params(; ang...)
    k = foil.f * foil.chord / flow.Uinf
    @show k
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        phi = get_phi(foil, wake)    # (output space) <-probably not that important                            
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)
        # (output space) <- p is a function of μ and we should be able to recreate this     
        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
        coeffs[:, i] = get_performance(foil, flow, p)
        # the coefficients of PERFROMANCE are important, but are just a scaling of P
        # if we can recreate p correctly, this will be easy to get also (not important at first)
        ps[:, i] = p # storage container of the output, nice!
    end
    t = range(0, stop = flow.Ncycles * flow.N * flow.Δt, length = flow.Ncycles * flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs[1, start:end], label = "Force", lw = 3, marker = :circle)
    b = plot(t[start:end], coeffs[2, start:end], label = "Lift", lw = 3, marker = :circle)
    c = plot(t[start:end], coeffs[3, start:end], label = "Thrust", lw = 3, marker = :circle)
    d = plot(t[start:end], coeffs[4, start:end], label = "Power", lw = 3, marker = :circle)
    plot(a, b, c, d, layout = (2, 2), legend = :topleft, size = (800, 800))
end

begin
    """
    Primitive approach to scraping data needed to train neural networks 
    """
    # Define parameter ranges
    T = Float32
    reduced_freq_values = LinRange{T}(0.25, 4, 5)
    ks = LinRange{T}(0.35, 2.0, 5)

    allofit = Vector{DataFrame}()
    # Nested loops to vary parameters

    for reduced_freq in reduced_freq_values
        for k in ks
            # Set motion parameters
            ang = deepcopy(defaultDict)
            ang[:N] = 64
            ang[:Nt] = 64
            ang[:Ncycles] = 5 # number of cycles
            ang[:Uinf] = 1.0
            ang[:f] = reduced_freq * ang[:Uinf]
            ang[:kine] = :make_ang
            ang[:T] = T
            ang[:k] = k
            a0 = 0.1 
            @show reduced_freq*a0
            ang[:motion_parameters] = [a0]
            T = ang[:T]
            # Initialize Foil and Flow objects
            foil, flow = init_params(; ang...)
            wake = Wake(foil)
            (foil)(flow)

            # Perform simulations and save results
            old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
            phi = zeros(foil.N)
            # Lets grab this data to start
            # [foil.col'  foil.normals' foil.tangents' foil.wake_ind_vel' foil.panel_vel' U_inf]
            datas = DataFrame(St = T[],
                reduced_freq = T[],
                k = T[],
                U_inf = T[],
                t = T[],
                σs = Matrix{T},
                panel_velocity = Matrix{T},
                position = Matrix{T},
                normals = Matrix{T},
                wake_ind_vel = Matrix{T},
                tangents = Matrix{T},
                μs = T[],
                pressure = T[])

            for i in 1:(flow.Ncycles * foil.N)
                time_increment!(flow, foil, wake)
                phi = get_phi(foil, wake)
                p = panel_pressure(foil, flow, old_mus, old_phis, phi)
                old_mus = [foil.μs'; old_mus[1:2, :]]
                old_phis = [phi'; old_phis[1:2, :]]

                if i == 1
                    datas = DataFrame(reduced_freq = [reduced_freq],
                        k = [k],
                        U_inf = [flow.Uinf],
                        t = [flow.n * flow.Δt],
                        σs = [foil.σs],
                        panel_vel = [foil.panel_vel],
                        position = [foil.col],
                        normals = [foil.normals],
                        wake_ind_vel = [foil.wake_ind_vel],
                        tangents = [foil.tangents],
                        μs = [foil.μs],
                        pressure = [p])

                else
                    append!(datas,
                        DataFrame(reduced_freq = [reduced_freq],
                            k = [k],
                            U_inf = [flow.Uinf],
                            t = [flow.n * flow.Δt],
                            σs = [foil.σs],
                            panel_vel = [foil.panel_vel],
                            position = [foil.col],
                            normals = [foil.normals],
                            wake_ind_vel = [foil.wake_ind_vel],
                            tangents = [foil.tangents],
                            μs = [foil.μs],
                            pressure = [p]))
                end
            end

            push!(allofit, datas)
        end
    end    
    path = joinpath("data", "starter_data.jls")

    allofit = vcat(allofit...)
    serialize(path, allofit)
end

column_data_types = Dict{Symbol, DataType}()
types = []
for col in names(allofit) .|> Symbol
    col_data_type = eltype(allofit[!, col])
    column_data_types[col] = col_data_type
    push!(types, col_data_type)
end
