
# this is  modifyed code after getting a results from emad.jl( still have some errors)
# modifyed function and pushing it to surrogates
# its not completed yet




function run_simulations(output_dir)
    # Define parameter ranges
    Strouhal_values = [0.1, 0.2, 0.3]  
    reduced_freq_values = [0.1, 0.2, 0.3]  
    wave_number_values = [0.1, 0.2, 0.3]  

    input_data = DataFrame(St = Float64[], reduced_freq = Float64[], wave_number = Float64[], σ = Float64[], panel_velocity = Float64[])
    output_data = DataFrame(mu = Float64[], pressure = Float64[])

    # Nested loops to vary parameters
    for St in Strouhal_values
        for reduced_freq in reduced_freq_values
            for wave_number in wave_number_values
                # Set motion parameters
                ang = deepcopy(defaultDict)
                ang[:N] = 64
                ang[:Nt] = 64
                ang[:Ncycles] = 5  # Number of cycles to simulate
                ang[:f] = reduced_freq * flow.Uinf / foil.chord  # Calculate frequency based on reduced frequency
                ang[:Uinf] = 1.0  # Update freestream velocity if necessary
                ang[:kine] = :make_ang 
                a0 = 0.1  # Amplitude of motion
                ang[:motion_parameters] = a0

                # Initialize foil, flow, and wake
                foil, flow = init_params(;ang...)
                wake = Wake(foil)
                (foil)(flow)

                # Data containers
                old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
                phi = zeros(foil.N)
                coeffs = zeros(4, flow.Ncycles * flow.N)
                ps = zeros(foil.N, flow.Ncycles * flow.N)
                # 'ps' is a matrix that stores the pressure values for each panel at each time step.

                # Performance metrics loop
                for i in 1:flow.Ncycles * flow.N #iterates over each time step.
                    time_increment!(flow, foil, wake)#update the flow, foil, and wake parameters for the current time step.
                    phi = get_phi(foil, wake)# Calculation of φ (output space).
                    p = panel_pressure(foil, flow, old_mus, old_phis, phi)
                    old_mus = [foil.μs'; old_mus[1:2, :]]
                    old_phis = [phi'; old_phis[1:2, :]]
                    coeffs[:, i] = get_performance(foil, flow, p)
                    ps[:, i] = p
                end

                # Phase averaging
                phase_avg_p = phase_average(ps, flow.Ncycles, flow.N)

                # Append input and output data
                push!(input_data, (St = St, reduced_freq = reduced_freq, wave_number = wave_number,
                    σ = flow.σ, panel_velocity = foil.μ_edge[1]))
                push!(output_data, (mu = foil.μs[1], pressure = phase_avg_p))

            end
        end
    end

    # Save input and output data
    save_data(input_data, output_data, output_dir)
end
