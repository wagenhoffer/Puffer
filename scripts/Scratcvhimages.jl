begin
    counter = 1
    for reduced_freq in reduced_freq_values
        for k in k_values
            for δ in δs
                @show counter, reduced_freq, k, δ
                counter += 1
            end
        end
    end
end
# Set motion parameters
# (counter, reduced_freq, k, δ) = (75, 2.125f0, 2.0f0, 0.65f0)
(counter, reduced_freq, k, δ) = (116, 4.0f0, 1.5875f0, 0.15f0)
# (counter, reduced_freq, k, δ) = (76, 3.0625f0, 0.35f0, 0.15f0)
starting_positions = [0.0  0.0 ; -δ δ ]
phases = [pi/2, -pi/2,pi/2, pi/2]    
fs     = [ reduced_freq, -reduced_freq]
ks     = [ k, -k]

motion_parameters = [a0 for i in 1:num_foils]

foils, flow = create_foils(num_foils, starting_positions, :make_wave;
        motion_parameters=motion_parameters, ψ=phases, Ncycles = 5,
        k= ks,  Nt = 100, f = fs);

wake = Wake(foils)


# Perform simulations and save results
totalN = sum(foil.N for foil in foils)
steps = flow.N*flow.Ncycles

old_mus, old_phis = zeros(3, totalN), zeros(3, totalN)
coeffs = zeros(length(foils), 4, steps)
μs = zeros(totalN)
phis = zeros(totalN)
ps = zeros(totalN)
    

movie = @animate for i in 1:steps
    rhs = time_increment!(flow, foils, wake)


    
    for (j, foil) in enumerate(foils)        
        phi = get_phi(foil, wake)
        phis[((j - 1) * foil.N + 1):(j * foil.N)] = phi
        p = panel_pressure(foil,
            flow,
            old_mus[:, ((j - 1) * foil.N + 1):(j * foil.N)],
            old_phis[:,((j - 1) * foil.N + 1):(j * foil.N)],
            phi)
        ps[((j - 1) * foil.N + 1):(j * foil.N)] = p
        μs[((j - 1) * foil.N + 1):(j * foil.N)] = foil.μs
        coeffs[j, :, i ] .= get_performance(foil, flow, p)
    end
    # old_mus = [μs'; old_mus[1:2, :]]
    # old_phis = [phis'; old_phis[1:2, :]]
    
    if i>5
        plot(foils, wake)
    else
        plot(foils)
    end
end
gif(movie, fps = 30)