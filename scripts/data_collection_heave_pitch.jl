using Puffer
using Plots
using DataFrames
using Serialization

begin
    """
    Primitive approach to scraping data needed to train neural networks 
    """
    # Define parameter ranges
    T = Float32
    reduced_freq_values = LinRange{T}(0.25, 4, 7)
    θs = LinRange{T}(0, 10, 6)
    hs = LinRange{T}(0.0, 0.25, 6)

    allofit = Vector{DataFrame}()
    allCoeffs = Vector{DataFrame}()
    local datas = Vector{DataFrame}()
    # Nested loops to vary parameters
    
    for reduced_freq in reduced_freq_values
        for θ in θs
            for h in hs    
                @show reduced_freq, θ, h
                # Set motion parameters
                h_p = deepcopy(defaultDict)
                h_p[:N] = 64
                h_p[:Nt] = 100
                h_p[:Ncycles] = 6 # number of cycles
                h_p[:Uinf] = 1.0
                h_p[:f] = reduced_freq * h_p[:Uinf]
                h_p[:kine] = :make_heave_pitch
                h_p[:T] = Float32
                h_p[:motion_parameters] = [h, θ]
                a0 = 0.1                 
                T = h_p[:T]
                # Initialize Foil and Flow objects
                foil, flow = init_params(; h_p...)
                wake = Wake(foil)
                (foil)(flow)

                # Perform simulations and save results
                old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
                coeffs = zeros(4, flow.Ncycles * flow.N)
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
                    pressure = T[],
                    RHS = T[])
                
                for i in 1:flow.Ncycles * flow.N                    
                    rhs = time_increment!(flow, foil, wake)
                    phi = get_phi(foil, wake)
                    p = panel_pressure(foil, flow, old_mus, old_phis, phi)
                    coeffs[:,i] = get_performance(foil, flow, p)
                    old_mus = [foil.μs'; old_mus[1:2, :]]
                    old_phis = [phi'; old_phis[1:2, :]]

                    if i == flow.N #skip first cycle
                        datas = DataFrame(reduced_freq = [reduced_freq],
                            h= [h],
                            θ = [θ],
                            U_inf = [flow.Uinf],
                            t = [flow.n * flow.Δt],
                            σs = [foil.σs],
                            panel_vel = [foil.panel_vel],
                            position = [foil.col],
                            normals = [foil.normals],
                            wake_ind_vel = [foil.wake_ind_vel],
                            tangents = [foil.tangents],
                            μs = [foil.μs],
                            pressure = [p], 
                            RHS = [rhs])

                    elseif i > flow.N
                        append!(datas,
                            DataFrame(reduced_freq = [reduced_freq],
                                h= [h],
                                θ = [θ],
                                U_inf = [flow.Uinf],
                                t = [flow.n * flow.Δt],
                                σs = [foil.σs],
                                panel_vel = [foil.panel_vel],
                                position = [foil.col],
                                normals = [foil.normals],
                                wake_ind_vel = [foil.wake_ind_vel],
                                tangents = [foil.tangents],
                                μs = [foil.μs],
                                pressure = [p],
                                RHS = [rhs]))
                    end
                end
                push!(allofit, datas)
                push!(allCoeffs, DataFrame(educed_freq = [reduced_freq], θ=[θ], h=[h], coeffs = [coeffs[:,h_p[:Nt]:end]]))
            end            
        end
    end  
 
    # path = joinpath("data", "starter_data.jls")
    path = joinpath("data", "single_swimmer_thetas_$(θs[1])_$(θs[end])_h_$(hs[1])_$(hs[end])_fs_$(reduced_freq_values[1])_$(reduced_freq_values[end])_h_p.jls")
    allofit = vcat(allofit...)
    serialize(path, allofit)
    path = joinpath("data", "single_swimmer_coeffs_thetas_$(θs[1])_$(θs[end])_h_$(hs[1])_$(hs[end])_fs_$(reduced_freq_values[1])_$(reduced_freq_values[end])_h_p.jls")
    allCoeffs = vcat(allCoeffs...)
    serialize(path, allCoeffs)
end