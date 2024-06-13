using Puffer
using Plots
using DataFrames
using Serialization

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

make_car(a0) = make_wave(a0; a=[0.2,-0.825, 1.625])
make_ang(a0) = make_wave(a0)
begin
    """
    Primitive approach to scraping data needed to train neural networks 
    """
    # Define parameter ranges
    T = Float32
    reduced_freq_values = LinRange{T}(0.25, 4, 10)
    ks = LinRange{T}(0.35, 2.0, 10)

    allofit = Vector{DataFrame}()
    allCoeffs = Vector{DataFrame}()
    # Nested loops to vary parameters
    waves = [:make_ang, :make_car]    
    for wave in waves
        for reduced_freq in reduced_freq_values
            for k in ks
                # Set motion parameters
                ang = deepcopy(defaultDict)
                ang[:N] = 64
                ang[:Nt] = 100
                ang[:Ncycles] = 7 # number of cycles
                ang[:Uinf] = 1.0
                ang[:f] = reduced_freq * ang[:Uinf]
                ang[:kine] = wave
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
                    df = DataFrame(reduced_freq = [reduced_freq],
                                            k = [k],
                                            U_inf = [flow.Uinf],
                                            t = [flow.n * flow.Δt],
                                            σs = [deepcopy(foil.σs)],
                                            panel_vel = [deepcopy(foil.panel_vel)],
                                            position = [deepcopy(foil.col)],
                                            normals = [deepcopy(foil.normals)],
                                            wake_ind_vel = [deepcopy(foil.wake_ind_vel)],
                                            tangents = [deepcopy(foil.tangents)],
                                            μs = [deepcopy(foil.μs)],
                                            pressure = [deepcopy(p)], 
                                            RHS = [deepcopy(rhs)])

                    if i == ang[:Nt] * 2 + 1#skip first 2 cycles
                        @show i
                        datas = df
                    elseif i > ang[:Nt] * 2 + 1                        
                        append!(datas,df)
                    end
                end

                push!(allofit, datas)
                push!(allCoeffs, DataFrame(wave = [wave], reduced_freq = [reduced_freq], k = [k], coeffs = [coeffs[:,ang[:Nt]*2+1:end]]))
            end
        end  
    end  
    # path = joinpath("data", "starter_data.jls")
    path = joinpath("data", "single_swimmer_ks_$(ks[1])_$(ks[end])_fs_$(reduced_freq_values[1])_$(reduced_freq_values[end])_ang_car.jls")    
    serialize(path, vcat(allofit...))
    path = joinpath("data", "single_swimmer_coeffs_ks_$(ks[1])_$(ks[end])_fs_$(reduced_freq_values[1])_$(reduced_freq_values[end])_ang_car.jls")
    
    serialize(path, vcat(allCoeffs...))
end

begin
    """
    Redo the above for multiple image swimmers
    """
    # Define parameter ranges
    T = Float32
    reduced_freq_values = LinRange{T}(0.25, 4, 5)
    k_values = LinRange{T}(0.35, 2.0, 5)
    a0 = 0.1
    δs = LinRange{T}(3*a0, 13*a0, 5)./T(2.0)

    num_foils = 2

    allofit = Vector{DataFrame}()
    allCoeffs = Vector{DataFrame}()
    datas = Vector{DataFrame}()
    # Nested loops to vary parameters
    counter = 1
    for reduced_freq in reduced_freq_values
        for k in k_values
            for δ in δs
                @show counter, reduced_freq, k, δ
                counter +=1
                # Set motion parameters
                starting_positions = [0.0  0.0 ; -δ δ ]
                phases = [pi/2, pi/2,pi/2, pi/2]    
                fs     = [ reduced_freq, -reduced_freq]
                ks     = [ -k, k]
                
                motion_parameters = [a0 for i in 1:num_foils]
            
                foils, flow = create_foils(num_foils, starting_positions, :make_wave;
                        motion_parameters=motion_parameters, ψ=phases, Ncycles = 5,
                        k= ks,  Nt = 64, f = fs);
                
                wake = Wake(foils)
                

                # Perform simulations and save results
                totalN = sum(foil.N for foil in foils)
                steps = flow.N*flow.Ncycles
                
                old_mus, old_phis = zeros(3, totalN), zeros(3, totalN)
                coeffs = zeros(length(foils), 4, steps)
                μs = zeros(totalN)
                phis = zeros(totalN)
                ps = zeros(totalN)
                    
                
                @time for i in 1:steps
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
                    old_mus = [μs'; old_mus[1:2, :]]
                    old_phis = [phis'; old_phis[1:2, :]]
                    
                    
                    vals = DataFrame( δ = [δ],
                                        reduced_freq = [reduced_freq],
                                        k            = [k],
                                        U_inf        = [flow.Uinf],
                                        t            = [flow.n * flow.Δt],
                                        σs           = [vcat([foil.σs for foil in foils]...)],
                                        panel_vel    = [vcat([foil.panel_vel for foil in foils]...)],
                                        position     = [vcat([foil.col for foil in foils]...)],
                                        normals      = [vcat([foil.normals for foil in foils]...)],
                                        wake_ind_vel = [vcat([foil.wake_ind_vel for foil in foils]...)],
                                        tangents     = [vcat([foil.tangents for foil in foils]...)],
                                        μs           = [vcat([foil.μs for foil in foils]...)],
                                        pressure     = [ps],
                                        RHS          = [rhs]                        
                                        )        
                    if i == flow.N
                        datas = vals
                    elseif i > flow.N
                        append!(datas, vals)
                    end

                    # plot(foils, wake)
                end
                # file = "d_$(@sprintf("%.2f", δ))_f_$(reduced_freq)_k_$(k).gif"
                # path = joinpath("images","gfx_images", file)
                # gif(movie, path, fps = 30)
                coeff_df = DataFrame(δ = [δ], reduced_freq = [reduced_freq], k = [k], coeffs = [coeffs[:,:,flow.N:end]])
                                
                push!(allCoeffs, coeff_df)
                push!(allofit, datas)
            end
        end
    end    
    path = joinpath("data", "multipleSwimmers_images_data.jls")    
    allofit = vcat(allofit...)
    serialize(path, allofit)
    path = joinpath("data", "multipleSwimmers_images_coeffs.jls")
    allCoeffs = vcat(allCoeffs...)
    serialize(path, allCoeffs)
end



begin
    """
    Redo the above for multiple inline swimmers
    """
    # Define parameter ranges
    T = Float32
    # reduced_freq_values = LinRange{T}(0.25, 4, 10)
    reduced_freq_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, π,  4.0] .|>T
    k_values = LinRange{T}(0.35, 2.0, 5)
    a0 = 0.1
    δs = LinRange{T}(1.25 ,2.0, 4)
    ψs = LinRange{T}(0, pi, 5)

    num_foils = 2


    allofit = Vector{DataFrame}()
    allCoeffs = Vector{DataFrame}()
    datas = Vector{DataFrame}()
    # Nested loops to vary parameters
    counter = 1
    for reduced_freq in reduced_freq_values
        for k in k_values
            for δ in δs
                for ψi in ψs
                    # @show counter, reduced_freq, k, δ , ψi
                    counter +=1
                    # Set motion parameters
                    starting_positions = [0.0  δ; 0.0 0.0 ]
                    phases = [0, ψi]    
                    fs     = [ reduced_freq, reduced_freq]
                    ks     = [ k, k]
                    
                    motion_parameters = [a0 for i in 1:num_foils]
                
                    foils, flow = create_foils(num_foils, starting_positions, :make_wave;
                            motion_parameters=motion_parameters, ψ=phases, Ncycles = 6,
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
                        

                    @time for i in 1:steps
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
                        old_mus = [μs'; old_mus[1:2, :]]
                        old_phis =     k_values = LinRange{T}(0.35, 2.0, 5)[phis'; old_phis[1:2, :]]
                        
                        
                        vals = DataFrame( δ = [δ],
                                            reduced_freq = [reduced_freq],
                                            k            = [k],
                                            U_inf        = [flow.Uinf],
                                            t            = [flow.n * flow.Δt],
                                            σs           = [vcat([foil.σs for foil in foils]...)],
                                            panel_vel    = [vcat([foil.panel_vel for foil in foils]...)],
                                            position     = [vcat([foil.col for foil in foils]...)],
                                            normals      = [vcat([foil.normals for foil in foils]...)],
                                            wake_ind_vel = [vcat([foil.wake_ind_vel for foil in foils]...)],
                                            tangents     = [vcat([foil.tangents for foil in foils]...)],
                                            μs           = [vcat([foil.μs for foil in foils]...)],
                                            pressure     = [ps],
                                            RHS          = [rhs]                        
                                            )        
                        if i == flow.N
                            datas = vals
                        elseif i > flow.N
                            append!(datas, vals)
                        end

                        # plot(foils, wake)
                    end
                    # file = "d_$(@sprintf("%.2f", δ))_f_$(reduced_freq)_k_$(k).gif"
                    # path = joinpath("images","gfx_images", file)
                    # gif(movie, path, fps = 30)
                    coeff_df = DataFrame(δ = [δ], reduced_freq = [reduced_freq], k = [k], coeffs = [coeffs[:,:,flow.N:end]])
                        
                    push!(allCoeffs, coeff_df)
                    push!(allofit, datas)
                    
                end
            end
        end
        
    end    
    path = joinpath("data", "multipleSwimmers_inline_data.jls")    
    allofit = vcat(allofit...)
    serialize(path, allofit)
    path = joinpath("data", "multipleSwimmers_inline_coeffs.jls")
    allCoeffs = vcat(allCoeffs...)
    serialize(path, allCoeffs)
end

begin
    # Define parameter ranges
    T = Float32
    reduced_freq_values = [0.5, 1.0, 2.0] .|>T
    k_values = [0.5, 1.0, 2.0] .|>T
    a0 = 0.1
    δxs = LinRange{T}(1.25, 2.0, 4)
    δys = LinRange{T}(0.5,  2.0, 4)
    ψs = LinRange{T}(0, pi/2, 3)
  
    reduced_freq = rand(reduced_freq_values)
    k = rand(k_values)
    δx = δxs[1]
    δy = δys[1]
    ψi = rand(ψs)

    num_foils = 4

    allofit = Vector{DataFrame}()
    allCoeffs = Vector{DataFrame}()
    datas = Vector{DataFrame}()
    # Nested loops to vary parameters
    counter = 1
    for reduced_freq in reduced_freq_values
        for k in k_values
            for δx in δxs
                for δy in δys
                    for ψi in ψs
                        counter +=1
                        # Set motion parameters
                        starting_positions = [0.0  δy; δx 0.0; 0.0 -δy; -δx 0.0]'
                        phases = [0, ψi, 0, ψi].|>mod2pi
                        fs = [reduced_freq for _ in 1:num_foils]
                        ks = [k for _ in 1:num_foils]
                        motion_parameters = [a0 for _ in 1:num_foils]

                        foils, flow = create_foils(num_foils, starting_positions, :make_wave;
                            motion_parameters=motion_parameters, ψ=phases, Ncycles = 8,
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
                                
        
                            @time for i in 1:steps
                            # (foils)(flow)
                            # movie = @animate for i in 1:steps
                            #     rhs = time_increment!(flow, foils, wake)
                            #     plot(foils, wake)
                            # end
                            # gif(movie, "test.gif", fps = 30)
        
                                
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
                                old_mus = [μs'; old_mus[1:2, :]]
                                old_phis = [phis'; old_phis[1:2, :]]
                                
                                
                                vals = DataFrame(   δx = [δx],
                                                    δy = [δy],
                                                    reduced_freq = [reduced_freq],
                                                    k            = [k],
                                                    ψi = [ψi],
                                                    U_inf        = [flow.Uinf],
                                                    t            = [flow.n * flow.Δt],
                                                    σs           = [vcat([foil.σs for foil in foils]...)],
                                                    panel_vel    = [vcat([foil.panel_vel for foil in foils]...)],
                                                    position     = [vcat([foil.col for foil in foils]...)],
                                                    normals      = [vcat([foil.normals for foil in foils]...)],
                                                    wake_ind_vel = [vcat([foil.wake_ind_vel for foil in foils]...)],
                                                    tangents     = [vcat([foil.tangents for foil in foils]...)],
                                                    μs           = [vcat([foil.μs for foil in foils]...)],
                                                    pressure     = [ps],
                                                    RHS          = [rhs]                        
                                                    )        
                                if i == flow.N
                                    datas = vals
                                elseif i > flow.N
                                    append!(datas, vals)
                                end
        
                                # plot(foils, wake)
                            end
                            # file = "d_$(@sprintf("%.2f", δ))_f_$(reduced_freq)_k_$(k).gif"
                            # path = joinpath("images","gfx_images", file)
                            # gif(movie, path, fps = 30)
                            # coeff_df = DataFrame(δ = [δ], reduced_freq = [reduced_freq], k = [k], coeffs = [coeffs[:,:,flow.N:end]])
                            coeff_df = DataFrame(δx = [δx], δy = [δy], reduced_freq = [reduced_freq], k = [k], ψi=[ψi], coeffs = [coeffs[:,:,flow.N:end]])
                                
                            push!(allCoeffs, coeff_df)
                            push!(allofit, datas)
                    end
                end
            end
        end
    end
    path = joinpath("data", "diamondSwimmers_13_42_phasediff_data.jls")    
    allofit = vcat(allofit...)
    serialize(path, allofit)
    path = joinpath("data", "diamondSwimmers_13_42_phasediff_coeffs.jls")
    allCoeffs = vcat(allCoeffs...)
    serialize(path, allCoeffs)
end