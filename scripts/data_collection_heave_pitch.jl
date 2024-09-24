using Puffer
using Plots
using DataFrames
using Serialization
using NLsolve

T = Float32
strouhals = LinRange{T}(0.2, 0.4, 7)
td = LinRange{Int}(0, 10, 6)
θs = deg2rad.(td).|> T    
hs = LinRange{T}(0.0, 0.25, 6)
δs = LinRange{T}(3, 14, 4)./T(2.0)


fna = Dict()
for θ0 in θs
    for h0 in hs
        for St in strouhals
            # Call the solver
            if θ0 == 0 && h0 == 0.0
                fna[(θ0, h0, St)] = (1.0f0, 0.0f0)
            else
                # @show θ0, h0, St
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

begin
    """
    Primitive approach to scraping data needed to train neural networks 
    """
    # Define parameter ranges

    allofit = Vector{DataFrame}()
    allCoeffs = Vector{DataFrame}()
    local datas = Vector{DataFrame}()
    # Nested loops to vary parameters
    
    for strou in strouhals
        for θ in θs
            for h in hs     
                # if  θ != 0 && h != 0.0 
                    f,a = fna[(θ, h, strou)]
                    
                    @show strou, θ, h, f, a 
                    # Set motion parameters
                    h_p = deepcopy(defaultDict)
                    # h_p[:N] = 64
                    h_p[:Nt] = 100
                    h_p[:Ncycles] = 7 # number of cycles
                    h_p[:Uinf] = 1.0
                    h_p[:f] = f
                    h_p[:kine] = :make_heave_pitch
                    h_p[:T] = Float32
                    h_p[:motion_parameters] = [h, θ]                        
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
                        # @show foil.panel_vel
                        phi = get_phi(foil, wake)
                        p = panel_pressure(foil, flow, old_mus, old_phis, phi)
                        
                        coeffs[:,i] = get_performance(foil, flow, p)
                        old_mus = [foil.μs'; old_mus[1:2, :]]
                        old_phis = [phi'; old_phis[1:2, :]]
                        df = DataFrame(reduced_freq = [f],
                                        h= [h],
                                        θ = [θ],
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
                        if i == flow.N*2+1 #skip first cycle
                            datas = df
                        elseif i > flow.N*2+1
                            append!(datas,df)                            
                        end
                        # plot(foil,wake)
                    end
                    push!(allofit, datas)
                    push!(allCoeffs, DataFrame(Strouhal = [strou], θ=[θ], h=[h], f = [f], coeffs = [coeffs[:,h_p[:Nt]*2+1:end]]))
                # end    
            end            
        end
    end  
 
    # path = joinpath("data", "starter_data.jls")
    path = joinpath("data", "a0_single_swimmer_thetas_$(td[1])_$(td[end])_h_$(hs[1])_$(hs[end])_fs_$(strouhals[1])_$(strouhals[end])_h_p.jls")
    allofit = vcat(allofit...)
    serialize(path, allofit)
    path = joinpath("data", "a0_single_swimmer_coeffs_thetas_$(td[1])_$(td[end])_h_$(hs[1])_$(hs[end])_fs_$(strouhals[1])_$(strouhals[end])_h_p.jls")
    allCoeffs = vcat(allCoeffs...)
    serialize(path, allCoeffs)
end

begin
    """
    Redo the above for multiple image swimmers
    """
    # Define parameter ranges
    T = Float32
    num_foils = 2

    allofit = Vector{DataFrame}()
    allCoeffs = Vector{DataFrame}()
    datas = Vector{DataFrame}()
    # Nested loops to vary parameters
    counter = 1
    for strou in strouhals
        for θ in θs[1:4]
            for h in hs[1:3]     
                if  θ != 0 && h != 0.0                     
                    for δ in δs
                        f,a = fna[(θ, h, strou)]
                        @show counter, strou, θ, h, f, a, δ
                        counter +=1
                        # Set motion parameters
                        starting_positions = [0.0  0.0 ; -δ*a δ*a ]
                        phases = [pi/2, -pi/2]    
                        motion_parameters = [h θ ; -h θ]
                        
                        
                    
                        foils, flow = create_foils(num_foils, starting_positions, :make_heave_pitch;
                                motion_parameters=motion_parameters, ψ=phases, Ncycles = 5,
                                 Nt = 100, f = f);
                        
                        wake = Wake(foils)
                        

                        # Perform simulations and save results
                        totalN = sum(foil.N for foil in foils)
                        steps = flow.N*flow.Ncycles
                        start_step = 1
                        
                        old_mus, old_phis = zeros(3, totalN), zeros(3, totalN)
                        coeffs = zeros(length(foils), 4, steps)
                        μs = zeros(totalN)
                        phis = zeros(totalN)
                        ps = zeros(totalN)
                            
                        
                        for i in 1:steps
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
                                                reduced_freq = [f],
                                                θ            = [θ],
                                                h            = [h],
                                                Strouhal     = [strou],
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
                            if i == start_step
                                datas = vals
                            elseif i > start_step
                                append!(datas, vals)
                            end
                            # if i>10
                            #     plot(foils, wake)
                            # else
                            #     plot(foils)
                            # end
                        end
                        # file = "d_$(@sprintf("%.2f", δ))_f_$(reduced_freq)_k_$(k).gif"
                        # path = joinpath("images","gfx_images", file)
                        # gif(movie, path, fps = 30)
                        push!(allofit, datas)
                        push!(allCoeffs, DataFrame(Strouhal = [strou], θ=[θ], h=[h], f = [f], coeffs = [coeffs[:,:,start_step:end]]))
                    end
                end
            end
        end
    end
 
    path = joinpath("data", "hnp_Swimmers_images_data.jls")    
    allofit = vcat(allofit...)
    serialize(path, allofit)
    path = joinpath("data", "hnp_Swimmers_images_coeffs.jls")
    allCoeffs = vcat(allCoeffs...)
    serialize(path, allCoeffs)
end