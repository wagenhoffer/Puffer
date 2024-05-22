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

fna = Dict()
for θ0 in θs
    for h0 in hs
        for St in strouhals
            # Call the solver
            if θ0 == 0 && h0 == 0.0
                continue
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
                if  θ != 0 && h != 0.0 
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
                        plot(foil,wake)
                    end
                    push!(allofit, datas)
                    push!(allCoeffs, DataFrame(Strouhal = [strou], θ=[θ], h=[h], f = [f], coeffs = [coeffs[:,h_p[:Nt]*2+1:end]]))
                end    
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