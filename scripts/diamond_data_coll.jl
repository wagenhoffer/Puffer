using Puffer
using Plots
using DataFrames
using Serialization
using CUDA


make_car(a0) = make_wave(a0; a=[0.2,-0.825, 1.625])
make_ang(a0) = make_wave(a0)
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
                                rhs = time_increment!(flow, foils, wake)
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


@inbounds @views macro dx(s,t) esc(:( ($t[1 ,:] .- $s[1,:]') )) end
@inbounds @views macro dy(s,t) esc(:( ($t[2 ,:] .- $s[2,:]') )) end
function cast_and_pull(sources, targets, Γs)
    @inbounds @views function vt!(vel,S,T,Γs,δ)
        n = size(S,2)
        mat = CUDA.zeros(Float32, n,n)    
        dx = @dx(S,T)
        dy = @dy(S,T)
        @. mat = Γs /(2π * sqrt((dx.^2 .+ dy.^2 )^2 + δ^4))
        @views vel[1,:] = -sum(dy .* mat, dims = 1)
        @views vel[2,:] =  sum(dx .* mat, dims = 1)
        return nothing
    end
    vel = CUDA.zeros(Float32, size(targets)...)
    vt!(vel, sources, targets, Γs, flow.δ)
    vel|>Array    
end

function vortex_to_target(sources::Matrix{T}, targets, Γs, flow) where {T <: Real}
    if CUDA.functional()
        cast_and_pull(vel, sources, targets, Γs)
    else
        vortex_to_target(sources, targets, Γs, flow)
    end
end


@time cast_and_pull(sources, targets, Γs)
@time vortex_to_target(wake, flow)