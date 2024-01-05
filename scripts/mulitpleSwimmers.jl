using Puffer
using Plots
using LinearAlgebra

function create_foils(num_foils, starting_positions, kine; kwargs...) 
    # Ensure starting_positions has correct dimensions
    # if size(starting_positions) != (2, num_foils)
    #     error("starting_positions must be a 2xN array, where N is the number of foils")
    # end
    pos = deepcopy(defaultDict)       
    pos[:kine] = kine
    foils = Vector{Foil{pos[:T]}}(undef, num_foils)
    flow = 0
    for i in 1:num_foils
        @show i
        for (k,v) in kwargs
            if size(v,1) ==1
                pos[k] = v
            else
                v = v[i,:]
                if size(v,1) == 1
                    pos[k] = v[1]
                else
                    pos[k] = v
                end
            end
        end   
        # @show pos     
        foil, flow = init_params(; pos...)      
        @show flow          
        foil.foil[1, :] .+= starting_positions[1, i] * foil.chord
        foil.foil[2, :] .+= starting_positions[2, i]
        foil.LE = [minimum(foil.foil[1, :]), foil.foil[2, (foil.N ÷ 2 + 1)]]
        norms!(foil)
        set_collocation!(foil)
        move_edge!(foil, flow;startup=true)
        foil.edge[1, end] = 2.0 * foil.edge[1, 2] - foil.edge[1, 1]
        foils[i] =  foil
    end
    @show flow
    foils, flow
end



begin
    num_foils = 2
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [0.0 1.5 ; 0.0 0.0 ]
    phases = [pi/2, -pi/2]
    θ0 = deg2rad(10)
    h0 = 0.0
    motion_parameters = [h0 θ0 ; -h0 θ0]

    foils, flow = create_foils(num_foils, starting_positions, :make_heave_pitch;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 5, Nt = 64);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake, old_mus, old_phis)

    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake, old_mus, old_phis; mask=[false, true])    
        xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        plot(foils, wake)#; xlims=xlims, ylims=ylims)
    end
    gif(movie, "newMulti.gif", fps = 30)
end

begin
    num_foils = 4
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [0.0 1.5 3.0 1.5; 0.0 0.5 0.0 -0.5 ]
    phases = [pi/2, -pi/2, -pi/2, pi/2]
    θ0 = deg2rad(5)
    h0 = 0.1
    motion_parameters = [h0 θ0 ; -h0 θ0; -h0 θ0 ; h0 θ0]

    foils, flow = create_foils(num_foils, starting_positions, :make_heave_pitch;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 5, Nt = 64);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake, old_mus, old_phis)

    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake, old_mus, old_phis; mask=[false, true, true, true])    
        xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        plot(foils, wake)#; xlims=xlims, ylims=ylims)
    end
    gif(movie, "newMulti.gif", fps = 30)
end
begin
    num_foils = 3
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [0.0 1.5 1.5; 0.0 0.5 -0.5 ]
    phases = [pi/2, -pi/2,  pi/2]
    θ0 = deg2rad(5)
    h0 = 0.1
    motion_parameters = [0.0 0.0 ; -h0 θ0; h0 θ0]

    foils, flow = create_foils(num_foils, starting_positions, :make_heave_pitch;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 5, Nt = 64);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake, old_mus, old_phis)

    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake, old_mus, old_phis; mask=[false, true, true])    
        xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        plot(foils, wake)#; xlims=xlims, ylims=ylims)
    end
    gif(movie, "newMulti.gif", fps = 30)
end
function vel_track(foils)
    plot()
    for (i,f) in enumerate(foils)
        plot!(f.panel_vel[1,:],label="U_$(i)")
        plot!(f.panel_vel[2,:],label="V_$(i)")
    end
    plot!()
end

begin
    ##now do the same with one swimmer and track its kutta mu
    heave_pitch = deepcopy(defaultDict)

    heave_pitch[:f] = 1.0
    heave_pitch[:Uinf] = 1
    heave_pitch[:kine] = :make_heave_pitch
    θ0 = deg2rad(5)
    h0 = 0.05
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    kutta = zeros(flow.Ncycles * flow.N)
    for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        kutta[i] = foil.μ_edge[1]

    end
end
plot(vcat(kuttas,kutta')')


begin
    ##lets run the time_increment per time-step and look at what gets constructed for buff and sigma
    num_foils = 2
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [0.0 0.0 ; 0.0 1000.0 ]
    foils, flow = create_foils(num_foils, starting_positions)
    wake = Wake(foils)
    (foils)(flow)

    heave_pitch = deepcopy(defaultDict)

    heave_pitch[:f] = 1.0
    heave_pitch[:Uinf] = 1
    heave_pitch[:kine] = :make_heave_pitch
    θ0 = deg2rad(5)
    h0 = 0.05
    heave_pitch[:motion_parameters] = [h0, θ0]
    sfoil, sflow = init_params(; heave_pitch...)
    swake = Wake(sfoil)
    (sfoil)(sflow)    
end
begin
    time_increment!(flow, foils, wake, old_mus, old_phis)
    time_increment!(sflow, sfoil, swake)
    ### CODE CHUNK FOR multi
    A, rhs, edge_body = make_infs(foils)
    [setσ!(foil, flow) for foil in foils]
    
    σs = zeros(totalN)
    buff = zeros(totalN)
    for (i, foil) in enumerate(foils)
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        # foil.σs -= normal_wake_ind[:]
        σs[((i - 1) * foil.N + 1):(i * foil.N)] = foil.σs
        buff += (edge_body[:, i] * foil.μ_edge[1])
    end
    μs = A \ (-rhs * σs - buff)
    ### CODE CHUNK FOR single
    sA, srhs, sedge_body = make_infs(sfoil)

    setσ!(sfoil, sflow)
    sfoil.wake_ind_vel = vortex_to_target(swake.xy, sfoil.col, swake.Γ, sflow)
    normal_wake_ind = sum(sfoil.wake_ind_vel .* sfoil.normals, dims = 1)'
    # sfoil.σs -= normal_wake_ind[:]
    sbuff = sedge_body * sfoil.μ_edge[1]
    sμs = sA \ (-srhs * sfoil.σs- sbuff)[:]
    @show norm(μs - [sμs' sμs']', 2)/norm(μs,2)
    plot(μs,marker=:circle,lw=0)
    plot!([sμs' sμs']', marker=:dtri,lw=0)    

    # plot(sfoil.σs)
    # plot!(foils[1].σs) #<--- this is the problem
end
begin
    #show a comparisons between the single and multi body representations
    @show sum(rhs[1:64,1:64] - srhs), sum(rhs[65:end,65:end] - srhs)
    @show sum(σs[1:64] - sfoil.σs) , sum(σs[65:end] - sfoil.σs)
    @show μs[end] - μs[65] - (sμs[end] - sμs[1])
    @show sum(sfoil.col - foils[1].col)
    @show sum(sfoil.edge - foils[1].edge)
    @show sfoil.μ_edge[1] - foils[1].μ_edge[1]
    @show wake.Γ[1] - swake.Γ[1]
    @show sum(wake.xy[:,1:2:end] - swake.xy)
    nothing 
end
begin
    # do it with gory detal
    steps = flow.N * flow.Ncycles

    movie = @animate for t in 1:steps
        A, rhs, edge_body = make_infs(foils) #̌#CHECK
        [setσ!(foil, flow) for foil in foils] #CHECK
        σs = [] #CHECK
        buff = []
        for (i, foil) in enumerate(foils)
            foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
            normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
            foil.σs -= normal_wake_ind[:]
            push!(σs, foil.σs...)
            push!(buff, (edge_body[:, i] * foil.μ_edge[1])...)
        end
        μs = A \ (-rhs * σs - buff)
        for (i, foil) in enumerate(foils)
            foil.μs = μs[((i - 1) * foil.N + 1):(i * foil.N)]
        end
        set_edge_strength!.(foils)
        cancel_buffer_Γ!(wake, foils)
        wake_self_vel!(wake, flow)
        totalN = sum(foil.N for foil in foils)
        phis = zeros(totalN)
        ps = zeros(totalN)
        old_mus = zeros(3, totalN)
        old_phis = zeros(3, totalN)
        coeffs = zeros(length(foils), 4, steps)
        for (i, foil) in enumerate(foils)
            body_to_wake!(wake, foil, flow)
            phi = get_phi(foil, wake)
            phis[((i - 1) * foils[i].N + 1):(i * foils[i].N)] = phi
            p = panel_pressure(foil,
                flow,
                old_mus[:, ((i - 1) * foils[i].N + 1):(i * foils[i].N)],
                old_phis[:, ((i - 1) * foils[i].N + 1):(i * foils[i].N)],
                phi)
            ps[((i - 1) * foils[i].N + 1):(i * foils[i].N)] = p
            coeffs[i, :, 1] .= get_performance(foil, flow, p)
        end
        old_mus = [μs'; old_mus[1:2, :]]
        old_phis = [phis'; old_phis[1:2, :]]
        f = plot(foils, wake)
        # wake.xy = sdf_fence(wake, foils, flow; mask = [0, 1, 0, 0] .|> Bool)
       
        # move_wake!(wake, flow)
        for foil in foils
            release_vortex!(wake, foil)
        end
        (foils)(flow)
       
        f
    end
    gif(movie, "multi.gif", fps = 30)
end

function time_increment!(flow::FlowParams, foils::Vector{Foil}, wake::Wake)
    A, rhs, edge_body = make_infs(foils) #̌#CHECK
    [setσ!(foil, flow) for foil in foils] #CHECK
    σs = [] #CHECK
    buff = []
    for (i, foil) in enumerate(foils)
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        foil.σs -= normal_wake_ind[:]
        push!(σs, foil.σs...)
        push!(buff, (edge_body[:, i] * foil.μ_edge[1])...)
    end
    μs = A \ (-rhs * σs - buff)
    for (i, foil) in enumerate(foils)
        foil.μs = μs[((i - 1) * foil.N + 1):(i * foil.N)]
    end
    set_edge_strength!.(foils)
    cancel_buffer_Γ!(wake, foils)
    wake_self_vel!(wake, flow)
    totalN = sum(foil.N for foil in foils)
    phis = zeros(totalN)
    ps = zeros(totalN)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    for (i, foil) in enumerate(foils)
        body_to_wake!(wake, foil, flow)
        phi = get_phi(foil, wake)
        phis[((i - 1) * foils[i].N + 1):(i * foils[i].N)] = phi
        p = panel_pressure(foil,
            flow,
            old_mus[:, ((i - 1) * foils[i].N + 1):(i * foils[i].N)],
            old_phis[:, ((i - 1) * foils[i].N + 1):(i * foils[i].N)],
            phi)
        ps[((i - 1) * foils[i].N + 1):(i * foils[i].N)] = p
        coeffs[i, :, 1] .= get_performance(foil, flow, p)
    end
    old_mus = [μs'; old_mus[1:2, :]]
    old_phis = [phis'; old_phis[1:2, :]]

    # move_wake!(wake, flow)
    for foil in foils
        release_vortex!(wake, foil)
    end
    do_kinematics!(foils, flow)
    nothing
end