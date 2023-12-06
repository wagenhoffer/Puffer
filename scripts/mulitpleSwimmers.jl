using Puffer
using Plots
using LinearAlgebra

function create_foils(num_foils, starting_positions)
    # Ensure starting_positions has correct dimensions
    if size(starting_positions) != (2, num_foils)
        error("starting_positions must be a 2xN array, where N is the number of foils")
    end

    pos1 = deepcopy(defaultDict)
    pos1[:kine] = :make_heave_pitch
    θ0 = deg2rad(5)
    h0 = 0.05
    pos1[:motion_parameters] = [h0, θ0]    
    foil1, flow = init_params(; pos1...)

    foils = Vector{typeof(foil1)}(undef, num_foils)
    for i in 1:num_foils
        foil = deepcopy(foil1)
        foil.foil[1, :] .+= starting_positions[1, i] * foil1.chord
        foil.foil[2, :] .+= starting_positions[2, i]
        foil.LE = [minimum(foil.foil[1, :]), foil.foil[2, (foil.N ÷ 2 + 1)]]
        norms!(foil)
        set_collocation!(foil)
        move_edge!(foil, flow;startup=true)
        foil.edge[1, end] = 2.0 * foil.edge[1, 2] - foil.edge[1, 1]
        foils[i] =  foil
    end

    foils, flow
end



begin
    num_foils = 2
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [0.0 0.0 ; 100.0 -100.0 ]
    foils, flow = create_foils(num_foils, starting_positions)
    wake = Wake(foils)
    (foils)(flow)
    steps = flow.N * flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake, old_mus, old_phis)
    kuttas[:, 1] .= [foil.μ_edge[1] for foil in foils]
    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake, old_mus, old_phis)        
        f1TEx = foils[1].foil[1, end] .+ (-1.25, 1.25)
        f1TEy = foils[1].foil[2, end] .+ (-0.5, 0.5)
        plot(foils, wake;xlims=f1TEx, ylims=f1TEy)        
        kuttas[:, t] .= [foil.μ_edge[1] for foil in foils]
        # plot!(foils[1].edge[1,:],foils[1].edge[2,:], color = :green, lw = 2,label="")
    end
    gif(movie, "newMulti.gif", fps = 30)
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