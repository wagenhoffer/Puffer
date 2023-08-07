# modify functions
function self_propell(foil::Foil,flow::FlowParams;forces=nothing, U = [0.0, 0.0])
    #only effects the position of the LE with the forces defining the new velocity of self-propellsion
    #perform kinematics
    mass = 0.1
    le = foil.foil[:,foil.N÷2+1]    
    if isnothing(forces)
        forces = zeros(4)
    end
    #force, lift, thrust, power 
    accel = forces[3:-1:2]/mass
    Un1 = U + accel*flow.Δt
    len1 = le + 0.5*(Un1)*flow.Δt
    if typeof(foil.kine) == Vector{Function}
        h = foil.kine[1](foil.f, flow.n * flow.Δt)
        θ = foil.kine[2](foil.f, flow.n * flow.Δt, -π/2)
        rotate_about!(foil, θ)
        foil.foil[1, :] .-= len1[1]
        foil.foil[2, :] .+= h #+ len1[2]
        #Advance the foil in flow
        # foil.foil .+= [-flow.Uinf, 0] .* flow.Δt #.* flow.n
    else
        foil.foil[2, :] = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, flow.n * flow.Δt)
        #Advance the foil in flow
        foil.foil .+= [-flow.Uinf, 0] .* flow.Δt
    end

    norms!(foil)
    set_collocation!(foil)
    move_edge!(foil, flow)
    flow.n += 1    
    Un1
end

begin
    #scripting
    upm = deepcopy(defaultDict)
    upm[:Nt]        = 64
    upm[:N]         = 64
    upm[:Ncycles]   = 3    
    upm[:Uinf]      = 1.0
    upm[:kine]      = :make_heave_pitch
    upm[:pivot]     = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick]     = 0.12
    upm[:f] = 1.0
    h0 = -0.01
    θ0 = deg2rad(0)	
    upm[:motion_parameters] = [h0, θ0]

    foil, flow = init_params(;upm...)
    wake = Wake(foil)
    Us = zeros(2, flow.Ncycles*flow.N)
    U = (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)

    anim = Animation()    
    for i = 1:flow.Ncycles*flow.N
        if flow.n != 1
            move_wake!(wake, flow)   
            release_vortex!(wake, foil)
        end    
        if i == 1
            U = (foil)(flow)
        else
            U = (foil)(flow; forces= coeffs[:,i-1], U = U)
        end
        Us[:,i] = U
        A, rhs, edge_body = make_infs(foil)
        A[getindex.(A .== diag(A))] .= 0.5
        setσ!(foil, flow)    
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs*foil.σs-buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)    
        phi =  get_phi(foil, wake)                                   
        p = panel_pressure(foil, flow,  old_mus, old_phis, phi)        
        
        old_mus = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] = get_performance(foil, flow, p)
        ps[:,i] = p
        f = plot_current(foil, wake)
        plot!(f, ylims=(-1,1))
        plot!(title="$(U)")
        frame(anim, f)
    end
    gif(anim, "handp.gif", fps=10)
end