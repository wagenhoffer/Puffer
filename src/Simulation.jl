#  Simulation functions
"""
    (foil::Foil)(fp::FlowParams)

"""
function (foil::Foil)(flow::FlowParams)
    #perform kinematics
    if typeof(foil.kine) == Vector{Function}
        h = foil.kine[1](foil.f, flow.n * flow.Δt)
        θ = foil.kine[2](foil.f, flow.n * flow.Δt, -π/2)
        rotate_about!(foil, θ)
        foil.foil[2, :] .+= h
        #Advance the foil in flow
        foil.foil .+= [-flow.Uinf, 0] .* flow.Δt .* flow.n
    else
        foil.foil[2, :] = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, flow.n * flow.Δt)
        #Advance the foil in flow
        foil.foil .+= [-flow.Uinf, 0] .* flow.Δt
    end
    norms!(foil)
    set_collocation!(foil)
    move_edge!(foil, flow)
    flow.n += 1
end


"""
    run_sim(; steps = flow.N*10, aoa = rotate(-0*pi/180)')


TBW
"""
function run_sim(; kwargs...)
    # Initialize the foil and flow parameters
    foil, flow = init_params(; kwargs...)
    steps = flow.N * flow.Ncycles
    # flow.δ /= 4
    den = 1 / 2 * (flow.Uinf^2 * foil.chord) # we avoid \rho unless needed
    # Rotate the foil based on the angle of attack (aoa)
    if haskey(kwargs, :aoa)
        foil._foil = (foil._foil' * rotation(kwargs[:aoa])')'            
    end
    
    # Create a wake object for the foil
    wake = Wake(foil)
    # Initialize the previous time step's vorticity values
    old_mus = zeros(3, foil.N)
    old_phis = zeros(3, foil.N)
    #coefficients of force, lift, thrust, power
    coeffs = zeros(4, steps)
    (foil)(flow)
    for i = 1:steps
        A, rhs, edge_body = make_infs(foil)
        A[getindex.(A .== diag(A))] .= 0.5
        setσ!(foil, flow)
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs*foil.σs - buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        phi =  get_phi(foil, wake)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)
        old_mus = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] .= get_performance(foil, flow, p)
             
        move_wake!(wake, flow)
        release_vortex!(wake, foil)
        (foil)(flow)

    end
    foil, flow,  wake, coeffs 
end


function get_performance(foil, flow, p)
    dforce = repeat(-p .* foil.panel_lengths', 2, 1) .* foil.normals
    dpress = sum(dforce .* foil.panel_vel, dims=2)
    force  = sum(dforce, dims=2)
    lift   = -force[2]
    thrust = force[1]
    power  = sum(dpress, dims=1)[1]

    [sum(sqrt.(force .^ 2)), 
     lift,    
     thrust,
     power / flow.Uinf]      
end


"""
    time_increment!(flow::FlowParams, foil::Foil, wake::Wake)

does a single timestep of the simulation. This is the core of the simulation.
Useful for debugging or for plotting purposes.

Reorder the operations so it ends with a fully defined system. This way we can
grab metrics of the foil. 
"""
function time_increment!(flow::FlowParams, foil::Foil, wake::Wake)
    if flow.n != 1
        move_wake!(wake, flow)   
        release_vortex!(wake, foil)
    end    
    (foil)(flow)
    
    A, rhs, edge_body = make_infs(foil)
    A[getindex.(A .== diag(A))] .= 0.5
    setσ!(foil, flow)    
    foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
    normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'
    foil.σs -= normal_wake_ind[:]
    buff = edge_body * foil.μ_edge[1]
    # prob = LinearProblem(A, (-rhs*foil.σs-buff)[:])
    foil.μs = A \ (-rhs*foil.σs-buff)[:]
    # sol = solve(prob)
    # foil.μs = sol.u
    set_edge_strength!(foil)
    cancel_buffer_Γ!(wake, foil)
    body_to_wake!(wake, foil, flow)
    wake_self_vel!(wake, flow)    
    nothing
end

function solve_n_update!(flow::FlowParams, foil::Foil, wake::Wake)
    A, rhs, edge_body = make_infs(foil)
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
    nothing
end

function time_increment!(flow::FlowParams, foils::Vector{Foil}, wake::Wake)
    A, rhs, edge_body = make_infs(foils) 
    [setσ!(foil, flow) for foil in foils] 
    σs = [] 
    buff = []
    for (i, foil) in enumerate(foils)
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'
        foil.σs -= normal_wake_ind[:]
        push!(σs, foil.σs...)
        push!(buff, (edge_body[:,i] * foil.μ_edge[1])...)    
    end
    μs =  A \ (-rhs*σs - buff)
    for (i, foil) in enumerate(foils)
        foil.μs = μs[(i-1)*foil.N+1:i*foil.N]
    end
    set_edge_strength!.(foils)
    cancel_buffer_Γ!(wake, foils)
    wake_self_vel!(wake, flow)
    totalN   = sum(foil.N for foil in foils)
    phis     = zeros(totalN)
    ps       = zeros(totalN)
    old_mus  = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs   = zeros(length(foils), 4, steps)
    for  (i, foil) in enumerate(foils)
        body_to_wake!(wake, foil, flow)
        phi =  get_phi(foil, wake)
        phis[(i-1)*foils[i].N+1:i*foils[i].N] = phi
        p  = panel_pressure(foil, flow, old_mus[:,(i-1)*foils[i].N+1:i*foils[i].N],
                                                old_phis[:,(i-1)*foils[i].N+1:i*foils[i].N], phi)
        ps[(i-1)*foils[i].N+1:i*foils[i].N] = p
        coeffs[i, : , 1] .= get_performance(foil, flow, p)
    end
    old_mus = [μs'; old_mus[1:2,:]]
    old_phis = [phis'; old_phis[1:2,:]]
        
    move_wake!(wake, flow)
    for foil in foils
        release_vortex!(wake, foil)
    end
    do_kinematics!(foils,flow)
    nothing
end


function get_phi(foil::Foil, wake::Wake)
    phi = zeros(foil.N)           
    for i = 1:size(wake.Γ)[1]
        dx = foil.col[1, :] .- wake.xy[1, i]
        dy = foil.col[2, :] .- wake.xy[2, i]
        @. phi = -wake.Γ[i] *mod2pi(atan(dy,dx))/(2π)        
    end
    phi
end


#initial allocation - not used (we don't care about the pressures at start now)
function get_dmudt!(foil::Foil, flow::FlowParams)
    old_mus = zeros(3, foil.N)
    old_mus[1, :] = foil.μs
    foil.μs / flow.Δt, old_mus
end

#after init
function get_dmudt!(old_mus, foil::Foil, flow::FlowParams)
    dmudt = (3 * foil.μs - 4 * old_mus[1, :] + old_mus[2, :]) / (2 * flow.Δt)
    old_mus = circshift(old_mus, (1, 0))
    old_mus[1, :] = foil.μs
    dmudt, old_mus
end

function get_dphidt!(oldphi, phi, flow::FlowParams)
    dmudt = (3 * phi - 4 * oldphi[1, :] + oldphi[2, :]) / (2 * flow.Δt)
    oldphi = circshift(oldphi, (1, 0))
    oldphi[1, :] = phi
    dmudt, oldphi
end

function get_dt(values, flow::FlowParams)
     (3 * values[1, :] - 4 * values[2, :] + values[3, :]) / (2 * flow.Δt)
end

function roll_values!(oldvals, newval)
    oldvals = circshift(oldvals, (1, 0))
    oldvals[1, :] = newval
    oldvals    
end
function get_qt(foil::Foil)
    acc_lens = 0.5 * foil.panel_lengths[1:end-1] + 0.5 * foil.panel_lengths[2:end]
    acc_lens = [0; cumsum(acc_lens)]

    # B = BSplineBasis(BSplineOrder(4), acc_lens[:])
    S = interpolate(acc_lens, foil.μs, BSplineOrder(3))
    dmu = Derivative(1) * S
    dmudl = -dmu.(acc_lens)

    #TODO: SPLIT σ into local and inducec? it is lumped into one now
    qt = repeat(dmudl', 2, 1) .* foil.tangents #+ repeat(foil.σs',2,1).*foil.normals
    qt
end