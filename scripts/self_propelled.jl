using BemRom

using Plots

"""
    turn_σ!(foil::Foil,flow::FlowParams, turn)

TBW
"""
function turn_σ!(foil::Foil,flow::FlowParams, turn)
    ff = get_mdpts(([foil._foil[1, :] .- foil.pivot  foil._foil[2, :].+  foil.kine.(foil._foil[1, :], foil.f, foil.k, flow.n * flow.Δt)] 
* rotation(-turn))') .+ foil.LE
    vel = (foil.col - ff)./flow.Δt 
    foil.σs += vel[1, :] .* foil.normals[1, :] +
               vel[2, :] .* foil.normals[2, :]
    nothing
end

begin
    #scripting
    upm = deepcopy(defaultDict)
    upm[:Nt]        = 64
    upm[:N]         = 64
    upm[:Ncycles]   = 4 
    upm[:Uinf]      = 1.0
    upm[:kine]      = :make_ang
    upm[:pivot]     = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick]     = 0.12
    upm[:f] = 1.0
    upm[:k] = 1.0
  

    foil, flow = init_params(;upm...)
    mass = 1e-1
    turns = zeros(flow.Ncycles*flow.N)
    wake = Wake(foil)
    Us = zeros(2, flow.Ncycles*flow.N)
    U = _propel(foil, flow;  U = [0.0, 0.0], mass = mass, turnto=turns[1])
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    stop = π
    steps = flow.N*flow.Ncycles
    turns = collect(foil.θ:stop/steps:stop).*0
    # trs = 0.0
    anim = Animation()    
    for i = 1:flow.Ncycles*flow.N
        if flow.n != 1
            move_wake!(wake, flow)   
            release_vortex!(wake, foil)
        end    
        if i == 1
            U = _propel(foil, flow;  U = [0.0, 0.0], mass = mass, turnto=turns[i])
            # Cast this to  2d vel
            U = [-1.0, 0.0]
        else
            U = _propel(foil, flow; forces= coeffs[:,i-1], U = U, mass = mass, turnto=turns[i], self_prop = true)
            # U = (foil)(flow)
        end
        # @show U
        Us[:,i] .= U
        U_bx = -U[1]
        A, rhs, edge_body = make_infs(foil)

        setσ!(foil, flow; U_b = U_bx)    
        get_panel_vels!(foil, flow)
        foil.σs = (-U[1] .+ foil.panel_vel[1, :]) .* foil.normals[1, :] +
                  (foil.panel_vel[2, :]) .* foil.normals[2, :]
        
        
        # σ is set - now pull in the information from the turn radius
        # turn_σ!(foil, flow, trs[i])
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
        p = panel_pressure(foil, flow,  old_mus, old_phis, phi; U_b = U_bx)        
        
        old_mus  = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] = get_performance(foil, flow, p)
        ###FAKE COEFF Thrust
        coeffs[3,i] = 0.2.*cos.(4π.*foil.f.*  i *flow.Δt )
        ps[:,i] = p
        f = plot_current(foil, wake)
        plot!(f, ylims=(-1,1))
        plot!(title="$(U)")
        frame(anim, f)
    end
    gif(anim, "./images/self_prop_ang.gif", fps=10)
end

t = flow.Δt:flow.Δt:flow.N*flow.Δt*flow.Ncycles
plot!(-0.2.*cos.(4π.*foil.f.*  t ))

"""
    trapezoid(f,a,b,n)

Apply the trapezoid integration formula for integrand `f` over
interval [`a`,`b`], broken up into `n` equal pieces. Returns
the estimate, a vector of nodes, and a vector of integrand values at the
nodes.
"""
function trapezoid(y,a,b,n)
    h = (b-a)/n
    t = range(a,b,length=n+1)
    # y = f.(t)
    T = h * ( sum(y[2:n]) + 0.5*(y[1] + y[n+1]) )
    return T,t,y
end

"""
    turn_σ!(foil::Foil,flow::FlowParams, turn)

    modify σ for a discrete turn, ie not part of the kinematics functional signal
"""


begin
    #scripting
    upm = deepcopy(defaultDict)
    upm[:Nt]        = 64
    upm[:N]         = 64
    upm[:Ncycles]   = 2  
    upm[:Uinf]      = 1.0
    upm[:kine]      = :make_heave_pitch
    upm[:pivot]     = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick]     = 0.12
    upm[:f] = 1.0
    h0 = 0.0
    θ0 = deg2rad(4)	
    upm[:motion_parameters] = [h0, θ0]

    foil, flow = init_params(;upm...)
    # foil.θ= π/2*0.0
    U = _propel(foil,flow)
    wake = Wake(foil)
  
    Us = zeros(2, flow.Ncycles*flow.N)

    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    turnrad = -π*flow.Δt/4.0
    trs = zeros(flow.N*flow.Ncycles)
    trs[flow.N:2*flow.N-32] .= turnrad
    anim = Animation()    
    # f = plot()
    for i = 1:flow.Ncycles*flow.N           
        if flow.n != 1
            move_wake!(wake, flow)   
            release_vortex!(wake, foil)
        end    
        if i == 1
            U = (foil)(flow)
            # Cast this to  2d vel
            U = [0.0, 0.0]
        else
            U = _propel(foil, flow; forces= coeffs[:,i-1], U = [0.0, 0.0], mass = 1e-5, turn=trs[i])
            # U = (foil)(flow)
        end
        Us[:,i] = U
        A, rhs, edge_body = make_infs(foil)
        A[getindex.(A .== diag(A))] .= 0.5
        setσ!(foil, flow)    
        turn_σ!(foil, flow, trs[i])
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
        
        old_mus  = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] = get_performance(foil, flow, p)
        ps[:,i] = p
        f = plot_current(foil, wake)
        plot!(f, ylims=(-1,1))
        plot!(title="$(U)")
        frame(anim, f)
        # plot!(f, foil.foil[1,:],foil.foil[2,:], label="", aspect_ratio=:equal)
    end
    # f
    gif(anim, "./images/handp.gif", fps=10)
end


begin
    #scripting
    upm = deepcopy(defaultDict)
    upm[:Nt]        = 64
    upm[:N]         = 64
    upm[:Ncycles]   = 2  
    upm[:Uinf]      = 1.0
    upm[:kine]      = :make_heave_pitch
    upm[:pivot]     = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick]     = 0.12
    upm[:f] = 1.0
    h0 = 0.05
    θ0 = deg2rad(0)	
    upm[:motion_parameters] = [h0, θ0]

    foil, flow = init_params(;upm...)
    foil.θ= π/2*0.0
    U = _propel(foil,flow)
    wake = Wake(foil)
  
    Us = zeros(2, flow.Ncycles*flow.N)

    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    turnrad = π*flow.Δt*0.0
    trs = zeros(flow.N*flow.Ncycles)
    trs[flow.N:2*flow.N-32] .= turnrad
    anim = Animation()    
    # f = plot()
    for i = 1:flow.Ncycles*flow.N
        if i == 3
            wake.Γ = [wake.Γ[1]]
            wake.xy = hcat(wake.xy[:,1])
            wake.uv = hcat(wake.uv[:,1])
        end        
        if flow.n != 1
            move_wake!(wake, flow)   
            release_vortex!(wake, foil)
        end    
        if i == 1
            U = (foil)(flow)
            # Cast this to  2d vel
            U = [0.0, 0.0]
        else
            U = self_propell(foil, flow; forces= coeffs[:,i-1], U = [0.0, 0.0], mass = 1e-5, turn=trs[i])
            # U = (foil)(flow)
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
        
        old_mus  = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] = get_performance(foil, flow, p)
        ps[:,i] = p
        f = plot_current(foil, wake)
        plot!(f, ylims=(-1,1))
        plot!(title="$(U)")
        frame(anim, f)
        # plot!(f, foil.foil[1,:],foil.foil[2,:], label="", aspect_ratio=:equal)
    end
    # f
    gif(anim, "./images/handp.gif", fps=10)
end


begin 
    #running a heave and pitch sim
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
    h0 = 0.0
    θ0 = deg2rad(5)	
    upm[:motion_parameters] = [h0, θ0]

    foil, flow = init_params(;upm...)
    # grab the θ signal and then push it directly as a turning radius signal
    # ensure that the σ values and performance metrics are the same for both methods
    theta(_t) = foil.kine[2](foil.f, _t, -π/2)
    t = 1:flow.Ncycles*flow.N
    t *= flow.Δt
    plot(t, theta.(t))
    flow.n = 0
    #look at velocity decomposition first
    turns = theta.(t).*flow.Δt
    discrete_vels = zeros(2, foil.N, flow.N*flow.Ncycles)
    ad_vels = zeros(2, foil.N, flow.N*flow.Ncycles)
    col = get_mdpts(foil._foil)
    movie = @animate for (i, turn) in enumerate(turns)        
        # discrete_vels[:,:,i]= (rot(turn)*col-foil.col)
        orig_pos  = deepcopy(foil.col)
        _propel(foil,flow; turnto=turn)
        discrete_vels[:,:,i]= -(foil.col - orig_pos) ./flow.Δt
        

        v = get_panel_vels(foil, flow)
        ad_vels[1,:,i] = v[1]
        ad_vels[2,:,i] = v[2]        
        f = plot(foil.col[1,:], foil.col[2,:],aspect_ratio=:equal, ylims=(-0.3,0.3))   
        f
    end
    gif(movie, "./images/forced_theta.gif")
    offset = 5
    au = plot(ad_vels[1,:,offset:end], st=:contourf)
    av = plot(ad_vels[2,:,offset:end], st=:contourf)
    du = plot(discrete_vels[1,:,offset:end] .- flow.Uinf , st=:contourf)
    dv = plot(discrete_vels[2,:,offset:end], st=:contourf)
    plot(au,av,du,dv, layout=(2,2), size = (1200,800))
end



a = plot(Us[1,:])
b = plot(Us[2,:])
plot(a,b, layout = (2,1))
