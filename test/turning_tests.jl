# '''
# Below we decompose the signal from the heave and pitch kinematic and use it to inform how the system should
# rotate mutating foil.θ instead. 
# This shows that the systems are effectively equivalent, although the AD on the kinematic signal will be more
# accurate than the 1st order velocity stencil used for the turnto method.

# This also directly informs how to augment foil.σ, as foil.panel_vel will just be a linear combination of the
# kinematic signal vel and the discrete velocity, aka add them
# '''
# using BemRom
include("../src/BemRom.jl") 

begin
    #running a heave and pitch sim
    upm = deepcopy(defaultDict)
    upm[:Nt] = 64
    upm[:N] = 64
    upm[:Ncycles] = 2
    upm[:Uinf] = 1.0
    upm[:kine] = :make_heave_pitch
    upm[:pivot] = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick] = 0.12
    upm[:f] = 1.0
    h0 = 0.0
    θ0 = deg2rad(5)
    upm[:motion_parameters] = [h0, θ0]

    foil, flow = init_params(; upm...)
    # grab the θ signal and then push it directly as a turning radius signal
    # ensure that the σ values and performance metrics are the same for both methods
    theta(_t) = foil.kine[2](foil.f, _t, π / 2)
    t = 1:flow.Ncycles*flow.N
    t *= flow.Δt
    plot(t, theta.(t))
    flow.n = 0
    #look at velocity decomposition first
    turns = theta.(t) .* flow.Δt
    discrete_vels = zeros(2, foil.N, flow.N * flow.Ncycles)
    ad_vels = zeros(2, foil.N, flow.N * flow.Ncycles)
    col = get_mdpts(foil._foil)
    movie = @animate for (i, turn) in enumerate(turns)
        # discrete_vels[:,:,i]= (rot(turn)*col-foil.col)
        orig_pos = deepcopy(foil.col)
        _propel(foil, flow; turnto=turn)
        discrete_vels[:, :, i] = (orig_pos - foil.col) ./ flow.Δt

        # @show -(foil.col - orig_pos) ./ flow.Δt - (foil.col - rot(foil.θ-turn)*foil.col)./flow.Δt         
        v = get_panel_vels(foil, flow)
        ad_vels[1, :, i] = v[1]
        ad_vels[2, :, i] = v[2]
        f = plot(foil.col[1, :], foil.col[2, :], aspect_ratio=:equal, ylims=(-0.3, 0.3))
        f
    end
    gif(movie, "./images/forced_theta.gif")
    offset = 5
    au = plot(ad_vels[1, :, offset:end], st=:contourf)
    av = plot(ad_vels[2, :, offset:end], st=:contourf)
    du = plot(-discrete_vels[1, end:-1:1, offset:end] .+ flow.Uinf, st=:contourf)
    dv = plot(discrete_vels[2, :, offset:end], st=:contourf)
    plot(au, av, du, dv, layout=(2, 2), size=(1200, 800))
end

# """ the next to code blocks show that performance metrcs are consistent for both methods, though small diffs 
# and a small lag occur. 
# the second script is also wildly adhoc for now 
# """
begin
    foil, flow = init_params(;upm...)
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs1 = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)        
        phi =  get_phi(foil, wake)    # (output space) <-probably not that important                            
        p = panel_pressure(foil, flow,  old_mus, old_phis, phi)    
        # (output space) <- p is a function of μ and we should be able to recreate this     
        old_mus = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs1[:,i] = get_performance(foil, flow, p)
        # the coefficients of PERFROMANCE are important, but are just a scaling of P
        # if we can recreate p correctly, this will be easy to get also (not important at first)
        ps[:,i] = p # storage container of the output, nice!
    end
    t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs1[1,start:end], label="Force"  ,lw = 3, marker=:circle)
    b = plot(t[start:end], coeffs1[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
    c = plot(t[start:end], coeffs1[3,start:end], label="Thrust" ,lw = 3, marker=:circle)
    d = plot(t[start:end], coeffs1[4,start:end], label="Power"  ,lw = 3, marker=:circle)
    plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(800,800))
end

begin
    foil, flow = init_params(;upm...)
    wake = Wake(foil)
    _propel(foil, flow; turnto=turns[1]) 
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs2 = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    movie = Animation()
    for i in 1:flow.Ncycles*flow.N
        turn = turns[i]
        if flow.n != 1
            move_wake!(wake, flow)   
            release_vortex!(wake, foil)
        end    
        orig_pos = deepcopy(foil.col)
        _propel(foil, flow; turnto=turn)        
         _vels = (orig_pos - foil.col ) ./ flow.Δt .-[flow.Uinf 0.0]'
         _vels[1, :, :] = -_vels[1, end:-1:1, :]
        foil.panel_vel = _vels
        A, rhs, edge_body = make_infs(foil)

        foil.σs = (-flow.Uinf .+ foil.panel_vel[1, :]).* foil.normals[1, :] +
                  foil.panel_vel[2, :] .* foil.normals[2, :]
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
        coeffs2[:,i] = get_performance(foil, flow, p)
        
        ps[:,i] = p # storage container of the output, nice!
        f = plot_current(foil,wake)
        frame(movie, f)
    end
    gif(movie)
    t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs2[1,start:end], label="Force"  ,lw = 3, marker=:circle)
    b = plot(t[start:end], coeffs2[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
    c = plot(t[start:end], coeffs2[3,start:end], label="Thrust" ,lw = 3, marker=:circle)
    d = plot(t[start:end], coeffs2[4,start:end], label="Power"  ,lw = 3, marker=:circle)
    plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(800,800))
end

c_diff = coeffs1 - coeffs2
t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
start = flow.N
a = plot(t[start:end], c_diff[1,start:end], label="Force"  ,lw = 3, marker=:circle)
b = plot(t[start:end], c_diff[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
c = plot(t[start:end], c_diff[3,start:end], label="Thrust" ,lw = 3, marker=:circle)
d = plot(t[start:end], c_diff[4,start:end], label="Power"  ,lw = 3, marker=:circle)
plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(800,800))
norm(c_diff, 2) / norm(coeffs1,2)