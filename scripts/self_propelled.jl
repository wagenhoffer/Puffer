using Puffer

using Plots

"""
    turn_σ!(foil::Foil, flow::FlowParams, turn)

This function modifies σ for a discrete turn, ie not part of the kinematics functional signal.
"""
function turn_σ!(foil::Foil, flow::FlowParams, turn)
    ff = get_mdpts(([foil._foil[1, :] .- foil.pivot foil._foil[2, :] .+
                                                    foil.kine.(foil._foil[1, :],
        foil.f,
        foil.k,
        flow.n * flow.Δt)] * rotation(-turn))') .+ foil.LE
    vel = (foil.col - ff) ./ flow.Δt
    foil.σs += vel[1, :] .* foil.normals[1, :] + vel[2, :] .* foil.normals[2, :]
    nothing
end

begin
    #scripting
    upm = deepcopy(defaultDict)
    upm[:Nt] = 64
    upm[:N] = 64
    upm[:Ncycles] = 10
    upm[:Uinf] = 1.0
    upm[:kine] = :make_ang
    upm[:pivot] = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick] = 0.12
    upm[:f] = 1.0
    upm[:k] = 1.25

    foil, flow = init_params(; upm...)
    mass = 1.0
    turns = zeros(flow.Ncycles * flow.N)
    wake = Wake(foil)
    Us = zeros(2, flow.Ncycles * flow.N)
    U = _propel(foil, flow; U = [0.0, 0.0], mass = mass, turnto = turns[1])
    #data containers
    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)
    stop = π
    steps = flow.N * flow.Ncycles
    turns = collect((foil.θ):(stop / steps):stop) .* 0
    # trs = 0.0
    anim = Animation()
    for i in 1:(flow.Ncycles * flow.N)
        if flow.n != 1
            move_wake!(wake, flow)
            release_vortex!(wake, foil)
        end
        if i == 1
            U = _propel(foil, flow; U = [0.0, 0.0], mass = mass, turnto = turns[i])
            # Cast this to  2d vel
            U = [0.95, 0.0]
        else
            U = _propel(foil,
                flow;
                forces = coeffs[:, i - 1],
                U = U,
                mass = mass,
                turnto = turns[i],
                self_prop = true,)
            # U = (foil)(flow)
        end
        # @show U
        Us[:, i] .= U
        U_bx = U[1]
        A, rhs, edge_body = make_infs(foil)

        setσ!(foil, flow; U_b = U_bx)
        get_panel_vels!(foil, flow)

        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs * foil.σs - buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        phi = get_phi(foil, wake)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi; U_b = U_bx)

        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
        coeffs[:, i] = get_performance(foil, flow, p)
        ###FAKE COEFF Thrust
        # coeffs[3,i] = 0.2.*cos.(4π.*foil.f.*  i *flow.Δt ) .+ 0.2
        # Make a fake coeff of thrust based on a sinusiodal signal and force it into

        ps[:, i] = p
        f = plot_current(foil, wake)
        plot!(f, ylims = (-1, 1))
        plot!(title = "$(U)")
        frame(anim, f)
    end

    @show maximum(coeffs[3,64:end])
    @show maximum(Us[1,64:end])
    # gif(anim, "./images/self_prop_ang.gif", fps = 10)
    plot(Us[1, 64:end])
end

    # MAKE A Plot FOR THE articial thrust signal
    # t = (flow.Δt):(flow.Δt):(flow.N * flow.Δt * flow.Ncycles)
    # plot(t, -0.2 .* cos.(4π .* foil.f .* t))
function get_peaks(signal, period, num_periods)
    " given a sinusoidal signal, skip the first period and then find the peaks"
    signal = signal[period:end]
    peaks = zeros(num_periods - 1)
    for i in 1:(num_periods - 1)
        start_index = (i - 1) * period + 1
        end_index = i * period
        peaks[i] = maximum(signal[start_index:end_index])
    end
    return peaks
end

peaks = get_peaks(coeffs[3,:], flow.N, flow.Ncycles)
diff(peaks)[end] < tolerance

function find_Us(;U_init = 1, St= 0.3, f=1.0, k=1.0, mass = 1.0, Ncycles = 3)
    upm = deepcopy(defaultDict)
    upm[:Nt] = 64
    upm[:N] = 64
    upm[:Ncycles] = Ncycles
    upm[:Uinf] = 1.0
    upm[:kine] = :make_ang
    upm[:pivot] = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick] = 0.12
    upm[:f] = f
    upm[:k] = k
    a0 = St / upm[:f]
    upm[:motion_parameters] = [a0]

    foil, flow = init_params(; upm...)
    mass = mass
    turns = zeros(flow.Ncycles * flow.N)
    wake = Wake(foil)
    Us = zeros(2, flow.Ncycles * flow.N)
    U = _propel(foil, flow; U = [0.0, 0.0], mass = mass, turnto = turns[1])
    #data containers
    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)

    # trs = 0.0

    for i in 1:(flow.Ncycles * flow.N)
        if flow.n != 1
            move_wake!(wake, flow)
            release_vortex!(wake, foil)
        end
        if i == 1
            U = _propel(foil, flow; U = [0.0, 0.0], mass = mass)
            # Cast this to  2d vel
            U = [U_init, 0.0]
        else
            U = _propel(foil,
                flow;
                forces = coeffs[:, i - 1],
                U = U,
                mass = mass,
                self_prop = true,)        
        end
       
        Us[:, i] .= U
        U_bx = U[1]
        A, rhs, edge_body = make_infs(foil)

        setσ!(foil, flow; U_b = U_bx)
        get_panel_vels!(foil, flow)

        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs * foil.σs - buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        phi = get_phi(foil, wake)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi; U_b = U_bx)

        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
        coeffs[:, i] = get_performance(foil, flow, p)

        ps[:, i] = p

    end
    Us
end

function us_relax(;n_loops=25,tolerance=1e-4, Ncycles=4, f=1.0, k=1.0, mass = 1.0, St = 0.3,U_init = 1.0  )
        
    for i in 1:n_loops
       
        Us = find_Us(U_init = U_init, Ncycles = Ncycles,mass=mass,f=f,k=k, St=St)
        peaks = get_peaks(Us[1,:], flow.N, Ncycles)
        convergence = diff(peaks)[end]
        if abs(convergence) < tolerance
            return U_init
        end
        U_init = convergence < 0 ?
         minimum(Us[1,flow.N*Ncycles-1:end]) :
         maximum(Us[1,flow.N*Ncycles-1:end])
        @show (f,k,St, mass), i, U_init, convergence
    end
    # return a negative value for non-converged cases
    return -U_init
end

begin
    T = Float32
    N = 3
    fs =LinRange{T}(0.4, 4, N)
    ks = LinRange{T}(0.35, 2.0, N)
    Sts = LinRange{T}(0.0125, 0.4, N)
    masses = [0.5,1.0, 2.0].|> T
    Uinits = zeros(N,N,N,N)
    for (i,f) in enumerate(fs)
        for (j,k) in enumerate(ks)
            for (l,mass) in enumerate(masses)
                for (n,St) in enumerate(Sts)
                    Uinits[i,j,l,n] = us_relax(;f=f,k=k,mass=mass, St=St, U_init=mass)
                end
                
            end
        end
    end
end

#count of Converged
length(Uinits) - sum(Uinits .< 0)

"""
    trapezoid(y, a, b, n)

Apply the trapezoid integration formula for integrand `f` over
interval [`a`,`b`], broken up into `n` equal pieces. Returns
the estimate, a vector of nodes, and a vector of integrand values at the
nodes.
"""
function trapezoid(y, a, b, n)
    h = (b - a) / n
    t = range(a, b, length = n + 1)
    # y = f.(t)
    T = h * (sum(y[2:n]) + 0.5 * (y[1] + y[n + 1]))
    return T, t, y
end


begin
    #scripting to visualize blowups 
    f = 0.4 
    k = 0.35
    St = Sts[2]
    mass = 0.5
    U_init = 1.0
    upm = deepcopy(defaultDict)
    upm[:Nt] = 64
    upm[:N] = 64
    upm[:Ncycles] = 2
    upm[:Uinf] = 1.0
    upm[:kine] = :make_ang
    upm[:pivot] = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick] = 0.12
    upm[:f] = f
    upm[:k] = k
    a0 = St / upm[:f]
    upm[:motion_parameters] = [a0]

    foil, flow = init_params(; upm...)
    
    turns = zeros(flow.Ncycles * flow.N)
    wake = Wake(foil)
    Us = zeros(2, flow.Ncycles * flow.N)
    U = _propel(foil, flow; U = [0.0, 0.0], mass = mass, turnto = turns[1])
    #data containers
    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)

    anim = Animation()
    # f = plot()
    for i in 1:(flow.Ncycles * flow.N)
        if flow.n != 1
            move_wake!(wake, flow)
            release_vortex!(wake, foil)
        end
        if i == 1
            U = _propel(foil, flow; U = [0.0, 0.0], mass = mass)
            # Cast this to  2d vel
            U = [U_init, 0.0]
        else
            U = _propel(foil,
                flow;
                forces = coeffs[:, i - 1],
                U = U,
                mass = mass,
                self_prop = true,)        
        end
       
        Us[:, i] = U
        U_bx = U[1]
        A, rhs, edge_body = make_infs(foil)
        A[getindex.(A .== diag(A))] .= 0.5
        setσ!(foil, flow)
        
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        foil.σs -= normal_wake_ind[:]

        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs * foil.σs - buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        phi = get_phi(foil, wake)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)

        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
        coeffs[:, i] = get_performance(foil, flow, p)
        ps[:, i] = p
        f = plot_current(foil, wake)
        plot!(f, ylims = (-1, 1))
        plot!(title = "$(U)")
        frame(anim, f)
        
    end

    gif(anim, "./images/blowups.gif", fps = 10)
end

begin
    #scripting
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
    h0 = 0.05
    θ0 = deg2rad(0)
    upm[:motion_parameters] = [h0, θ0]

    foil, flow = init_params(; upm...)
    foil.θ = π / 2 * 0.0
    U = _propel(foil, flow)
    wake = Wake(foil)

    Us = zeros(2, flow.Ncycles * flow.N)

    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)
    turnrad = π * flow.Δt * 0.0
    trs = zeros(flow.N * flow.Ncycles)
    trs[(flow.N):(2 * flow.N - 32)] .= turnrad
    anim = Animation()
    # f = plot()
    for i in 1:(flow.Ncycles * flow.N)
        if i == 3
            wake.Γ = [wake.Γ[1]]
            wake.xy = hcat(wake.xy[:, 1])
            wake.uv = hcat(wake.uv[:, 1])
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
            U = self_propell(foil,
                flow;
                forces = coeffs[:, i - 1],
                U = [0.0, 0.0],
                mass = 1e-5,
                turn = trs[i],)
            # U = (foil)(flow)
        end
        Us[:, i] = U
        A, rhs, edge_body = make_infs(foil)
        A[getindex.(A .== diag(A))] .= 0.5
        setσ!(foil, flow)
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs * foil.σs - buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        phi = get_phi(foil, wake)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)

        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
        coeffs[:, i] = get_performance(foil, flow, p)
        ps[:, i] = p
        f = plot_current(foil, wake)
        plot!(f, ylims = (-1, 1))
        plot!(title = "$(U)")
        frame(anim, f)
        # plot!(f, foil.foil[1,:],foil.foil[2,:], label="", aspect_ratio=:equal)
    end
    # f
    gif(anim, "./images/handp.gif", fps = 10)
end
