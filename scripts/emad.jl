using Puffer
using Plots

"""
I have published a paper that covers a lot of the material implemented here. I haqve emailed that to you. 

Skip the acoustics. 

It talks about the traveling wave form, anguilliform is already implemented (the other is a few minutes to set up).
There are three big non-dimensional numbers we care about variation of to make a surrogate of a "physical" swimmer. 
The Strouhal number -> St = f*a0/U  and reduced frequency f* = fc/U and ND wavenumber k* = kc 
here f is frequency, a0 is amplitude of motion, c is chord or length of the swimmer, k is the wavenumber of the wave 
traveling down the chord of the swimmer (think λ if that helps) and U is the freestream velocity. 

In my paper, I used the ranges of 0.25 <= f* <= 4 and 0.025 <= St <= 0.4 and 0.35 <= k* <= 2. You only need to adjust the values of 
frequency and freestream to obtain the first two ranges and the third if an independent variable to adjust. The chord should not change from 1 meter (it plots nice) and the 
a0 value is a physical measurement approximation. 

They are you input parameters to start, you need to figure out how refined you need to go in order for the 
approximation to work. We will talk about validation in the future. 

I have placed a copy of the time_increment! function here with comments to explain what is a function of the inputs
and what is a solution space. 

function time_increment!(flow::FlowParams, foil::Foil, wake::Wake)
    if flow.n != 1
        move_wake!(wake, flow)   
        release_vortex!(wake, foil)
    end    
    (foil)(flow) #does the kinematics of the foil and sets up the next time step to be solved 
                 # this includes new panel positions, and the normals and tangents of the foil (input space)
    
    A, rhs, edge_body = make_infs(foil)
    #A <- large dense influence matrix (input space)
    #rhs <- velocity from the body onto itself and from the vortex wake onto the panels (input space)
    #edgebody <- edge panel influence onto the panels (input space)
    setσ!(foil, flow)   # velocity influences from motion of foil and freestream (input)
    foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow) 
    # wake induced velocity (input)
    normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)' #linear alg
    foil.σs -= normal_wake_ind[:] #linear alg
    buff = edge_body * foil.μ_edge[1] #linear alg
    foil.μs = A \ (-rhs*foil.σs-buff)[:]
    # above we solve the system, this effectively tells us what vortices can be used to represent the foil
    # the μs are the first (and possibly most important) of the output space variables
    #the next two lines are taking the dynamic info we just obtained to make a static snapshot of the flow 
    set_edge_strength!(foil)
    cancel_buffer_Γ!(wake, foil)
    #the next two funcs find what velocity is induced on the vortex particles in the wake (to define how they move)
    body_to_wake!(wake, foil, flow)
    wake_self_vel!(wake, flow)    
    nothing
end


"""

ang = deepcopy(defaultDict)
ang[:N] = 64      #number of panels (elements) to discretize the foil
ang[:Nt] = 64     #number of timesteps per period of motion
ang[:Ncycles] = 3 #number of periods to simulate
ang[:f] = 1.0      #frequency of wave motion 
ang[:Uinf] = 1.0    #free stream velocity 
ang[:kine] = :make_ang
a0 = 0.1 # how much the leading edge heaves up and down wrt the chord(length) of the swimmer
ang[:motion_parameters] = a0

begin
    foil, flow = init_params(; ang...)
    k = foil.f * foil.chord / flow.Uinf
    @show k
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        phi = get_phi(foil, wake)    # (output space) <-probably not that important                            
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)
        # (output space) <- p is a function of μ and we should be able to recreate this     
        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
        coeffs[:, i] = get_performance(foil, flow, p)
        # the coefficients of PERFROMANCE are important, but are just a scaling of P
        # if we can recreate p correctly, this will be easy to get also (not important at first)
        ps[:, i] = p # storage container of the output, nice!
    end
    t = range(0, stop = flow.Ncycles * flow.N * flow.Δt, length = flow.Ncycles * flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs[1, start:end], label = "Force", lw = 3, marker = :circle)
    b = plot(t[start:end], coeffs[2, start:end], label = "Lift", lw = 3, marker = :circle)
    c = plot(t[start:end], coeffs[3, start:end], label = "Thrust", lw = 3, marker = :circle)
    d = plot(t[start:end], coeffs[4, start:end], label = "Power", lw = 3, marker = :circle)
    plot(a, b, c, d, layout = (2, 2), legend = :topleft, size = (800, 800))
end

begin
    # watch a video of the motion, does it blow up? if so, what went wrong? 
    foil, flow = init_params(; ang...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:(flow.Ncycles * flow.N * 1.75)
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0,
            maximum(foil.foil[1, :]) + foil.chord * 2)
        win = nothing
        f = plot_current(foil, wake; window = win)
        plot!(f, ylims = (-1, 1))
    end
    gif(movie, "handp.gif", fps = 10)
end
