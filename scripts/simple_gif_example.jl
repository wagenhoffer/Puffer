using Plots
using Puffer

"""
Below is an example on how to make a gif of a NACA0012 airfoil to perform a heaving and pitching motion.
The foil pivots about the leading edge. This position can be changed with the pivot argument in the dict. 

``h = h_0 sin(2\pi ft)``
``θ = θ_0 sin(2\pi ft + \psi)``

These kinematics are available in the package for V&V against experimental works, where it is easier to 
    control with rigid body dynamics.
"""

begin
    heave_pitch = deepcopy(defaultDict)
    heave_pitch[:N] = 64
    heave_pitch[:Nt] = 64
    heave_pitch[:Ncycles] = 2
    heave_pitch[:f] = 1.0
    heave_pitch[:Uinf] = 1
    heave_pitch[:kine] = :make_heave_pitch
    θ0 = deg2rad(5)
    h0 = 0.05
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    movie = @animate for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        # win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)
        win = nothing
        f = plot_current(foil, wake; window = win)
        f
    end
    gif(movie, "./images/h_$(h0)_p_$(rad2deg(θ0)).gif", fps = 30)
end

"""
    Anguilliform swimmer gif Below
"""

begin
    ang = deepcopy(defaultDict)
    ang[:N] = 64
    ang[:Nt] = 64
    ang[:Ncycles] = 2
    ang[:f] = 1.0
    ang[:Uinf] = 1
    # The first major difference is the kinematics function to dispatch on
    ang[:kine] = :make_ang
    ang[:k] = 1.0

    foil, flow = init_params(; ang...)
    wake = Wake(foil)
    (foil)(flow)
    movie = @animate for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        # win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)
        win = nothing
        f = plot_current(foil, wake; window = win)
        f
    end
    gif(movie, "./images/ang_f_$(ang[:f])_k_$(ang[:k]).gif", fps = 30)
end
