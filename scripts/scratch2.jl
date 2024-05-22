
using Puffer
motions 
# build out a fake sim and copy the panel velocities for the inputs of motions
for motion in motions[192:192]
    f,a = out[(0.17453292f0, 0.25f0, 0.4f0)]
    test = deepcopy(defaultDict)
    test[:N] = 50
    test[:Nt] = 100
    test[:Ncycles] = 5
    test[:f] = f
    test[:Uinf] = 1
    test[:kine] = :make_heave_pitch
    
    test[:ψ] = -pi/2
    test[:motion_parameters] = [0.05f0, 0.0f0]

    
    begin
        foil, flow = init_params(; test...)
        wake = Wake(foil)
        (foil)(flow)
        vels = []
        ### EXAMPLE OF AN ANIMATION LOOP
        movie = @animate for i in 1:(flow.Ncycles * flow.N)
            time_increment!(flow, foil, wake)
            # Nice steady window for plotting
            push!(vels, deepcopy(foil.panel_vel))
            plot(foil, wake)            
        end
        gif(movie, "./images/handp.gif", fps = 30)
        # plot(foil, wake)
    end
end