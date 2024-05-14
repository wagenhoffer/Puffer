
using Puffer
motions 
# build out a fake sim and copy the panel velocities for the inputs of motions
for motion in motions[192:192]
    test = deepcopy(defaultDict)
    test[:N] = 50
    test[:Nt] = 100
    test[:Ncycles] = 5
    test[:f] = motion[2]
    test[:Uinf] = 1
    test[:kine] = motion[1]
    test[:k] = motion[3]
    test[:ψ] = 0.0
    test[:motion_parameters] = 0.1

    
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