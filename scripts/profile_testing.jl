include("../src/BemRom.jl")

using ProfileView
using Profile

heave_pitch = deepcopy(defaultDict)
heave_pitch[:N] = 64
heave_pitch[:Nt] = 64
heave_pitch[:Ncycles] = 10
heave_pitch[:f] = 1.
heave_pitch[:Uinf] = 1
heave_pitch[:kine] = :make_heave_pitch
θ0 = deg2rad(5)
h0 = 0.0
heave_pitch[:motion_parameters] = [h0, θ0]
VSCodeServer.@profview run_sim(;heave_pitch...)
@profile run_sim(;heave_pitch...)

@timev run_sim(;heave_pitch...);