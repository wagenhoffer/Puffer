module Puffer

include("./FlowParams.jl")
export FlowParams, init_params

include("./Body.jl")
export Foil, make_naca, make_teardrop, make_vandevooren, make_waveform, make_ang
export make_heave_pitch, make_eldredge, no_motion, angle_of_attack, norms
export norms!, move_edge!, set_collocation!, rotation, next_foil_pos
export move_foil!, rotate_about, rotate_about!, create_foils

include("./Wake.jl")
export Wake, move_wake!, wake_self_vel!, body_to_wake!, vortex_to_target, release_vortex!
export cancel_buffer_Γ!, cast_and_pull

include("./Panel.jl")
export source_inf,
    doublet_inf, get_panel_vels, get_panel_vels!, panel_frame, get_circulationsm
export make_infs,
    set_edge_strength!, setσ!, turn_σ!, panel_pressure, edge_to_body, get_mdpts, set_ledge_strength!

include("./Simulation.jl")
export _propel, run_sim, get_performance, time_increment!, solve_n_update!
export get_phi, get_dmudt!, get_dphidt!, get_dt, roll_values!, get_qt

include("./Utils.jl")
export defaultDict, f, plot_coeffs, cycle_averaged, spalarts_prune!, create_foils

include("./Lesp.jl")
export set_ledge!, ledge_inf, get_μ!

include("./Fence.jl")
export make_splines, foilsdf, minsmax, sdf_fence, sdf_fence!

end # module
