include("Deep_Green.jl")
include("hnp_nlsolve_strou.jl")

layers = [[64, 32, 16, 8]]
# layers = [[64,64,64,64]]


##LOADING LOOP##
begin
    prefix = "a0hnp_"
    hp_file = "a0_single_swimmer_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    hp_coeff = "a0_single_swimmer_coeffs_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    # prefix="wave"
    # hp_file = ""
    # hp_coeff = ""

    layer = layers |> first
    mp = ModelParams(layer, 0.001, 500, 4096, errorL2, gpu)
    layerstr = join(["$(layer)->" for layer in mp.layers])[1:end-2]

    bAE, μAE, convNN, perfNN = build_networks(mp)
    μstate, bstate, convNNstate, perfNNstate = build_states(mp, bAE, μAE, convNN, perfNN)
    B_DNN = build_BDNN(convNN, bAE, mp)
    dataloader,motions = build_dataloaders(mp, data_file=hp_file, coeffs_file=hp_coeff)
    # dataloader, motions = build_dataloaders(mp)
    #load previous state
    load_AEs(; μ=prefix * "μAE_L$(layerstr).jld2", bdnn=prefix * "B_DNN_L$(layerstr).jld2")
    L, Lstate, latentdata = build_L_with_data(mp, dataloader, μAE, B_DNN)
    load_L(; l=prefix * "L_L$(layerstr).jld2")
    @show "Train Pressure nets and controllers"
    controller = sampled_motion(motions, dataloader.data.inputs[3:end-2, :, :, :]; num_samps=layer[end], nT=500)
    upconv, upstate, pNN, pNNstate, losses, plosses, pndata = upconverter_pressure_networks(latentdata, controller, dataloader.data.P, dataloader.data.RHS, mp)

end

##TRAINING LOOP##
for layer in layers
    @show layer
    prefix = "a0hnp_"
    hp_file = "a0_single_swimmer_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    hp_coeff = "a0_single_swimmer_coeffs_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    # prefix="wave"
    # hp_file = "a0_single_swimmer_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    # hp_coeff = "a0_single_swimmer_coeffs_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"

    mp = ModelParams(layer, 0.001, 1000, 4096, errorL2, gpu)
    bAE, μAE, convNN, perfNN = build_networks(mp)
    μstate, bstate, convNNstate, perfNNstate = build_states(mp, bAE, μAE, convNN, perfNN)
    dataloader, motions = build_dataloaders(mp; data_file=hp_file, coeffs_file=hp_coeff)
    @show "Train Conv and AEs"
    B_DNN, (convNNlosses, μlosses, blosses) = train_AEs(dataloader, convNN, convNNstate, μAE, μstate, bAE, bstate, mp; ϵ=1e-4)
    @show "Train L"
    L, Lstate, latentdata = build_L_with_data(mp, dataloader, μAE, B_DNN)
    Glosses = train_L(L, Lstate, latentdata, mp, B_DNN, μAE; linearsamps=100)
    @show "Save State and Make Plots $layer"
    save_state(mp; prefix=prefix)
    make_plots_and_save(mp, dataloader, latentdata, convNN, μAE, B_DNN, L, Glosses, μlosses, blosses, convNNlosses; prefix=prefix)
    @show "make a controller for training"
    controller = sampled_motion(motions, dataloader.data.inputs[3:end-2, :, :, :]; num_samps=layer[end], nT=500)
    # P= dataloader.data.P
    # RHS = dataloader.data.RHS
    upconv, upstate, pNN, pNNstate, losses, plosses, pndata = upconverter_pressure_networks(latentdata, controller, dataloader.data.P, dataloader.data.RHS, mp)
    save_up_and_p(mp; prefix=prefix)

    perfs = latentdata.data.perfs
    lift = perfs[2,:] 
    thrust = perfs[3,:] 
    pres  = pNN[5].layers(pNN[1:4](pndata.data.images|>gpu))|>cpu
    Γs = latentdata.data.μs 
    Γs = Γs[1,:] - Γs[end,:]
    P = dataloader.data.P
        
    y = pNN(pndata.data.images|>gpu)|>cpu
    (wl, wt, wg, wp) = find_worst(latentdata, pndata, pNN)
    plot_controller(wh=wl; prefix=prefix, name="worst_lift")
    plot_controller(wh=wt; prefix=prefix, name="worst_thrust")
    plot_controller(wh=wg; prefix=prefix, name="worst_Γ")
    plot_controller(wh=wp; prefix=prefix, name="worst_pressure")
    plot_controller(wh=nothing; prefix=prefix, name="Random")

end

