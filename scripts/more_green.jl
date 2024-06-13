include("Deep_Green.jl")
# layers = [[64,32,16,8]]
layers = [[64,64,64,64]]


##TRAINING LOOP##
for layer in layers        
    @show layer
    # prefix="a0hnp_"
    # hp_file = "a0_single_swimmer_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    # hp_coeff = "a0_single_swimmer_coeffs_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    prefix="wave"
    # hp_file = "a0_single_swimmer_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    # hp_coeff = "a0_single_swimmer_coeffs_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"

    mp = ModelParams(layer,  0.001, 500, 4096, errorL2, gpu)
    bAE, μAE, convNN, perfNN = build_networks(mp)
    μstate, bstate, convNNstate, perfNNstate = build_states(mp, bAE, μAE, convNN, perfNN)
    dataloader,motions = build_dataloaders(mp)#;data_file=hp_file, coeffs_file=hp_coeff)
    @show "Train Conv and AEs"
    B_DNN,(convNNlosses, μlosses, blosses) = train_AEs(dataloader, convNN, convNNstate, μAE, μstate, bAE, bstate, mp; ϵ=1e-4)
    @show "Train L"
    L, Lstate, latentdata = build_L_with_data(mp, dataloader, μAE, B_DNN)
    Glosses = train_L(L, Lstate, latentdata, mp, B_DNN, μAE)    
    @show "Save State and Make Plots $layer"
    save_state(mp;prefix=prefix)
    make_plots_and_save(mp, dataloader, latentdata, convNN, μAE, B_DNN, L,  Glosses, μlosses, blosses, convNNlosses;prefix=prefix)
    @show "make a controller for training"
    controller = sampled_motion(motions, dataloader.data.inputs[3:end-2,:,:,:]; num_samps=layer[end], nT=500);
    # P= dataloader.data.P
    # RHS = dataloader.data.RHS
    upconv, upstate, pNN, pNNstate, losses, plosses, pndata = upconverter_pressure_networks(latentdata, controller, dataloader.data.P,dataloader.data.RHS, mp);
    save_up_and_p(mp;prefix=prefix)
    (wl,wt,wg,wp) = find_worst(latentdata, pndata, pNN)
    plot_controller(wh=wl;prefix=prefix, name="worst_lift")
    plot_controller(wh=wt;prefix=prefix, name="worst_thrust")
    plot_controller(wh=wg;prefix=prefix, name="worst_Γ")
    plot_controller(wh=wp;prefix=prefix, name="worst_pressure")
    plot_controller(wh=nothing;prefix=prefix, name="Random")

end
##LOADING LOOP##
begin
    prefix="a0hnp_"
    hp_file = "a0_single_swimmer_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    hp_coeff = "a0_single_swimmer_coeffs_thetas_0_10_h_0.0_0.25_fs_0.2_0.4_h_p.jls"
    # prefix="wave"
    # hp_file = ""
    # hp_coeff = ""

    layer = layers|>first
    mp = ModelParams(layer,  0.001, 500, 4096, errorL2, gpu)
    layerstr = join(["$(layer)->" for layer in mp.layers])[1:end-2]
    
    bAE, μAE, convNN, perfNN = build_networks(mp)
    μstate, bstate, convNNstate, perfNNstate = build_states(mp, bAE, μAE, convNN, perfNN)
    B_DNN = build_BDNN(convNN, bAE, mp)
    # dataloader,motions = build_dataloaders(mp, data_file=hp_file, coeffs_file=hp_coeff)
    dataloader,motions = build_dataloaders(mp)
    #load previous state
    load_AEs(;μ=prefix*"μAE_L$(layerstr).jld2", bdnn=prefix*"B_DNN_L$(layerstr).jld2")
    L, Lstate, latentdata = build_L_with_data(mp, dataloader, μAE, B_DNN)
    load_L(;l=prefix*"L_L$(layerstr).jld2")
    @show "Train Pressure nets and controllers"
    controller = sampled_motion(motions, dataloader.data.inputs[3:end-2,:,:,:]; num_samps=layer[end], nT=500);
    upconv, upstate, pNN, pNNstate, losses, plosses, pndata = upconverter_pressure_networks(latentdata, controller, dataloader.data.P,dataloader.data.RHS, mp);
    
end


 ###BELOW HERE IS SCRATCH WORK   
    
convNNlosses = []    
kick = false
for epoch = 1:mp.epochs
    for (x, y,_,_,_) in dataloader        
        x = x |> mp.dev
        y = y |> mp.dev
        ls = 0.0
        grads = Flux.gradient(convNN) do m
            # Evaluate model and loss inside gradient context:                
            ls = errorL2(m(x), x)  
            #TODO: increase the weight on the latent space loss
            # the resnet is not learning the latent space well
            latent = errorL2(m.layers[1](x),y)
            ls += latent
        end
        Flux.update!(convNNstate, convNN, grads[1])
        push!(convNNlosses, ls)  # logging, outside gradient context
        if ls < ϵ
            println("Loss is below the threshold. Stopping training at epoch: $epoch")
            kick = true
        end
    end
    if kick == true
        break
    end
    if epoch %  (mp.epochs /10) == 0
        println("Epoch: $epoch, Loss: $(convNNlosses[end])")
    end
end 

which = rand(1:size(dataloader.data[1],4))
which = minss[3]
simnum = which÷500
@show h_p_coeffs[simnum, :]
inval = dataloader.data.inputs[:,:,:,which:which]|>mp.dev
B = dataloader.data.RHS[:,which]
ine = B_DNN.layers[:convenc](inval)
ind = B_DNN.layers[1:3](inval)
bd = B_DNN.layers[2:3](B|>gpu)
plot(B, label = "Truth",lw=4,c=:red)
plot!(ine, label="CN(IN)->B Approx", marker=:star, ms=2,lw=1)
plot!(ind, marker=:circle,lw=0, label="DEC(ENC(CN(IN)))->B Approx")
plot!(bd, marker=:circle,lw=0, label="DEC(ENC(B))->B Approx")
title!("RHS : error = $(errorL2(ine|>cpu, B|>cpu))")



ine = B_DNN.layers[:convenc](inputs|>mp.dev)
ind = B_DNN.layers[1:3](inputs|>gpu)
bd = B_DNN.layers[2:3](RHS|>gpu)
RHSg = RHS|>gpu
errors =[]
for i = 1:count
    push!(errors, [errorL2(ine[:,i], RHSg[:,i]), 
                   errorL2(ind[:,i], RHSg[:,i]), 
                   errorL2(bd[:,i],  RHSg[:,i])])
end
errors = hcat(errors...)
maxes = [argmax(e) for e in eachrow(errors)]
minss = [argmin(e) for e in eachrow(errors)]
c2 = argmax(errors[2,:])
c3 = argmax(errors[3,:])
