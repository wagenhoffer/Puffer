


# Define the convolutional neural network with embedded autoencoder
unet =
    Chain(enc=Chain(
    Conv((4,2), C=>C*2, tanh),
    MeanPool((2,1)),  
    Conv((4,1), C*2=>C, tanh),
    MeanPool((2,1)),
    ConvTranspose((1,2), C=>C,tanh),
    dec=Chain(
    Upsample((2,1)),
    ConvTranspose((5,1), C=>C, tanh),
    Upsample((2,1)),
    ConvTranspose((5,1), C=>C, tanh))
) |> mp.dev
unet(input) |>size
unet.layers[:enc](input) |>size
ustate = Flux.setup(Adam(0.01), unet)
losses = []
for epoch = 1:1
    for (x,_,_,_) in dataloader        
        x = x |> mp.dev        
        ls = 0.0
        grads = Flux.gradient(unet) do m
            # Evaluate model and loss inside gradient context:                
            ls = errorL2(m(x), x)                          
        end
        Flux.update!(ustate, unet, grads[1])
        push!(losses, ls)  # logging, outside gradient context
    end
    if epoch % 1 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end

plot(losses, yscale=:log10)
begin
    #inputs has channels of [position' row.normals' row.wake_ind_vel' row.panel_vel' ] 
    samp = rand(1:size(input,4))
    which = input[:,:,:,samp]
    a = plot(which[:,1,1],which[:,2,1], aspect_ratio=:equal, label="input")
    quiver!(a,which[:,1,1],which[:,2,1], quiver=(which[:,1,2],which[:,2,2]), aspect_ratio=:equal, label="normals input")

    recon = unet(input[:,:,:,samp:samp])
    b = plot(recon[:,1,1],recon[:,2,1], aspect_ratio=:equal, label="recon")
    quiver!(b,recon[:,1,1],recon[:,2,1], quiver=(recon[:,1,2],recon[:,2,2]), aspect_ratio=:equal, label="normals recon")
    # plot(a,b, layout = (2,1), size = (800,800))
     
    unet.layers[:enc](input) 
    #latent space representation
    latent = unet.layers[:enc](input[:,:,:,samp:samp])
    plot(latent[:,1,1],latent[:,2,1], aspect_ratio=:equal, label="latent")
    c = quiver!(latent[:,1,1],latent[:,2,1], quiver=(latent[:,1,2],latent[:,2,2]), aspect_ratio=:equal, label="normals latent")
    plot(a,c,b, layout = (1,3), size = (900,300))
end


 # Define parameter ranges
 T = Float32
 reduced_freq_values = [0.5, 1.0, 2.0] .|>T
 k_values = [0.5, 1.0, 2.0] .|>T
 a0 = 0.1
 δxs = LinRange{T}(1.25, 2.0, 4)
 δys = LinRange{T}(0.25,  2.0, 8)
 ψs = LinRange{T}(0, pi, 5)


num_foils = 4
reduced_freq = reduced_freq_values[2]
k = k_values[2]
δx = δxs[1]
δy = δys[1]
ψi = ψs[2]                    
# counter +=1
# Set motion parameters
starting_positions = [0.0  δy; δx 0.0; 0.0 -δy; -δx 0.0]'
phases = [0, ψi, 0, ψi].|>mod2pi
fs = [reduced_freq for _ in 1:num_foils]
ks = [k for _ in 1:num_foils]
motion_parameters = [a0 for _ in 1:num_foils]

foils, flow = create_foils(num_foils, starting_positions, :make_wave;
    motion_parameters=motion_parameters, ψ=phases, Ncycles = 6,
    k= ks,  Nt = 100, f = fs);

wake = Wake(foils)                    

# Perform simulations and save results
totalN = sum(foil.N for foil in foils)
steps = flow.N*flow.Ncycles


coeffs = zeros(length(foils), 4, steps)
μs = zeros(totalN)
phis = zeros(totalN)
ps = zeros(totalN)
time_increment!(flow, foils, wake)
begin
    movie = @animate for i in flow.n:flow.N*flow.Ncycles
        @show i
        old_mus, old_phis = zeros(3, totalN), zeros(3, totalN)
        rhs = time_increment!(flow, foils, wake; mask = [false, true, false, false])


        
        for (j, foil) in enumerate(foils)        
            phi = get_phi(foil, wake)
            phis[((j - 1) * foil.N + 1):(j * foil.N)] = phi
            p = panel_pressure(foil,
                flow,
                old_mus[:, ((j - 1) * foil.N + 1):(j * foil.N)],
                old_phis[:,((j - 1) * foil.N + 1):(j * foil.N)],
                phi)
            ps[((j - 1) * foil.N + 1):(j * foil.N)] = p
            μs[((j - 1) * foil.N + 1):(j * foil.N)] = foil.μs
            coeffs[j, :, i ] .= get_performance(foil, flow, p)
        end
        # old_mus = [μs'; old_mus[1:2, :]]
        # old_phis = [phis'; old_phis[1:2, :]]
        plot(foils, wake)
    end
    gif(movie, "newMulti.gif", fps = 30)
end           
a = plot()
for i=1:4
plot!(a, coeffs[i,4,flow.N:end], label=i)
end
a