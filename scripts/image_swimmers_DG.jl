###TODO: Hacked scripting for multipleSwimmers###
# for traveling waves
# data_files  = ["multipleSwimmers_images_data.jls"]#,   "multipleSwimmers_inline_data.jls"] #, "diamondSwimmers_13_42_phasediff_data.jls"]
# coeff_files = ["multipleSwimmers_images_coeffs.jls"]#, "multipleSwimmers_inline_coeffs.jls"]#, "diamondSwimmers_13_42_phasediff_coeffs.jls"]

data_files = [ "hnp_Swimmers_images_data.jls"] #to be coupled with a0hnp_    
coeff_files =  ["hnp_Swimmers_images_coeffs.jls"] #to be coupled with a0hnp_    
data = [deserialize(joinpath("data", data_file))  for data_file in data_files]
data = vcat(data...)
coeffs = vcat([deserialize(joinpath("data", coeff_file))  for coeff_file in coeff_files]...)
RHS = hcat(data[:, :RHS]...).|>Float32
μs  = hcat(data[:, :μs]...).|>Float32
P   = vcat(data[:, :pressure]...)'.|>Float32
N = 64
nbodies = (size(RHS,1)÷N)
cnt = nbodies*size(RHS, 2)

RHS = reshape(RHS, (N, cnt))
μs = reshape(μs, (N, cnt))
if size(P,2) != cnt
    P = reshape(P, (N, cnt))
end

pad = 2
inputs = zeros(Float32, N + pad*2, 2, 4, cnt);

i = 1
for row in eachrow(data)  
    for j = 0:nbodies-1
        grab = j*2+1:j*2+2
        position        = row.position[grab,:] 
        # position[1,:] .-= minimum(position[1,:] , dims=1) 

        x_in = [position' row.normals[grab,:]' row.wake_ind_vel[grab,:]' row.panel_vel[grab,:]' ]    
        x_in = vcat(x_in[end-pad+1:end,:,:,:],x_in,x_in[1:pad,:,:,:])
        x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
        inputs[:,:,:,(i-1)*nbodies + j+1] = x_in
    end
    i += 1
end

perfs = zeros(4, cnt);
ns = coeffs[1,:coeffs]|>size|>last
i = 1
motions = []
if "Strouhal" in names(coeffs)
    for r in eachrow(coeffs)        
        perfs[:,ns*nbodies*(i-1)+1:ns*nbodies*i]  =  hcat([r.coeffs[i,:,:] for i=1:nbodies]...) #r.coeffs
        f,a = fna[(r.θ, r.h, r.Strouhal)]
        push!(motions, (:make_heave_pitch, f, r.h, r.θ , r.Strouhal))
        i += 1
    end
else
    for r in eachrow(coeffs)    
        perfs[:,ns*nbodies*(i-1)+1:ns*nbodies*i] = hcat([r.coeffs[i,:,:] for i=1:nbodies]...)
        
        push!(motions, (:make_ang,r.reduced_freq,r.k))        
        i += 1        
    end
end

n_swimmers = 2
time_steps_sim = 500
dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs, P=P), batchsize=4096, shuffle=true)    
inputs|>size
toremove = unique(div.(findall(x -> any(abs.(x) .> 100), eachcol(P)), time_steps_sim))
starts = toremove * time_steps_sim .+ 1
stops = (toremove .+ 1) .* time_steps_sim
mask = trues(size(RHS,2))

for (start,stop) in zip(starts, stops)
    mask[start:stop] .= false
end
RHS_filtered = RHS[:, mask]

μs_filtered = μs[:, mask]
inputs_filtered = inputs[:,:,:,mask]
perfs_filtered = perfs[:, mask]
P_filtered = P[:, mask]
dataloader = DataLoader((inputs=inputs_filtered, RHS=RHS_filtered, μs=μs_filtered, perfs=perfs_filtered, P=P_filtered), batchsize=4096, shuffle=true)





##LET's drive the inputs down to a small latent space (8?)
# then see if we can train a simple Interaction kernel to augment the L matrix
inputss = inputs_filtered[3:end-2,1:2,:,:]
w,H,C,Samps = size(inputss[:,:,1:2,:])

cat([inputss[:,:,1:2,i*2+1:i*2+2] for i = 0:2]...,dims=5)
ims = zeros(Float32, w,H,C,n_swimmers,Samps÷2)
for i = 0:Samps÷2-1
    ims[:,:,:,:,i+1] .= inputss[:,:,1:2,i*2+1:i*2+2]
end
datar = DataLoader((ims=ims,), batchsize=4096, shuffle=true)

# convenc = Chain(Conv((4,2), C=>C, Flux.tanh, pad=SamePad()),
#                 Conv((4,2), C=>1, Flux.tanh, pad=SamePad()),                                
#                 Flux.flatten,
#                 Dense(w*H=>w,Flux.tanh), 
#                 Dense(w=>w))|>gpu
# convdec = Chain(Dense(w=>w, Flux.tanh),
#                 Dense(w=>w*H, Flux.tanh),
#                 x->reshape(x, (w,H,1,size(x,2))),
#                 ConvTranspose((4,2), 1=>C,Flux.tanh, pad=SamePad()),
#                 ConvTranspose((4,2), C=>C, pad=SamePad())) |>gpu

convenc = Chain(Flux.flatten, 
                Dense(w*H*C, w*H,Flux.tanh),
                Dense(w*H, mp.layers[end],Flux.tanh),
                Dense(mp.layers[end],mp.layers[end]))|>gpu                
convdec = Chain(Dense(mp.layers[end],mp.layers[end], Flux.tanh),
                Dense(mp.layers[end], w*H, Flux.tanh),
                Dense(w*H, w*H*C),
                x->reshape(x, (w,H,C,size(x,2)))) |>gpu

                
out = convenc(first(datar)[:ims][:,:,:,1,:]|>gpu)             
# squish = SkipConnection(Chain(convenc, convdec),.+)
squish = Chain(convenc, convdec)
squish(first(datar)[:ims][:,:,:,1,:]|>gpu)# |>size
losses = []
sqstate = Flux.setup(Adam(0.001), squish)
for epoch = 1:1_000
    for (x,) in datar        
        x = x |> mp.dev        
        ls = 0.0
        grads = Flux.gradient(squish) do m        
            
            y = [m(x[:,:,:,i,:]) for i = 1:n_swimmers]
            yl =[m[1](x[:,:,:,i,:]) for i = 1:n_swimmers]
            ls += sum([errorL2(y[i][:,:,1,:], x[:,:,1,i,:])*10.0 for i = 1:n_swimmers])   
            ls += sum([errorL2(y[i][:,:,2,:], x[:,:,2,i,:]) for i = 1:n_swimmers])              
            # ls += errorL2(yl[1][3:end,:],yl[2][3:end,:])
        end
        Flux.update!(sqstate, squish, grads[1])
        push!(losses, ls)  # logging, outside gradient context        
    end
    if epoch % 10 == 0
        println("Epoch $epoch, Loss: $(mean(losses))")
    end
end

begin
    grab = rand(1:size(inputss,4)÷2)*2
    x = inputss[:,:,1:2,grab:grab+1] 
    xhat = [squish(x[:,:,:,i:i]|> gpu)|>cpu for i = 1:n_swimmers]
    a = plot()
    b = plot()
    # @show errorL2(xhat[:,:,1], x[:,:,1]),   errorL2(xhat[:,:,2], x[:,:,2])
    for i = 1:n_swimmers
        plot!(a, x[:,1,1,i], x[:,2,1,i], label="Original")
        quiver!(a, x[:,1,1,i], x[:,2,1,i], quiver= (x[:,1,2,i]/10, x[:,2,2,i]/10), label="Original")
        plot!(b, xhat[i][:,1,1], xhat[i][:,2,1], label="Reconstructed")    
        quiver!(b, xhat[i][:,1,1,1], xhat[i][:,2,1,1], quiver=(xhat[i][:,1,2,1]/10, xhat[i][:,2,2,1]/10), label="Reconstructed")
    end
    c = plot(a,b)
    lhat = [squish[1](x[:,:,:,i:i]|> gpu)|>cpu for i = 1:n_swimmers]
    cc = plot(lhat)
    plot(c, cc,layout=(2,1))
end



##ASSUME that all other things have been AE'd etc by this point
## the data needs to be stacked (pairs of swimmers here) for training

# dataloader = DataLoader((inputs=inputs_filtered, RHS=RHS_filtered, μs=μs_filtered, perfs=perfs_filtered, P=P_filtered, ξs = ξs), batchsize=4096, shuffle=false)

ξs = squish[1](inputss[:,:,1:2,:]|>gpu)|>cpu
ξs = vcat([ξs[:,n:n_swimmers:end] for n = 1:n_swimmers]...)
νs = μAE[:encoder](dataloader.data.μs|>mp.dev)
νs = vcat([νs[:,n:n_swimmers:end] for n = 1:n_swimmers]...)
βs = B_DNN.layers[1:2](dataloader.data[:inputs]|>mp.dev)
# βs = B_DNN.layers[:enc](dataloader.data.RHS|>mp.dev)
βs = vcat([βs[:,n:n_swimmers:end] for n = 1:n_swimmers]...)
μs = vcat([dataloader.data.μs[:,n:n_swimmers:end] for n = 1:n_swimmers]...)
Bs = vcat([dataloader.data.RHS[:,n:n_swimmers:end] for n = 1:n_swimmers]...)
latentsdata = DataLoader((νs=νs, βs=βs, μs=μs, Bs=Bs, ξs = ξs),
     batchsize=mp.batchsize, shuffle=true)


# L = rand(Float32, mp.layers[end],mp.layers[end])|>mp.dev
# Lstate = Flux.setup(Adam(mp.η), L)|>mp.dev
# L, Lstate, latentsdata   

here = Chain(Dense(mp.layers[end],mp.layers[end],Flux.tanh),      
             Dense(mp.layers[end],mp.layers[end],Flux.tanh),
             Dense(mp.layers[end],mp.layers[end]))|>mp.dev
there = Chain(Dense(mp.layers[end],mp.layers[end],Flux.tanh),
            Dense(mp.layers[end],mp.layers[end],Flux.tanh),
            Dense(mp.layers[end],mp.layers[end]))|>mp.dev     
# everywhere = Dense(mp.layers[end]*2,mp.layers[end])|>mp.dev
# Koupler = Parallel((x,y)->everywhere(vcat(x,y)), here, there)|>gpu    

Koupler = Parallel(.*, here, there)|>gpu    

# Koupler = Chain(Dense(mp.layers[end]*2,mp.layers[end]*2,Flux.tanh),
#                 Dense(mp.layers[end]*2,mp.layers[end],Flux.tanh),
#                 Dense(mp.layers[end],mp.layers[end],Flux.tanh),
#                 Dense(mp.layers[end],mp.layers[end]))|>gpu
# K = zeros(Float32, n_swimmers, mp.layers[end], size(B,2))|>mp.dev
# for n = 1:n_swimmers
#     for m = 1:n_swimmers                                
#             if m != n
#                 K[n,:,:] += Koupler((ξ[(n-1)*mp.layers[end]+1:n*mp.layers[end],:],
#                                      ξ[(m-1)*mp.layers[end]+1:m*mp.layers[end],:]))                                    
#             end                
#     end
# end
# K = vcat([K[i,:,:] for i = 1:n_swimmers]...)

# K_parts = [zeros(Float32, mp.layers[end], size(B, 2)) |> mp.dev for _ in 1:n_swimmers]

# vcat([sum(
#         Koupler((ξ[(n-1)*mp.layers[end]+1:n*mp.layers[end],:],
#                  ξ[(m-1)*mp.layers[end]+1:m*mp.layers[end],:]))
#         for m = 1:n_swimmers if m != n
#     ) for n = 1:n_swimmers]... )


# return vcat(K_parts...)

# Koupler((ξs[1:8,1:10],ξs[9:16,1:10]).|>gpu)            
# Koupler((ξs[9:16,1:10],ξs[1:8,1:10]).|>gpu)            


G = inv(L|>cpu)
G_matrix = [G zeros(size(G)); zeros(size(G)) G]|>mp.dev
losses = []
kstate = Flux.setup(Adam(0.001), Koupler)
for epoch = 1:5_000
    for (ν, β, μ, B, ξ) in latentsdata     
        ν = ν |> mp.dev
        β = β |> mp.dev
        μ = μ |> mp.dev
        B = B |> mp.dev
        ξ = ξ |> mp.dev
        losss = 0.0
        grads = Flux.gradient(Koupler) do m        
            νsingle = G_matrix*β  
            K = vcat([sum(
                        m((ξ[(i-1)*mp.layers[end]+1:i*mp.layers[end],:],
                                 ξ[(j-1)*mp.layers[end]+1:j*mp.layers[end],:]))
                        for i = 1:n_swimmers if j != i
                    ) for j = 1:n_swimmers]... )
            
                
            νmany =  νsingle + K.*circshift(ν,(mp.layers[end],0))
            losss += errorL2(νmany, ν)
            # dec = vcat([μAE[:decoder](νmany[(i-1)*mp.layers[end]+1:i*mp.layers[end],:]) for i = 1:n_swimmers]...)
            # losss += errorL2(dec, μ)
        end
        Flux.update!(kstate, Koupler, grads[1])
        push!(losses, losss)  # logging, outside gradient context        
    end
    if epoch % 10 == 0
        println("Epoch $epoch, Loss: $(mean(losses))")
    end
end
begin 
    Gs = G_matrix|>cpu
    cs = [:red, :blue]
    wh = rand(1:size(latentsdata.data[1],2)) 
    νs = latentsdata.data.νs[:,wh:wh]|>cpu
    ξ = latentsdata.data.ξs[:,wh:wh]|>gpu
    K = vcat([sum(
        Koupler((ξ[(i-1)*mp.layers[end]+1:i*mp.layers[end],:],
                 ξ[(j-1)*mp.layers[end]+1:j*mp.layers[end],:]))
        for i = 1:n_swimmers if j != i
    ) for j = 1:n_swimmers]... )|>cpu
    BB = zeros(Float32, mp.layers[end]*n_swimmers,1)
    for m = 1:n_swimmers                             
        BB[(m-1)*mp.layers[end]+1:m*mp.layers[end],:] = B_DNN.layers[:enc](latentsdata.data.Bs[(m-1)*mp.layers[1]+1:m*mp.layers[1],wh:wh]|>mp.dev)|>cpu           
    end
    nuk = Gs*BB 
    plot(νs, label="truth")
    plot!(nuk, label="No Kernel")
    nus = Gs*BB + K
    a = plot!(nus, label="approx")

    #decode them
    μs = latentsdata.data.μs[:,wh:wh]|>cpu
    μsHat = zeros(Float32, mp.layers[1]*n_swimmers,1)
    μsK   = zeros(Float32, mp.layers[1]*n_swimmers,1)
    for m = 1:n_swimmers                             
        μsHat[(m-1)*mp.layers[1]+1:m*mp.layers[1],:] = μAE[:decoder](nus[(m-1)*mp.layers[end]+1:m*mp.layers[end],:]|>mp.dev)|>cpu         
        μsK[(m-1)*mp.layers[1]+1:m*mp.layers[1],:] = μAE[:decoder](nuk[(m-1)*mp.layers[end]+1:m*mp.layers[end],:]|>mp.dev)|>cpu         
    end
    plot(μs, label="truth")
    plot!(μsK, label="No Kernel")
    b = plot!(μsHat, label="approx")
    @show errorL2(μsHat, μs), errorL2(nus, νs)
    plot(a,b,layout=(2,1),size=(800,600))
end


(ν, β, μ, B, ξ) =  first(latentsdata)
(ν, β, μ, B, ξ) = map(gpu, (ν, β, μ, B, ξ))
K = vcat([sum(
            Koupler((ξ[(i-1)*mp.layers[end]+1:i*mp.layers[end],:],
                        ξ[(j-1)*mp.layers[end]+1:j*mp.layers[end],:]))
            for i = 1:n_swimmers if j != i
        ) for j = 1:n_swimmers]... )

νsingle = G_matrix  *β
νmany =  νsingle + K.*νsingle
νmany =  (K.+ 1) .*νsingle

losss += errorL2(νmany, ν)
dec = vcat([μAE[:decoder](νmany[(i-1)*mp.layers[end]+1:i*mp.layers[end],:]) for i = 1:n_swimmers]...)
losss += errorL2(dec, μ)

K = vcat([sum(
    m((ξ[(i-1)*mp.layers[end]+1:i*mp.layers[end],:],
             ξ[(j-1)*mp.layers[end]+1:j*mp.layers[end],:]))
    for i = 1:n_swimmers if j != i
) for j = 1:n_swimmers]... )
νsingle = G_matrix*β      
νmany =  νsingle + K
losss += errorL2(νmany, ν)
dec = vcat([μAE[:decoder](νmany[(i-1)*mp.layers[end]+1:i*mp.layers[end],:]) for i = 1:n_swimmers]...)
losss += errorL2(dec, μ)


params = Flux.params(L, Koupler)
Glosses = []
opt = Adam(0.001)
for epoch = 1:mp.epochs*1
    for (ν, β, μ, B, ξ) in latentsdata
        # (ν, β, μ, B, ξ) =  first(latentsdata)
        ν = ν |> mp.dev
        β = β |> mp.dev
        μ = μ |> mp.dev
        B = B |> mp.dev
        ξ = ξ |> mp.dev
        losss = 0.0
        # Ks = zeros(Float32,n_swimmers, mp.layers[end], size(B,2))
        grads = Flux.gradient(params) do 
              for n = 1:n_swimmers
                K = zeros(Float32, mp.layers[end], size(B,2))|>mp.dev
                for m = 1:n_swimmers                                
                    if m != n
                        Ki = Koupler((ξ[(n-1)*mp.layers[end]+1:n*mp.layers[end],:],
                                      ξ[(m-1)*mp.layers[end]+1:m*mp.layers[end],:]))                
                        K += Ki
                    end                
                end
                # Ks[n,:,:] .= K
                grab = [(n-1)*mp.layers[end]+1:n*mp.layers[end],:] 
                fg   = [(n-1)*mp.layers[1]+1:n*mp.layers[1],:] #full grab
                # Lν = (L .+ K)*ν[grab...] 
                Lν = L*ν[grab...]
                # lk = (L.+ K)
                # Lν = hcat((eachslice(lk,dims=3).*eachcol(ν[grab...]))...)
                βhat = β + K
                losss += errorL2(Lν, βhat) 
                # los = errorL2(Lν, β[grab...]) 

                losss += errorL2(B_DNN.layers[:dec](Lν - K), B[fg...]) 
                # los += errorL2(B_DNN.layers[:dec](Lν), B[fg...]) 
                # # solve for \nu in L\nu = \beta                
                Gb = L\βhat
                # Gb = (L+K)\β[grab...]
                # Gb = hcat((eachslice(lk,dims=3).\eachcol(β[grab...]))...)
                decμ =  μAE[:decoder](Gb)
                losss += errorL2(decμ, μ[fg...])    
            end
            # los += errorL2(Ks[1,:,:], -Ks[2,:,:])                   
            losss         
        end    
        Flux.update!(opt, params, grads)
        push!(Glosses, losss)  # logging, outside gradient context        
    end
    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(Glosses[end])")
    end
end
Glosses



begin  
    G = inv(L|>cpu)      
    cs = [:red, :blue]
    wh = rand(1:size(latentsdata.data[1],2)) 
    ξ = latentsdata.data.ξs[:,wh:wh]
    Ks = zeros(Float32,n_swimmers,  mp.layers[end])
    for n = 1:n_swimmers
        for m = 1:n_swimmers                                
            if m != n
                @show m,n
                Ks[n,:] += Koupler((ξ[(n-1)*mp.layers[end]+1:n*mp.layers[end],:],
                                    ξ[(m-1)*mp.layers[end]+1:m*mp.layers[end],:]).|>gpu)|>cpu                
            end                
        end
    end
    a = plot()
    b = plot()
    for n = 1:2
        
        grab = (n-1)*mp.layers[end]+1:n*mp.layers[end] 
        fg   = (n-1)*mp.layers[1]+1:n*mp.layers[1] #full grab
        # β = B_DNN.layers[:enc](latentsdata.data.Bs[fg,wh:wh]|>mp.dev)|>cpu 
        β  = latentsdata.data.βs[grab,wh:wh]|>cpu
        recon_ν = G*(β+Ks[n,:,:])
        plot!(a,latentsdata.data.νs[grab,wh]|>cpu, label="latent signal",lw=1,marker=:circle, c=cs[n])
        plot!(a, recon_ν, label="Approx", marker=:star, ms=1,lw=0.5, c=cs[n])
        # title!(a,"ν : error = $(round(errorL2(recon_ν|>cpu, latentsdata.data.νs[grab,wh]|>cpu), digits=4))")
        plot!(b, latentsdata.data.μs[fg,wh], label="μ Truth", c=cs[n])
        plot!(b, μAE.layers.decoder(recon_ν|>mp.dev)|>cpu, label="μ DG ",lw=0, marker=:star, ms=4, c=cs[n])
        # title!(b,"μs : error = $(round(errorL2(μAE.layers.decoder(recon_ν|>mp.dev)|>cpu, latentdata.data.μs[:,wh]|>cpu), digits=4))")
    end
    plot(a,b)
end