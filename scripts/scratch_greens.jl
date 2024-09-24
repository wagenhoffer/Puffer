
begin
    which = rand(1:199)
    ns = 501
    c = plot(latentdata.data.νs[:,ns*which+1:ns*(which+1)]|>Array, st=:heatmap, label="input")
    plot!(c, colorbar=:false)
    title!("Latent")
    plot(latentdata.data.perfs[2,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    a = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[1,:], label="Approx", marker=:circle, ms=1)
    title!("Lift")
    plot(latentdata.data.perfs[3,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    b = plot!(perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[2,:], label="Approx", marker=:circle, ms=1)
    title!("Thrust")
    Γ = latentdata.data.μs[1,ns*which+1:ns*(which+1)] - latentdata.data.μs[end,ns*which+1:ns*(which+1)]
    d = plot(Γ, label="Truth",lw=4)
    plot!(d, perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[3,:], label="Approx", marker=:circle, ms=1)
    title!("Γ strength")
    plot(c,a,b,d, layout = (4,1), size = (1200,800))
end



begin
    which = rand(1:200)
    ns = 501
    xs = 1:ns
    plot(xs, dataloader.data.perfs[2:3,ns*which+1:ns*(which+1)][2,:], label="Truth")
    inp = dataloader.data.μs[:,ns*which+1:ns*(which+1)]
    timed = inp
    b = plot!(1:2:ns, b2perf(timed|>gpu)[2,1:2:ns], label="Approx", marker=:circle, ms=4,lw=1)
    title!("Thrust")
end


# make an embedding of the latent space with x,y,t components
begin 
    # adhoc 
    data_file="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
    coeffs_file= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
    data = deserialize(joinpath("data", data_file))
    coeffs = deserialize(joinpath("data", coeffs_file))
        # load/ prep  training data
        data = deserialize(joinpath("data", data_file))
        coeffs = deserialize(joinpath("data", coeffs_file))
       
        RHS = hcat(data[:, :RHS]...).|>Float32
        μs  = hcat(data[:, :μs]...).|>Float32
        P   = vcat(data[:, :pressure]...).|>Float32
    
        N = mp.layers[1]
        
        pad = 2
        inputs = zeros(Float32, N + pad*2, 2, 4, size(data,1))
        
        for (i,row) in enumerate(eachrow(data))       
            position = row.position .- minimum(row.position, dims=2) 
        
            x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]    
            x_in = vcat(x_in[end-pad+1:end,:,:,:],x_in,x_in[1:pad,:,:,:])
            x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
            inputs[:,:,:,i] = x_in
        end
    
        #force, lift, thrust, power 
        perfs = zeros(4, size(data,1))
        ns = coeffs[1,:coeffs]|>size|>last

        for (i,r) in enumerate(eachrow(coeffs))
            # perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2/r.k)
            # perfs[:,ns*(i-1)+1:ns*i] = perf
            perfs[:,ns*(i-1)+1:ns*i] = r.coeffs

        end
    
        dataloader = DataLoader((inputs=inputs, RHS=RHS, μs=μs, perfs=perfs, P=P'), batchsize=mp.batchsize, shuffle=true)    
end
forces = coeffs[:,:coeffs]
which = rand(1:200)
ns = 501
full_sim = ns*which+1:ns*(which+1)
# latentdata.data.νs[:,full_sim]
clts = zeros(2,ns-1,200)
νembs = zeros(mp.layers[end]+1,ns-1,200)|>cpu

allbetas = B_DNN.layers[1:2](inputs[:,:,:,:]|>mp.dev)|>cpu
allnus = μAE[:encoder](dataloader.data.μs|>mp.dev)|>cpu
approxnus = hcat([G*b for b in eachcol(allbetas)]...)
pos = zeros(Float32, 64,2, (ns -1), 200)
μtrim = zeros(Float32, 64, (ns-1),200)
for i=0:199
    full_sim = ns*i+1:ns*(i+1) - 1
    times = data[full_sim,:t]
    # times = times .- times[1]
    times = times .% times[101]
    times .-= times[1]
    # poss = data[full_sim,:position]
    normals = data[full_sim,:normals]
    nx = map(normal->mean(normal[1,:]), normals)
    ny = map(normal->mean(normal[2,33:end]), normals)
    νs = allnus[:,full_sim]
    clts[:,:,i+1] = forces[i+1][2:3,1:end-1]
    νembs[:,:,i+1] .=  vcat(νs, times')
    pos[:,:,:,i+1] .= inputs[3:end-2,:,1,full_sim]
    μtrim[:,:,i+1] = μs[:,full_sim]
    # νembs[:,:,i+1] .=  νs
end
νembs = reshape(νembs, (mp.layers[end]+1,200*(ns-1 )))
clts = reshape(clts, (2,200*(ns-1)))
pos = reshape(pos, (size(pos)[1:2]...,200*(ns-1)))
μtrim = reshape(μtrim, (size(μtrim)[1]...,200*(ns-1)))
pinndata = DataLoader((νs=νembs.|>Float32, clts=clts.|>Float32, pos = pos.|>Float32, μs = μtrim.|>Float32), batchsize=100, shuffle=false)

test = collect(Iterators.take(pinndata,1000))




nusize = mp.layers[end] #size(allnus,1)
nuext = nusize + 3 #x,y,t1
# inputs ν[8x1],x,y,t ->  ̂ν[2,1], P_ish
# PINN = Chain(Dense(nuext, nuext, tanh),
#              Dense(nuext, nuext, tanh),             
#              Dense(nuext, 3))|>mp.dev
#idea - batch up sequences -> output 3x1 of latent space nu P Γ
timebatch = 2
PINN = Chain(GRU(timebatch => 1),
            x-> permutedims(x),
            Dense(nuext, nuext, tanh),             
            Dense(nuext, 3))|>mp.dev                         
        
ts = 55
trunk = Chain(Conv((4,), 2=>2, Flux.tanh),
              Conv((4,), 2=>2, Flux.tanh),
              Conv((4,), 2=>1, Flux.tanh),
              Flux.flatten,
             Dense(ts, ts, tanh),
             Dense(ts, 2)) |>mp.dev
txy = trunk(pos[:,:,1:2]|>mp.dev)
# pipe out puts 4x1 of latent space nu P x y
# pipe = Parallel(vcat, PINN, trunk)
comp(x,y) = transpose(vcat(x,y))
pipe = PairwiseFusion(comp, trunk, PINN)
pipe(pos[:,:,1:2]|>mp.dev, νembs[:,1:2]|>mp.dev)
pinnstate = Flux.setup(Adam(0.01), pipe)


function set_dir(ndirs, whichdir)
    dir = zeros(Float32, ndirs)
    dir[whichdir] = 1.0
    dir
end
ϵ = eps(Float32)^0.5
dx = set_dir(nuext,1)*ϵ |>mp.dev
dy = set_dir(nuext,2)*ϵ|>mp.dev
dtdir = set_dir(nuext, nuext)|>mp.dev

pinnlosses = []
for epoch = 1:10
    @show epoch
    for (ν,lt,pxy,mus) in pinndata
        nsamps = size(ν,2)
        # ν = [ν[:,i:i+timebatch-1]' for i=1:nsamps-timebatch] |>mp.dev
        ν = ν |> mp.dev
        lift  = lt[1,:] |> mp.dev
        thrst = lt[2,:] |> mp.dev
        pxy = pxy |> mp.dev
        mus = mus |>mp.dev
        Γ = mus[1,:] - mus[end,:]
        Δt = (ν[2][end] - ν[1][end])
        dt = Δt*dtdir |>mp.dev
        global ls = 0.0
        Flux.reset!(pipe)
        
        grads = Flux.gradient(pipe) do m
           # Evaluate model and loss inside gradient context:
        #    sum(Flux.Losses.mse.([model(x)[1] for x ∈ X[2:end]], Y[2:end]))
            # ls = 0
            pinn = m.layers[2]
            out  = m.([(pxy[:,:,i:i+timebatch-1],ν[:, i:i+timebatch-1]) for i=1:nsamps-timebatch+1])
            xy   = hcat([a for (a,b) in out]...)#[:,1:2:end]
            yhat = hcat([b for (a,b) in out]...)
            # me = Flux.mse(y.^2, 0.0)
            lat = reshape(vcat(xy, repeat(ν, inner=(1,2))[:,2:end-1]), (nuext,timebatch,nsamps-1))
            
            # lat = reshape(lat, (size(lat)[1], timebatch, size(lat)[2]÷timebatch))                    
            ddx = hcat([(pinn((lat[:,:,i] .+ dx)') - pinn((lat[:,:,i] .- dx)'))./(2ϵ)    for i=1:nsamps-timebatch+1]...)
            ddy = hcat([(pinn((lat[:,:,i] .+ dy)') - pinn((lat[:,:,i] .- dy)'))./(2ϵ)    for i=1:nsamps-timebatch+1]...)
            ddt = hcat([(pinn((lat[:,:,i] .+ dt)') - pinn((lat[:,:,i] .- dt)'))./(2*Δt)  for i=1:nsamps-timebatch+1]...)
            # # # #approximate the Unsteady Bernoulli's equation
            # # # ∂ν/∂t + ν ∇⋅(ν) + |∇ν|^2 + P_ish = 0
            bern = ddt[1,:] .+ (ddx[1,:].^2 .+ ddy[1,:].^2) .+ yhat[2,:]
            # bern = y[1,:].^2/2.0 + y[2,:]
            be = Flux.mse(bern, 0.0)            
            # # # then forces are equal to 
            # # # 0 = -Pish⋅n⋅dS
            
            ct = errorL2(-xy[1,1:2:end].*yhat[2,:] , thrst[timebatch:end])
            cl = errorL2(-xy[2,1:2:end].*yhat[2,:] , lift[timebatch:end])
            Γe = errorL2(yhat[3,:], Γ[timebatch:end])                
            ls = ct + cl + be + Γe
            
        end
        Flux.update!(pinnstate, pipe, grads[1])
        push!(pinnlosses, ls)  # logging, outside gradient context
    end

    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(pinnlosses[end])")
    end
end


 reshape(vcat(xy, repeat(ν, inner=(1,2))[:,2:end-1]), (11,2,99))

tiledν   = repeat(ν, inner=(1,timebatch))
tiledpxy = repeat(pxy, inner=(1,1,timebatch))
out = vcat(trunk(tiledpxy),tiledν)
tt =  @view out[:,2:end-1]
# tt = reshape(tt, (size(tt)[1], timebatch , size(tt)[2]÷timebatch))
tt = reshape(tt, (timebatch, size(tt)[1], size(tt)[2]÷timebatch))
PINN(tt)

pinnlosses = []
for epoch = 1:10
    @show epoch
    for (ν,lt,pxy,mus) in pinndata
        ν = [ν[:,i:i+timebatch-1]' for i=1:nsamps-timebatch] |>mp.dev
        lift  = lt[1,:] |> mp.dev
        thrst = lt[2,:] |> mp.dev
        pxy = pxy |> mp.dev
        mus = mus |>mp.dev
        Γ = mus[1,:] - mus[end,:]

        Δt = (ν[end,2] - ν[end,1])
        dt = Δt*dtdir
        global ls = 0.0
        Flux.reset!(PINN)
        nsamps = size(ν,2)
        grads = Flux.gradient(PINN) do m
           # Evaluate model and loss inside gradient context:
        #    sum(Flux.Losses.mse.([model(x)[1] for x ∈ X[2:end]], Y[2:end]))
            # ls = 0
            
            out = m.([ν[i] for i=1:nsamps-timebatch])
            # me = Flux.mse(y.^2, 0.0)
            # for i = 2:nsamps
            wind = 2:nsamps
            xy, out = m(pxy[:,:,wind],ν[:,wind])                
            lat = vcat(xy,ν[:,wind])
            # ddt = (m(ν .+ dt) - m(ν))/ϵ
            ddx = (pinn(lat .+ dx) - pinn(lat .- dx))/(2ϵ)
            ddy = (pinn(lat .+ dy) - pinn(lat .- dy))/(2ϵ)
            ddt = (pinn(lat .+ dt) - pinn(lat .- dt))./(Δt)
            # # # #approximate the Unsteady Bernoulli's equation
            # # # ∂ν/∂t + ν ∇⋅(ν) + |∇ν|^2 + P_ish = 0
            bern = ddt[1,:] + sum(ddx[1,:].^2 .+ ddy[1,:].^2, dims=2) + out[2,:]
            # bern = y[1,:].^2/2.0 + y[2,:]
            be = Flux.mse(bern, 0.0)            
            # # # then forces are equal to 
            # # # 0 = -Pish⋅n⋅dS
            ct = errorL2(-xy[1,:].*out[2,:], thrst[wind])
            cl = errorL2(-xy[2,:].*out[2,:], lift[wind])
            Γe = errorL2(out[3,:], Γ[wind])                
            ls = cl + ct + be + Γe
            
        end
        Flux.update!(pinnstate, pipe, grads[1])
        push!(pinnlosses, ls)  # logging, outside gradient context
    end

    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(pinnlosses[end])")
    end
end


# begin
#     which = rand(1:199)
#     ns = 501
#     samp = pinndata.data.νs[:,ns*which+1:ns*(which+1)]
#     x  = samp[9,:]
#     y  = samp[10,:]
#     c = plot(samp|>Array, st=:heatmap, label="input")
#     plot!(c, colorbar=:false)
#     title!("Latent")
#     plot(pinndata.data.clts[1,ns*which+1:ns*(which+1)], label="Truth",lw=4)
#     a = plot!(-x.*PINN(samp)[2,:], label="Approx", marker=:circle, ms=1)
#     title!("Lift")
#     plot(pinndata.data.clts[2,ns*which+1:ns*(which+1)], label="Truth",lw=4)
#     b = plot!(-y.*PINN(samp)[2,:], label="Approx", marker=:circle, ms=1)
#     title!("Thrust")
#     c = plot(-y.*PINN(samp)[2,:], label="y", marker=:circle, ms=1)
#         plot!(c, x.*PINN(samp)[2,:], label="x", marker=:circle, ms=1)

#     # Γ = latentdata.data.μs[1,ns*which+1:ns*(which+1)] - latentdata.data.μs[end,ns*which+1:ns*(which+1)]
#     # d = plot(Γ, label="Truth",lw=4)
#     # plot!(d, perfNN(latentdata.data.νs[:,ns*which+1:ns*(which+1)])[3,:], label="Approx", marker=:circle, ms=1)
#     # title!("Γ strength")
#     plot(a,b,c,  layout = (3,1), size = (1200,800))
# end


begin
    which = rand(1:199)
    ns = 501
    Flux.reset!(pipe)
    slice = ns*which+1:ns*(which+1)-1 #ns*i+1:ns*(i+1) - 1
            # ns*i+1:ns*(i+1) - 1
    samp = pinndata.data.νs[:,slice]|>mp.dev
    spxy = pinndata.data.pos[:,:,slice]|>mp.dev
    Γs = pinndata.data.μs[1,slice] - pinndata.data.μs[end,slice]
    xy, out = pipe(spxy,samp)
    nuh = out[1,:]
    ph  = out[2,:]
    x = xy[1,:]
    y = xy[2,:]
    plot(pinndata.data.clts[1,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    a = plot!(-ph.*y, label="Approx", marker=:circle, ms=1)
    title!("Lift")
    plot(pinndata.data.clts[2,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    b = plot!(-ph.*x, label="Approx", marker=:circle, ms=1)
    title!("Thrust")
    plot(x, label="x")
    c = plot!(y, label="y", lw= 0 , marker=:circle, ms=2)
    plot(Γs, label="Γ Truth",lw=4)
    d = plot!(out[3,:], label="Approx", marker=:circle, ms=1)
    
    plot(a,b,c,d, layout = (4,1), size = (1200,800))

end


otherdata= DataLoader((νs=νembs, clts=clts, pos = pos, μs = μtrim), batchsize=4096, shuffle=true)

opinn = Chain(Dense(nuext,nuext,tanh),
              Dense(nuext,nuext,tanh),
              Dense(nuext,3))|>gpu  
flatter = Chain(Conv((4,), 2=>2, Flux.tanh, pad=SamePad()),                
        Conv((4,), 2=>2, Flux.tanh),
        AdaptiveMeanPool((32,)),
        Conv((4,), 2=>2, Flux.tanh),
        AdaptiveMeanPool((16,)),                        
        x->reshape(x,(size(x,1)*size(x,2), size(x,3))),
        Dense(N÷2=>N÷2, Flux.tanh),
        Dense(N÷2=>N÷2, Flux.tanh),
        Dense(N÷2=>2))  |>mp.dev
flatter(pos[:,:,1:2]|>gpu)  
mlp = PairwiseFusion(vcat, flatter, opinn)
mlp(pos[:,:,1:2]|>gpu, νembs[:,1:2]|>gpu)[2]
ostate = Flux.setup(Adam(0.001), mlp)
            
olosses = []
for epoch = 1:1000
    # @show epoch
    for (ν,lt,pxy,mus) in otherdata
        ν = ν|>mp.dev
        lift  = lt[1,:] |> mp.dev
        thrst = lt[2,:] |> mp.dev
        pxy = pxy |> mp.dev
        mus = mus |>mp.dev
        Γ = mus[1,:] - mus[end,:]


        global los = 0.0        
        nsamps = size(ν,2)

        grads = Flux.gradient(mlp) do m
            # Evaluate model and loss inside gradient context:
            # ls = 0
            
            xy, out = m(pxy,ν)     

            ct = errorL2(-xy[1,:].*out[1,:], thrst)
            cl = errorL2(-xy[2,:].*out[2,:], lift)
            Γe = errorL2(out[3,:], Γ)                
            los = ct + cl +  Γe
            
        end
        Flux.update!(ostate, mlp, grads[1])
        push!(olosses, los)  # logging, outside gradient context
    end

    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(olosses[end])")
    end
end


begin
    which = rand(1:199)
    ns = 500
   
    slice = ns*which+1:ns*(which+1)-1 #ns*i+1:ns*(i+1) - 1
            # ns*i+1:ns*(i+1) - 1
    samp = pinndata.data.νs[:,slice]
    spxy = pinndata.data.pos[:,:,slice]
    Γs = pinndata.data.μs[1,slice] - pinndata.data.μs[end,slice]
    xy, out = mlp(spxy|>gpu,samp|>gpu)

    x = xy[1,:]
    y = xy[2,:]

    @show data[slice[1], [:k,:reduced_freq]]
    plot(pinndata.data.clts[1,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    a = plot!(-xy[2,:].*out[2,:], label="Approx", marker=:circle, ms=1)
    title!("Lift")
    plot(pinndata.data.clts[2,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    b = plot!(-xy[1,:].*out[1,:], label="Approx", marker=:circle, ms=1)
    title!("Thrust")
    plot(x, label="x")
    c = plot!(y, label="y", lw= 0 , marker=:circle, ms=2)
    plot(Γs, label="Γ Truth",lw=4)
    d = plot!(out[3,:], label="Approx", marker=:circle, ms=1)
    
    plot(a,b,c,d, layout = (4,1), size = (1200,800))

end

begin
    # which = rand(1:199)
    # ns = 500
   
    slice = ns*which+1:ns*(which+1)-1 #ns*i+1:ns*(i+1) - 1

    samp = pinndata.data.νs[:,slice]
    spxy = pinndata.data.pos[:,:,slice]
    xy, out = mlp(spxy|>gpu,samp|>gpu)
    nuh = out[1,:]
    ph  = out[2,:]
    xm,ym = maximum([v for v in  eachcol(xy)])
    x = xy[1,:] .- xm
    y = xy[2,:] .- ym
    body = pos[:,:,slice]
    movie = @animate  for i=1:ns-1
        plot(body[:,1,i],body[:,2,i], label="", aspect_ratio=:equal)
        plot!(x,y,label="")
        plot!(x[i:i],y[i:i], ms=10, marker=:circle, label="", aspect_ratio=:equal)
        title!("$(i%100)")
    end
    gif(movie, "test.gif", fps = 30)
    
end




###start with a cnn to try and get the force data and Γ from full data space
C = 4
N = 16
bigM = Chain(Conv((4,2), C=>C, Flux.tanh, pad=SamePad()),                
                Conv((5,1), C=>C, Flux.tanh),
                MeanPool((2,1)),                
                Conv((1,2), C=>1, Flux.tanh), 
                MeanPool((2,1)),
                x->reshape(x,(size(x,1), size(x,4))),
                Dense(N=>N, Flux.tanh),
                Dense(N=>N, Flux.tanh),
                Dense(N=>3))  |>mp.dev

dl = dataloader|>first
bigM(dl.inputs|>mp.dev)

bigMstate = Flux.setup(Adam(0.01), bigM)

bigMlosses = []
for epoch = 1:1_000
    # @show epoch
    for (inputs,_,mus,lt,_) in dataloader
        inputs = inputs |>mp.dev
        lift  = lt[2,:] |> mp.dev
        thrst = lt[3,:] |> mp.dev                
        mus   = mus |>mp.dev
        Γ = mus[1,:] - mus[end,:]       
        los = 0.0                         
        grads = Flux.gradient(bigM) do m            
            y = m(inputs)     
            ct = errorL2(y[1,:], thrst)
            cl = errorL2(y[2,:], lift)
            Γe = errorL2(y[3,:], Γ)                
            los = ct + cl +  Γe            
        end
        Flux.update!(bigMstate, bigM, grads[1])
        push!(bigMlosses, los)  # logging, outside gradient context
    end

    if epoch % 10 == 0
        println("Epoch: $epoch, Loss: $(bigMlosses[end])")
    end
end

begin
    which = rand(1:199)
    ns = 501
   
    slice = ns*which+1:ns*(which+1)-1 #ns*i+1:ns*(i+1) - 1
            # ns*i+1:ns*(i+1) - 1
    @show data[slice[1], [:k,:reduced_freq]]
    # @show data[slice[end], [:k,:reduced_freq]]
    spxy = inputs[:,:,:,slice]
    Γs = μs[1,slice] - μs[end,slice]
    y = bigM(spxy|>mp.dev)
    
    plot(perfs[2,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    a = plot!(y[2,:], label="Approx", marker=:circle, ms=1)
    title!("Lift")
    plot(perfs[3,ns*which+1:ns*(which+1)], label="Truth",lw=4)
    b = plot!(y[1,:], label="Approx", marker=:circle, ms=1)
    title!("Thrust")

    c = plot(y[3,:], label="Γ approx", lw= 0 , marker=:circle, ms=2)
    plot!(Γs, label="Γ Truth",lw=4)
    
    
    plot(a,b,c, layout = (3,1), size = (1200,800))

end


begin
    #the above gave crap, why? 
    #lets explore the data
    n  = data[slice,:normals]
    p  = data[slice,:position]
    pv = data[slice,:panel_vel]
    iv = data[slice,:wake_ind_vel]
    pr = data[slice,:pressure]

    movie = @animate for i = 1:500
        a = quiver(p[i][1,:],p[i][2,:], quiver=(iv[i][1,:],iv[i][2,:]), label="")        
        b = plot(p[i][1,:],pr[i][:], label="")
        plot(a,b, layout = (2,1))
    end
    gih = gif(movie, "test.gif", fps = 30)
end

for i = 1:5
    perflosses = train_perfNN(latentdata, perfNN, perfNNstate, mp)
end

begin

    which = rand(1:29)
    ns = 501
   

    (ν,βs,μs,_,perf) = latentdata.data
    image = permutedims(cat(ν, βs, dims=3), (1,3,2))
    image = reshape(image, (1, size(image)...))
    lift = perf[2,:] 
    thrust = perf[3,:] 
    Γs = μs 
    Γs = Γs[1,:] - Γs[end,:]
        
    y = perfNN(image|>gpu)|>cpu
            # Evaluate model and loss inside gradient context:
            
    lifte   = errorL2(y[1,:], lift) #lift
    thruste = errorL2(y[2,:], thrust)#thrust  
    gammae  = errorL2(y[3,:], Γs)             
    @show (lifte, thruste, gammae)
    le = []
    te = []
    ge = []
    for i = 1:29
        slice = ns*i+1:ns*(i+1)-1 
        push!(le, errorL2(y[1,slice], lift[slice]))
        push!(te, errorL2(y[2,slice], thrust[slice]))
        push!(ge, errorL2(y[3,slice], Γs[slice])   )
    end

    slice = ns*which+1:ns*(which+1)-1 
    # @show k,r = data[slice[1], [:k,:reduced_freq]]
    plot(lift[slice])
    a = plot!(y[1,slice])
    plot(thrust[slice])
    b = plot!(y[2,slice])
    plot(Γs[slice])
    c = plot!(y[3,slice])
    plot(a,b,c, layout = (3,1), size = (1200,800))
    # title!("k: $k, r: $r")

end
    plot(le, label="Lift")
    plot!(te, label="Thrust")
    plot!(ge, label="Γ")
