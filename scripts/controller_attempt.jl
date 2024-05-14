using Puffer 
using Flux
using Serialization
using DataFrames

data_file="single_swimmer_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
coeffs_file= "single_swimmer_coeffs_ks_0.35_2.0_fs_0.25_4.0_ang_car.jls"
# load/ prep  training data
data = deserialize(joinpath("data", data_file))
coeffs = deserialize(joinpath("data", coeffs_file))

RHS = hcat(data[:, :RHS]...).|>Float32
μs  = hcat(data[:, :μs]...).|>Float32
P   = vcat(data[:, :pressure]...).|>Float32
low = -10000
high = -low
these = low .< (data[:,:reduced_freq] ./ data[:,:k] ) .< high
count = sum(these)



inputs = zeros(Float32, N , 2, 4, count)

i = 1
for row in eachrow(data)
    if low < row.reduced_freq / row.k < high
        position = row.position  
        position[1,:] .-= minimum(position[1,:])
        x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]            
        x_in = reshape(x_in, size(x_in,1), 2, size(x_in, 2) ÷ 2, 1) #shape the data to look like channels
        inputs[:,:,:,i] = x_in
        i += 1
    end
end

#force, lift, thrust, power 
# perfs = zeros(4, size(data,1))
perfs = zeros(4, count)
ns = coeffs[1,:coeffs]|>size|>last
i = 1
motions = []
for r in eachrow(coeffs)
    # perf =  r.reduced_freq < 1 ? r.coeffs : r.coeffs ./ (r.reduced_freq^2/r.k)        
    # perfs[:,ns*(i-1)+1:ns*i] = perf
    if low < r.reduced_freq / r.k < high
        perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
        push!(motions, (r.wave,r.reduced_freq,r.k))
        i += 1
    end
    # perfs[:,ns*(i-1)+1:ns*i] = r.coeffs
end

# dataloader = DataLoader((inputs=inputs, RHS=RHS[:,findall(these)], μs=μs[:,findall(these)], perfs=perfs, P=P[findall(these), :]'), batchsize=mp.batchsize, shuffle=true)    

# test ways to sample the data for an approximator
num_samps = 8
images = zeros(Float32, num_samps, 2, 4,  length(motions)*501);

start = num_samps ÷ 2 # number of sample/2
stride = 64 ÷ num_samps
for (j,motion) in enumerate(motions)    
    sampler =  deepcopy(defaultDict)
    sampler[:Nt] = 100
    sampler[:N]  = num_samps  
    sampler[:motion_parameters] = [0.1]
    sampler[:f] = motion[2]
    sampler[:k] = motion[3]
    sampler[:wave] = motion[1]
    sampler[:ψ] = -π/2
    foil, flow = init_params(; sampler...)
    # (foil)(flow)
    image = []
    slice = ns*(j-1)+1:ns*j 
    ivs = inputs[stride:stride:end,:,3,slice]    
    for i = 1:501
        (foil)(flow)
        get_panel_vels!(foil, flow)
        foil.col[1,:]  = foil.col[1,:] .- minimum(foil.foil[1,:])
        # @show foil.panel_vel
        push!(image, reshape([foil.col' foil.panel_vel' foil.normals' ivs[:,:,i] ], (num_samps,2,4)))
        # # foil.col[2,:]  .+= 0.05*sin(-2π*foil.f*flow.Δt*flow.n)
        # plot(foil,marker=:circle,lw=0, label="sample")
        # plot!(inputs[:,1,1,i], inputs[:,2,1,i], label="panels")
        # plot!(ylims=(-0.3,0.3))
    end
    images[:,:,:,slice] = cat(image..., dims=4)
    # gif(movie, "panels.gif", fps = 30)

end

plot(foil,marker=:circle,lw=0, label="sample")
plot!(inputs[:,1,1,i], inputs[:,2,1,i], label="panels")
plot!(inputs[2:4:end,1,1,i], inputs[2:4:end,2,1,i],marker=:square,lw=0, label="panels")
plot!(ylims=(-0.3,0.3))

plot(inputs[stride:stride:end,1,1,i], label="panels")
plot!(foil.col[1,:], marker=:circle,lw=0, label="sample")


dl = DataLoader((approx = images, RHS = RHS), batchsize=2048, shuffle=true)
sd = dl|>first
W,H,C,N = size(images)
model = Chain(Conv((2,2), C=>C, Flux.tanh, pad=SamePad()),
              Conv((4,2), C=>C, Flux.tanh, pad=SamePad()),
              Conv((num_samps,1), C=>1, Flux.tanh, pad=SamePad()),
              Flux.flatten,
              Dense(W*H, W*H, Flux.tanh),              
              Dense(W*H, 64)) |> gpu
              

model(sd.approx|>gpu)

state = Flux.setup(ADAM(0.01), model)

losses = []
for epoch = 1:100
    for (x, y) in dl  
        x = x |> mp.dev
        y = y |> mp.dev
        ls = 0.0
        grads = Flux.gradient(model) do m
            # Evaluate model and loss inside gradient context:                
            ls = Flux.mse(m(x), y)              
        end
        Flux.update!(state, model, grads[1])
        push!(losses, ls)  # logging, outside gradient context
        
    end
    if epoch %  10 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end
out = model(images|>gpu) |>cpu
@show errorL2(out, RHS)
ers = norm.([x for x in eachcol(out .- RHS)], 2)./norm.([x for x in eachcol(RHS)], 2)
val, this = findmax(ers)
val, this = findmin(ers)
# this -= 1
this = rand(1:size(images,4))
plot(RHS[:,this], label="RHS")
plot!(model(images[:,:,:,this:this]|>gpu) |>cpu, seriestype=:scatter, label="model")


outo = B_DNN.layers[1](dataloader.data.inputs|>gpu)|>cpu
@show errorL2(outo, dataloader.data.RHS)
erso = norm.([x for x in eachcol(outo .- RHS)], 2)./norm.([x for x in eachcol(RHS)], 2)
val, this = findmax(erso)
val, this = findmin(erso)
# this -= 1
this = rand(1:size(images,4))
plot(RHS[:,this], label="RHS")
plot!(B_DNN.layers[1](dataloader.data.inputs[:,:,:,this:this]|>gpu)|>cpu, seriestype=:scatter, label="model")

encdec = Chain(Dense(64, num_samps, Flux.tanh), Dense(num_samps, 64)) |> gpu
estate = Flux.setup(ADAM(0.01), encdec)
losses = []
for epoch = 1:100
    for (x, y) in dl  
        x = x |> mp.dev
        y = y |> mp.dev
        ls = 0.0
        grads = Flux.gradient(encdec) do m
            # Evaluate model and loss inside gradient context:                
            ls = errorL2(m(y), y)              
        end
        Flux.update!(estate, encdec, grads[1])
        push!(losses, ls)  # logging, outside gradient context
        
    end
    if epoch %  10 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end

eout = encdec(RHS|>gpu) |>cpu
@show errorL2(eout, RHS)
encer = norm.([x for x in eachcol(eout .- RHS)], 2)./norm.([x for x in eachcol(RHS)], 2)
val, this = findmax(encer)
val, this = findmin(encer)

this = rand(1:size(images,4))
plot(RHS[:,this], label="RHS")
plot!(eout[:,this], seriestype=:scatter, label="model")