# make a file to explore what is the smallest space 
# that can be used to represent the data for the input images 
# x_in = [position' row.normals' row.wake_ind_vel' row.panel_vel' ]    


# try compressing the data to 1/4 of the original size

dataloader
W,H,C,N = size(dataloader.data.inputs)
convenc = Chain(Conv((4,2), C=>C, Flux.tanh, pad=SamePad()),                
                Conv((5,1), C=>C, Flux.tanh),
                Conv((1,2), C=>1, Flux.tanh), 
                x->reshape(x,(size(x,1), size(x,4))),
                Dense(N=>N, Flux.tanh),
                Dense(N=>N, Flux.tanh)) |>mp.dev

convdec = Chain(Dense(N=>N, Flux.tanh),
                Dense(N=>N, Flux.tanh),
                x->reshape(x,(size(x,1), 1, 1, size(x,2))),
                ConvTranspose((1,2), 1=>C, Flux.tanh),
                ConvTranspose((5,1), C=>C, Flux.tanh),
                ConvTranspose((4,2), C=>C, Flux.tanh, pad=SamePad())) |> mp.dev         
left = Chain(Conv((4,1), C=>C, Flux.tanh, pad = SamePad()),
            #  AdaptiveMaxPool((64,2)),    
             Conv((4,1), C=>C, Flux.tanh, pad = SamePad()),
            #  AdaptiveMaxPool((32,2)),
             Conv((4,1), C=>C, Flux.tanh, pad = SamePad()),
            #  AdaptiveMaxPool((16,2)),
             Conv((4,1), C=>C, Flux.tanh, pad = SamePad()),
            #  AdaptiveMaxPool((8,2)),
             Conv((4,1), C=>C, pad = SamePad()))
right = Chain(ConvTranspose((4,1),C=>C, pad=SamePad()),
            #   Upsample((2,1)),
              ConvTranspose((4,1), C=>C, Flux.tanh, pad = SamePad()),
            #   Upsample((2,1)),
              ConvTranspose((4,1), C=>C, Flux.tanh, pad = SamePad()),
            #   Upsample((2,1)),
              ConvTranspose((4,1), C=>C,  pad = SamePad()))

(left(dataloader.data.inputs[3:end-2,:,:,1:1])) |> size
right(left(dataloader.data.inputs[3:end-2,:,:,1:1])) |> size

linAE = SkipConnection(Chain(left,right),.+)|> mp.dev
linAE = Chain(left,right)|> mp.dev
oAE = Flux.setup(Adam(mp.η), linAE)

losses = []
for epoch = 1:00
    for (x, _,_,_,_) in dataloader        
        x = x[3:end-2,:,:,:] |> mp.dev        
        local ls = 0.0
        grads = Flux.gradient(linAE) do m
            # Evaluate model and loss inside gradient context:                
            ls = errorL2(m(x), x)                          
        end
        Flux.update!(oAE, linAE, grads[1])
        push!(losses, ls)  # logging, outside gradient context

    end
    if epoch %  mp.epochs /10 == 0
        println("Epoch: $epoch, Loss: $(losses[end])")
    end
end

begin
    which = rand(1:N)
    sample = dataloader.data.inputs[3:end-2,:,:,which:which] |> mp.dev
    rs = reshape(sample, (64,8))
    approx = reshape(linAE(sample|>mp.dev) |> cpu, (64,8))
    @show L2 = errorL2(approx, rs|>cpu)
    a = plot(rs,st=:contourf,  label= "")
    b =plot!(approx, st=:contourf , label= "")
    c = plot(rs[:,1:2], label="truth")
    plot!(c, approx[:,1:2],marker=:circle, ms=1, label="recon")
    plot(a,b,c, layout=(3,1))
   
end


begin 
    which = rand(1:199)
    ns = 501
   
    slice = ns*which+1:ns*(which+1)-1 
    cut = dataloader.data.inputs[3:end-2,:,:,slice] 
    lat = linAE.layers[1](cut|> mp.dev) |> cpu
    approx = linAE(cut |> mp.dev) |> cpu

    movie = @animate for i = 1:500
        pos = cut[:,:,1,i]
        apos = approx[:,:,1,i]
        lpos = lat[:,:,1,i]
        plot(pos[:,1],pos[:,2], label="truth")
        plot!(apos[:,1],apos[:,2], label="recon")
        plot!(lpos[:,1],lpos[:,2], label="latent")
    end
    gif(movie, "test.gif", fps = 10)

end