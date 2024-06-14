#simple deeponet test
using Flux
using CUDA
using Plots

function normv(x, y)
    dxdy = hcat([(x - circshift(x, 1)) / 2, (y - circshift(y, 1)) / 2]...)'
    lens = sqrt.(sum(dxdy .^ 2, dims=1))
    tans = dxdy ./ lens
    return hcat([tans[2, :], -tans[1, :]]...)'
end
function circle(n)
    x = [cos(2π * i / n) for i in 1:n]
    y = [sin(2π * i / n) for i in 1:n]
    return x, y
end
function phi_field(field, body, σs, μs)
    x1, x2, y = panel_frame(field, body)
    phid = zeros(size(x1))
    phis = zeros(size(x1))
    @. phis= x1*log(x1^2 + y^2) - x2*log(x2^2 + y^2) + 2*y*(atan(y, x2) - atan(y, x1)) 
    @. phid = (atan(y, x2) - atan(y, x1))
    phis * σs / 4π - phid * μs / 2π
end
function ∇Φ(field, body, normals, σs, μs)
    x1, x2, y = panel_frame(field, body)
    nw, nb = size(x1)
    lexp = zeros((nw, nb))
    texp = zeros((nw, nb))
    yc = zeros((nw, nb))
    xc = zeros((nw, nb))
    β = atan.(-normals[1, :], normals[2, :])
    β = repeat(β, 1, nw)'
    # doublets    
    @. lexp = -(y / (x1^2 + y^2) - y / (x2^2 + y^2)) / 2π
    @. texp = (x1 / (x1^2 + y^2) - x2 / (x2^2 + y^2)) / 2π
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv = [xc * μs yc * μs]'
    # sources
    @. lexp = log((x1^2 + y^2) / (x2^2 + y^2)) / (4π)
    @. texp = (atan(y, x2) - atan(y, x1)) / (2π)
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv += [xc * σs yc * σs]'
end
function panel_frame(target, source)
    Ns = size(source, 2)
    Nt = size(target, 2)
    Ns -= 1 #start/end at same location
    x1 = zeros(Ns, Nt)
    x2 = zeros(Ns, Nt)
    y = zeros(Ns, Nt)
    ns = normv(source[1, :], source[2, :])[:,2:end]
    ts = hcat(ns[2,:], -ns[1,:])'
    txMat = repeat(ts[1, :]', Nt, 1)
    tyMat = repeat(ts[2, :]', Nt, 1)
    dx = repeat(target[1, :], 1, Ns) - repeat(source[1, 1:(end - 1)]', Nt, 1)
    dy = repeat(target[2, :], 1, Ns) - repeat(source[2, 1:(end - 1)]', Nt, 1)
    x1 = dx .* txMat + dy .* tyMat
    y = -dx .* tyMat + dy .* txMat
    x2 = x1 - repeat(sum(diff(source, dims = 2) .* ts, dims = 1), Nt, 1)
    x1, x2, y
end
function stencil_points(points; ϵ=sqrt(eps(Float32)))
    xp = deepcopy(points)
    xp[1,:] .+= ϵ
    xm = deepcopy(points)
    xm[1,:] .-= ϵ
    yp = deepcopy(points)
    yp[2,:] .+= ϵ
    ym = deepcopy(points)
    ym[2,:] .-= ϵ
    (xp, xm, yp, ym)
end

mutable struct Circle    
    normals::Matrix{Float64}
    body::Matrix{Float64}    
    σs::Vector{Float64}
    μs::Vector{Float64}
    ϕ0::Vector{Float64}
    ∇Φ::Matrix{Float64}
    function Circle(n)
        x,y = circle(n)
        normals = normv(x, y)
        body = [x... x[1]; y... y[1]]
        σs = circshift(sin.(LinRange(rand(0:pi/64:pi/2), rand(1:8)*pi, n)).+rand(n).-0.5, rand(1:n))
        μs = circshift(sin.(LinRange(rand(0:pi/64:pi/2), rand(1:8)*pi, n)).+rand(n).-0.5,rand(1:n))
        ϕ0 = phi_field(field, body, σs, μs)
        ∇ϕ = ∇Φ(field, body, normals,σs, μs)
        new(normals, body,σs, μs, ϕ0, ∇ϕ)
    end
end


XS = []
YS = []        
for (i,r) in enumerate(1.5:0.25:5)
    X, Y = circle(10 + 2*(i-1)) .* r            
    push!(XS, X)
    push!(YS, Y)        
end
field = vcat(vcat(XS...)', vcat(YS...)')
# sort the field based on x row as the key
# field = field[:,sortperm(field[1,:])]

Circle(2) #warmup
circles = [Circle(8) for i in 1:4096*10]
phis  = hcat([c.ϕ0 for c in circles]...)
μs    = hcat([c.μs for c in circles]...)
σs    = hcat([c.σs for c in circles]...)
us  = hcat([c.∇Φ[1,:] for c in circles]...)
vs  = hcat([c.∇Φ[2,:] for c in circles]...)
ϵ=cbrt(eps(Float32))

gf = field|>gpu

dl = Flux.DataLoader((phis = phis.|>gpu, μs = μs.|>gpu, us = us.|>gpu, vs = vs.|>gpu, σs= σs.|>gpu), 
                    batchsize = 2048, shuffle = true)

# simple DeepOnet
trunk  = Chain(Dense(2,  32, Flux.tanh),
               Dense(32, 32, Flux.tanh),
               Dense(32, 32, Flux.tanh))|>gpu
branch = Chain(Dense(16,  32, Flux.tanh),
               Dense(32, 32, Flux.tanh),
               Dense(32, 32, Flux.tanh))|>gpu


trunk(field|>gpu)
branch(vcat(μs,σs)|>gpu)

deepO = Parallel((x,y)->permutedims(x'*y,(2,1)), branch, trunk)|>gpu
deepO(map(gpu,(vcat(μs,σs),field)))


ostate = Flux.setup(Adam(0.001), deepO)

loss2(m,x,y) = Flux.mse(m(x),y)/Flux.mse(y, zeros(size(y)))
loss2(x,y) = Flux.mse(x,y)/Flux.mse(y, zeros(size(y))|>gpu)


ϕlosses = []
for i=1:500
    for (phif, mub,_,_,sigb) in dl
        phif = phif|>gpu
        mub  = mub|>gpu
        sigb = sigb|>gpu
        mub = vcat(mub,sigb)
        loss = 0.0
        grads = Flux.gradient(deepO) do m            
            out   = m((mub,gf))            
            loss = loss2(out, phif)
        end
        Flux.update!(ostate, deepO, grads[1])
    
        push!(ϕlosses, loss)
    end
    if i%10 == 0
        println("Epoch: $i, Loss: $(ϕlosses[end])")
    end    
end    



begin
    acircle = rand(circles)
    φ = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gf)) |> cpu
    δϕ = φ.-acircle.ϕ0
    a = scatter(field[1,:], field[2,:],aspect_ratio=:equal, ms=5,markerstrokewidth=0, marker_z =φ,    st=:scatter, c=:coolwarm, label="model")
    b = scatter(field[1,:], field[2,:],aspect_ratio=:equal, ms=5,markerstrokewidth=0, marker_z= acircle.ϕ0, st=:scatter, c=:coolwarm, label="truth")
    c = scatter(field[1,:], field[2,:],aspect_ratio=:equal, ms=5,markerstrokewidth=0, marker_z= δϕ, st=:scatter, c=:coolwarm, label="Δϕ")
    @show err = Flux.mse(φ, acircle.ϕ0)/Flux.mse(acircle.ϕ0, zeros(size(acircle.ϕ0)))
    title!("L_2 error: $(err)")
    
    plot(a,b,c,layout=(1,3), size=(900,400))
end

# contour(field[1,:], field[2,:], φ,aspect_ratio=:equal, c=:coolwarm)


ostate = Flux.setup(Adam(0.001), deepO)

loss2(m,x,y) = Flux.mse(m(x),y)/Flux.mse(y, zeros(size(y)))
loss2(x,y) = Flux.mse(x,y)/Flux.mse(y, zeros(size(y))|>gpu)

ϵ = cbrt(eps(Float32))
(xp, xm, yp, ym)  = stencil_points(field;ϵ=ϵ)
(gxp, gxm, gyp, gym) = map(gpu, (xp, xm, yp, ym))
ϕlosses = []
for i=1:500
    for (phif, mub, us, vs,sigb) in dl
        phif = phif|>gpu
        mub  = mub|>gpu
        sigb = sigb|>gpu
        mub = vcat(mub,sigb)
        us   = us|>gpu
        vs   = vs|>gpu

        loss = 0.0
        grads = Flux.gradient(deepO) do m            
            out = m((mub,gf))
            plx = m((mub,gxp))
            mlx = m((mub,gxm))
            ply = m((mub,gyp))
            mly = m((mub,gym))
            um  = (plx - mlx)/2ϵ
            vm  = (ply - mly)/2ϵ
            Δϕ = -4*out .+ plx .+ mlx .+ ply .+ mly
            loss = Flux.mse(Δϕ, 0.0)            
            loss += loss2(out, phif)
            loss += loss2(um, us)
            loss += loss2(vm, vs)

        end
        Flux.update!(ostate, deepO, grads[1])
    
        push!(ϕlosses, loss)
    end
    if i%10 == 0
        println("Epoch: $i, Loss: $(ϕlosses[end])")
    end    
end    




begin
    ϵ = cbrt(eps(Float32))
    (xp, xm, yp, ym)  = stencil_points(field;ϵ=ϵ)
    (gxp, gxm, gyp, gym) = map(gpu, (xp, xm, yp, ym))
    #see if this can be used to learn the velocity field
    acircle = rand(circles)
    φ   = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gf)) |> cpu 
    φpx = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gxp)) |> cpu 
    φmx = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gxm)) |> cpu 
    φpy = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gyp)) |> cpu 
    φmy = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gym)) |> cpu 
    Δϕ = -4*φ .+ φpx .+ φmx .+ φpy .+ φmy
    @show Flux.mse(Δϕ,0.0)
    u = (φpx -φmx)/2ϵ
    v = (φpy -φmy)/2ϵ

    modelc = mod2pi.(atan.(v,u))./2pi
    @show Flux.mse(acircle.∇Φ[1,:], u)/Flux.mse(acircle.∇Φ[1,:], zeros(size(acircle.∇Φ[1,:])))
    @show Flux.mse(acircle.∇Φ[2,:], v)/Flux.mse(acircle.∇Φ[2,:], zeros(size(acircle.∇Φ[2,:])))


    quiver(field[1,:] , field[2,:], quiver=(u,v), aspect_ratio=:equal, c=:red, lw=2, label="model")
    quiver!(field[1,:], field[2,:], quiver=(acircle.∇Φ[1,:],acircle.∇Φ[2,:]), aspect_ratio=:equal, c=:blue, lw=2, label="truth")
    # quiver!(field[1,:].+15, field[2,:], quiver=(u.-acircle.∇Φ[1,:],v.-acircle.∇Φ[2,:]), aspect_ratio=:equal, c=:green, lw=2, label="Δv")
end