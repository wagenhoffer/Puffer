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
    x1, x2, y = panels_frame(field, body)
    phid = zeros(size(x1))
    phis = zeros(size(x1))
    @. phis= x1*log(x1^2 + y^2) - x2*log(x2^2 + y^2) + 2*y*(atan(y, x2) - atan(y, x1)) 
    @. phid = (atan(y, x2) - atan(y, x1))
    phis * σs / 4π - phid * μs / 2π
end
function ∇Φ(field, body, normals, σs, μs)
    x1, x2, y = panels_frame(field, body)
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
function panels_frame(target, source)
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

#TODO: bust into spatial param and values
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
        σs = rand(-10:0.1:10).*circshift(sin.(LinRange(rand(0:pi/64:pi/2), rand(1:8)*pi, n)).+rand(n).-0.5, rand(1:n))
        μs = rand(-10:0.1:10).*circshift(sin.(LinRange(rand(0:pi/64:pi/2), rand(1:8)*pi, n)).+rand(n).-0.5,rand(1:n))
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
phis = phis 
noise = rand(size(phis)...)./20 .|>Float32  #5% noise
dl = Flux.DataLoader((phis = phis.|>gpu, μs = μs.|>gpu, us = us.|>gpu,
                      vs = vs.|>gpu, σs= σs.|>gpu, noise = noise.|>gpu), 
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

# Purely data driven training
ϕlosses = []
for i=1:50
    for (phif, mub,_,_,sigb, noise) in dl
        phif = phif|>gpu
        noise = noise|>gpu
        phif .+= noise
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
    choose = rand(1:size(circles,1))
    acircle = circles[choose]
    danoise = dl.data.noise[:,choose]
    φ = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gf)) |> cpu
    δϕ = φ.-acircle.ϕ0
    clims = extrema(acircle.ϕ0)
    a = scatter(field[1,:], field[2,:],aspect_ratio=:equal, ms=5,markerstrokewidth=0,clims=clims, marker_z =φ,    st=:scatter, c=:coolwarm, label="model")
    b = scatter(field[1,:], field[2,:],aspect_ratio=:equal, ms=5,markerstrokewidth=0,clims=clims, marker_z= acircle.ϕ0, st=:scatter, c=:coolwarm, label="truth")
    c = scatter(field[1,:], field[2,:],aspect_ratio=:equal, ms=5,markerstrokewidth=0, marker_z= δϕ, st=:scatter, c=:coolwarm, label="Δϕ")
    d = scatter(field[1,:], field[2,:],aspect_ratio=:equal, ms=5,markerstrokewidth=0, marker_z= danoise, st=:scatter, c=:coolwarm, label="noise")
    e = plot(a,b, layout =(1,2), size=(900,400))   
    f = plot(c,d, layout =(1,2), size=(900,400))
    @show err = Flux.mse(φ, acircle.ϕ0)/Flux.mse(acircle.ϕ0, zeros(size(acircle.ϕ0)))
    title!("L_2 error: $(err)")
    
    plot(e, f,layout=(2,1), size=(900,800))
end

# contour(field[1,:], field[2,:], φ,aspect_ratio=:equal, c=:coolwarm)


ostate = Flux.setup(Adam(0.001), deepO)

loss2(m,x,y) = Flux.mse(m(x),y)/Flux.mse(y, zeros(size(y)))
loss2(x,y) = Flux.mse(x,y)/Flux.mse(y, zeros(size(y))|>gpu)

# ϵ = cbrt(eps(Float32))
(xp, xm, yp, ym)  = stencil_points(field;ϵ=ϵ)
(gxp, gxm, gyp, gym) = map(gpu, (xp, xm, yp, ym))
ϕlosses = []
for i=1:500
    for (phif, mub, us, vs,sigb,noise) in dl
        phif = phif|>gpu
        noise = noise|>gpu
        # phif .+= noise
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
            Δϕ = (-4*out .+ plx .+ mlx .+ ply .+ mly)/ϵ^2
            loss = Flux.mse(Δϕ, 0.0)            
            loss += loss2(out, phif)*10
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
    # ϵ = cbrt(eps(Float32))
    (xp, xm, yp, ym)  = stencil_points(field;ϵ=ϵ)
    (gxp, gxm, gyp, gym) = map(gpu, (xp, xm, yp, ym))
    #see if this can be used to learn the velocity field
    acircle = rand(circles)
    φ   = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gf)) |> cpu 
    φpx = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gxp)) |> cpu 
    φmx = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gxm)) |> cpu 
    φpy = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gyp)) |> cpu 
    φmy = deepO((vcat(acircle.μs,acircle.σs)|>gpu,gym)) |> cpu 
    Δϕ = (-4*φ .+ φpx .+ φmx .+ φpy .+ φmy)/ϵ^2
    @show Flux.mse(Δϕ,0.0)
    u = (φpx -φmx)/2ϵ
    v = (φpy -φmy)/2ϵ
    scl = [√2 √2] #[norm(u,2) norm(v,2)]
    modelc = mod2pi.(atan.(v,u))./2pi
    @show Flux.mse(acircle.∇Φ[1,:], u)/Flux.mse(acircle.∇Φ[1,:], zeros(size(acircle.∇Φ[1,:])))
    @show Flux.mse(acircle.∇Φ[2,:], v)/Flux.mse(acircle.∇Φ[2,:], zeros(size(acircle.∇Φ[2,:])))

    px  = phi_field(xp, acircle.body, acircle.σs, acircle.μs)
    mx  = phi_field(xm, acircle.body, acircle.σs, acircle.μs)
    py  = phi_field(yp, acircle.body, acircle.σs, acircle.μs)
    my  = phi_field(ym, acircle.body, acircle.σs, acircle.μs)
    us = (px - mx) /2ϵ
    vs = (px - mx) /2ϵ

    quiver(field[1,:] , field[2,:], quiver=(u./scl[1],v./scl[2]), aspect_ratio=:equal, c=:red, lw=2, label="model")
    quiver!(field[1,:], field[2,:], quiver=(acircle.∇Φ[1,:]./scl[1],acircle.∇Φ[2,:]./scl[2]), aspect_ratio=:equal, c=:blue, lw=2, label="truth")
    # quiver!(field[1,:], field[2,:], quiver=(us./scl[1],vs./scl[2]), aspect_ratio=:equal, c=:green, lw=2, label="stencil")
    # quiver!(field[1,:].+15, field[2,:], quiver=(u.-acircle.∇Φ[1,:],v.-acircle.∇Φ[2,:]), aspect_ratio=:equal, c=:green, lw=2, label="Δv")
end
begin
    plot(φ, label="Deep")
    plot!(acircle.ϕ0, label="Truth")
    plot!(phi_field(field, acircle.body, acircle.σs, acircle.μs), label="Stencil")
end


#verification of the process : ∇^2ϕ = 0 and [u,v] = ∇ϕ(px,mx,py,my) = stencil_points(field;ϵ=ϵ)

thiscircle = circles[42]

ori = phi_field(field, thiscircle.body, thiscircle.σs, thiscircle.μs)
px  = phi_field(xp, thiscircle.body, thiscircle.σs, thiscircle.μs)
mx  = phi_field(xm, thiscircle.body, thiscircle.σs, thiscircle.μs)
py  = phi_field(yp, thiscircle.body, thiscircle.σs, thiscircle.μs)
my  = phi_field(ym, thiscircle.body, thiscircle.σs, thiscircle.μs)
φpx = deepO((vcat(thiscircle.μs,thiscircle.σs)|>gpu,gxp)) |> cpu 
φmx = deepO((vcat(thiscircle.μs,thiscircle.σs)|>gpu,gxm)) |> cpu
φpy = deepO((vcat(thiscircle.μs,thiscircle.σs)|>gpu,gyp)) |> cpu
φmy = deepO((vcat(thiscircle.μs,thiscircle.σs)|>gpu,gym)) |> cpu

Δϕ = (-4*ori .+ px .+ mx .+ py .+ my)/ϵ^2
extrema(Δϕ)
√sum(abs2, Δϕ)
this∇Φ = ∇Φ(field, thiscircle.body, thiscircle.normals, thiscircle.σs, thiscircle.μs)
stenuv = [ (px - mx) (py - my)]'/2ϵ
# sten2  = [ (px - ori)/ϵ (py - ori)/ϵ]'
guv    = [(φpx - φmx) (φpy - φmy)]'/2ϵ
thiscircle.∇Φ

plot(guv[1,:], label="Deep")
plot!(stenuv[1,:], label="Stencil")
p1 = plot!(this∇Φ[1,:], label="Truth")

plot(guv[2,:], label="Deep")
plot!(stenuv[2,:], label="Stencil")
p2 = plot!(this∇Φ[2,:], label="Truth")

plot(p1,p2, layout=(2,1), size=(900,800))

quiver(field[1,:] , field[2,:], quiver=(stenuv[:,1],stenuv[:,2]), aspect_ratio=:equal, c=:red, lw=2, label="model")
quiver!(field[1,:] , field[2,:], quiver=(thiscircle.∇Φ[1,:],thiscircle.∇Φ[2,:]), aspect_ratio=:equal, c=:blue, lw=2, label="truth")


α = -π/4
x = 0.0
y = 1.0
mu = 1.0

phi_anal(x,y) = -mu/2π*(x*cos(α) + y*sin(α))/(x^2 + y^2)
u_anal(x,y)   = mu/2π*(x^2*cos(α) - y^2*cos(α) + 2*x*y*sin(α))/(x^2 + y^2)^2
v_anal(x,y)   = mu/2π*(y^2*sin(α) - x^2*sin(α) + 2*x*y*cos(α))/(x^2 + y^2)^2
ϵ = sqrt(eps(Float32))
ppx(x,y) = phi_anal(x.+ϵ, y)
pmx(x,y) = phi_anal(x.-ϵ, y)
ppy(x,y) = phi_anal(x,   y.+ϵ)
pmy(x,y) = phi_anal(x,   y.-ϵ)

usten(x,y) = (ppx(x,y) - pmx(x,y))/2ϵ
vsten(x,y) = (ppy(x,y) - pmy(x,y))/2ϵ

usten(x,y)
vsten(x,y)
u_anal(x,y)
v_anal(x,y)

nxs = 11
nys = 15
xs = LinRange(-10,10,nxs)
ys = LinRange(-5,5,nys)
Xs = repeat(xs,  1,  nys)
Ys = repeat(ys', nxs, 1)
errs = zeros(2,nxs,nys)
phif = zeros(nxs,nys)
velf = zeros(2,nxs,nys)
for (i,x) in enumerate(xs) 
    for (j,y) in enumerate(ys)
        errs[1, i, j] = loss2(usten(x,y), u_anal(x,y))
        errs[2, i, j] = loss2(vsten(x,y), v_anal(x,y))
        phif[i,j] = phi_anal(x,y)
        velf[1,i,j] = u_anal(x,y)
        velf[2,i,j] = v_anal(x,y)
    end
end
#clean the NaNs
errs[isnan.(errs)] .= 0

a = plot(xs,ys, errs[1,:,:]', st=:contourf, color=:coolwarm)
b = plot(xs,ys, errs[2,:,:]', st=:contourf, color=:coolwarm)
c = plot(xs,ys, phif', st=:contourf, color=:coolwarm)
d =  quiver(Xs, Ys, quiver=(velf[1,:,:],velf[2,:,:]))
plot(a,b, layout=(2,1), size=(900,800))
extrema(errs)