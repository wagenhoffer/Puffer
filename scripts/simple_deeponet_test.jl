#simple deeponet test

using Flux


#define a discrete circle
function circle(n)
    x = [cos(2π * i / n) for i in 1:n]
    y = [sin(2π * i / n) for i in 1:n]
    return x, y
end
function normv(x, y)
    dxdy = hcat([(x - circshift(x, 1)) / 2, (y - circshift(y, 1)) / 2]...)'
    lens = sqrt.(sum(dxdy .^ 2, dims=1))
    tans = dxdy ./ lens
    return hcat([tans[2, :], -tans[1, :]]...)'
end
x, y = circle(8)
normals = normv(x, y)
plot(x, y)
#collocation points
cx, cy = (x + circshift(x, 1)) / 2, (y + circshift(y, 1)) / 2
plot!(cx, cy, st=:scatter)
quiver!(cx, cy, quiver=(normals[1, :], normals[2, :]))
body = [x... x[1]; y... y[1]]
#Exterior field to find ϕ at 
i = 0
XS = []
YS = []
plot()
for r = 1.5:0.25:5
    X, Y = circle(10 + i) .* r
    plot!(X, Y, st=:scatter, label="")
    push!(XS, X)
    push!(YS, Y)
    i += 2
end
plot!()

XS = vcat(XS...)
YS = vcat(YS...)
field = vcat(XS', YS')

μs = LinRange(-1, 1, 8)
function phi_field(field, body, μs)
    x1, x2, y = panel_frame(field, body)
    phid = zeros(size(x1))
    @. phid = (atan(y, x2) - atan(y, x1))
    -phid * μs / 2π
end

ϕ0 = phi_field(field, body, μs)

heatmap(XS, YS, ϕ0, st=:surface, c=:viridis, view=(0, 180, 90))
plot(XS,YS, st=:scatter, c=:viridis, ms=ϕ0*100, msw=0, msc=:viridis, legend=false)
function ∇Φ(field, body, normals, μs)
    x1, x2, y = panel_frame(field, body)
    nw, nb = size(x1)
    lexp = zeros((nw, nb))
    texp = zeros((nw, nb))
    yc = zeros((nw, nb))
    xc = zeros((nw, nb))
    β = atan.(-normals[1, :], normals[2, :])
    β = repeat(β, 1, nw)'
    # #cirulatory effects    
    @. lexp = -(y / (x1^2 + y^2) - y / (x2^2 + y^2)) / 2π
    @. texp = (x1 / (x1^2 + y^2) - x2 / (x2^2 + y^2)) / 2π
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    uv = [xc * μs yc * μs]'
end

