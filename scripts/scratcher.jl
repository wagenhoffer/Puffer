#### A bunch of functions and data for testing and debugging    ####
function is_sym(values::Vector, mid=25)
    l, r = values[1:mid], values[end:-1:mid+1]
    @show sum(l - r), l - r
    abs(sum(l - r)) < 1e-10
end
function is_sym(values::Matrix, mid=25)
    l, r = values[:, 1:mid], values[:, end:-1:mid+1]
    @show l - r
    sum(l - r) == 0
end
function is_opp(values::Matrix, mid=25)
    l, r = values[:, 1:mid], values[:, end:-1:mid+1]
    @show l + r
    sum(l + r) == 0
end
function is_opp(values::Vector, mid=25)
    l, r = values[1:mid], values[end:-1:mid+1]
    @show l + r
    sum(l + r) == 0
end
acc_lens = cumsum(foil.panel_lengths[:])
# B = BSplineBasis(BSplineOrder(4), acc_lens[:])
S = interpolate(acc_lens, foil.μs, BSplineOrder(4))
S = interpolate(acc_lens, mu, BSplineOrder(4))
# R_n = RecombinedBSplineBasis(Derivative(1), S)
dmu = Derivative(1) * S
dmudl = dmu.(acc_lens)
plot(acc_lens, S.(acc_lens))
plot!(acc_lens, dmu.(acc_lens), st=:scatter)
#TODO: SPLIT σ into local and induced? it is lumped into one now
qp = repeat(dmudl', 2, 1) .* foil.tangents + repeat(foil.σs', 2, 1) .* foil.normals
fdmu = diff(foil.μs)' ./ diff(acc_lens)'

dmudt, old_mus = get_dmudt!(foil, flow)
dmudt, old_mus = get_dmudt!(old_mus, foil, flow)
qp = get_qp(foil)
p_s = -flow.ρ * sum(qp .^ 2, dims=1) / 2.0
p_us = flow.ρ .* dmudt + flow.ρ .* (qp[1, :] .* (flow.Uinf) .+ qp[2, :] .* foil.panel_vel[2, :])
p = p_s .+ p_us'
cp = p ./ (0.5 * flow.ρ * flow.Uinf^2)

dforce = repeat(-p .* foil.panel_lengths', 2, 1) .* foil.normals
dpress = sum(dforce .* foil.panel_vel, dims=2)
force = sum(dforce, dims=2)
lift = force[1]
thrust = force[2]
power = sum(dpress, dims=1)[1]
den = 1 / 2 * (flow.ρ * flow.Uinf^2 * foil.chord)
cforce = sum(sqrt.(force .^ 2)) / den
clift = lift / den
cthrust = thrust / den
cpower = power / den / flow.Uinf
plot(foil.col[1, :], cp')
### SCRATHCERS
# rotating old datums - mu, phi, induced velocity
old_mus = zeros(3, foil.N)
for i = 1:2
    old_mus[i, :] .= i
end
old_mus

old_mus[2, :]
dmudt = (3 * foil.μs - 4 * old_mus[1, :] + old_mus[2, :]) / (2 * flow.Δt)
old_mus = circshift(old_mus, (1, 0))
old_mus[1, :] = foil.μs
# %% 
sten = (old_cols[2, :, :] - old_cols[1, :, :]) / (flow.Δt) #.-[flow.Uinf,0]
kine(t) = foil.kine.(foil._foil[1, :], foil.f, foil.k, t)
vy = ForwardDiff.derivative(kine, flow.Δt * 1)

yp = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, 0.02)
ym = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, 0.0)
dy = (yp - ym) / (0.02) #velocity at 0.01
# %%
#IT WORKS!


kine(x) = foil.kine(x, foil.f, foil.k, flow.n * flow.Δt)
ForwardDiff.derivative(kine, flow.Δt * 1)
ForwardDiff.derivative.(kine, foil._foil[1, :])


A, rhs = make_infs(foil, flow)
mu = A \ rhs

foil.μs = mu
release_vortex!(wake, foil)
set_edge_strength!(foil)

plot!(mu, marker=:circle)



function stupid_scope(x)
    @show flow.δ
end



function foil_dx_dy(N, foil)
    dx = zeros(N, N)
    dy = zeros(N, N)
    for i = 1:N
        @inbounds dx[:, i] = foil[1, :] .- foil[1, i]
        @inbounds dy[:, i] = foil[2, :] .- foil[2, i]
    end
    [dx, dy]
end
dx, dy = foil_dx_dy(N, foil)
# Find distance matrices -> first with on swimmer and then with several
#target - source
function this(N, foil)
    dx = zeros(N, N)
    dy = zeros(N, N)
    for i = 1:N
        @inbounds dx[:, i] = foil[1, :] .- foil[1, i]
        @inbounds dy[:, i] = foil[2, :] .- foil[2, i]
    end
    [dx, dy]
end
function that(N, foil)
    dx = repeat(foil[1, :], 1, N) - repeat(foil[1, :]', N, 1)
    dy = repeat(foil[2, :], 1, N) - repeat(foil[2, :]', N, 1)
    nothing
end

@btime this(N, foil) #N=2001 35.652 ms (8008 allocations: 184.69 MiB)
@btime that(N, foil) #N=2001 118.981 ms (20 allocations: 183.35 MiB)

function norms(foil)
    dxdy = diff(foil, dims=2)
    lengths = sqrt.(sum(abs2, diff(foil, dims=2), dims=1))
    tx = dxdy[1, :]' ./ lengths
    ty = dxdy[2, :]' ./ lengths
    # tangents x,y normals x, y  lengths
    return tx, ty, -ty, tx, lengths
end

tx, ty, nx, ny, lengths = norms(foil)
@assert [nx, ny] ⋅ [tx, ty] == 0.0
plot(foil[1, :], foil[2, :], aspect_ratio=:equal, marker=:circle)
plot!(foil_col[1, :], foil_col[2, :], aspect_ratio=:equal, marker=:star, color=:red)
quiver!(foil_col[1, :], foil_col[2, :], quiver=(nx[1:end], ny[1:end]))

plot(foil[1, :], foil[2, :], aspect_ratio=:equal, marker=:circle)
quiver!(foil_col[1, :], foil_col[2, :], quiver=(tx[1:end], ty[1:end]))


begin
    move = copy(foil)
    move[2, :] += h.(foil[1, :], 0.25)
    tx, ty, nx, ny, ll = norms(move)
    col = get_mdpts(move)

    plot(move[1, :], move[2, :], aspect_ratio=:equal, marker=:circle)
    plot!(col[1, :], col[2, :], aspect_ratio=:equal, marker=:star, color=:red)
    quiver!(col[1, :], col[2, :], quiver=(nx[1:end], ny[1:end]), length=0.1)
end


tx, ty, nx, ny, ll = norms(foil)
##CHANGE THE PANEL FRAMES -endpnts are the sources, col are targets 
x1 = zeros(N - 1, N - 1)
x2 = zeros(N - 1, N - 1)
y = zeros(N - 1, N - 1)
txMat = repeat(tx, N - 1, 1)
tyMat = repeat(ty, N - 1, 1)
dx = repeat(foil_col[1, :], 1, N - 1) - repeat(foil[1, 1:end-1]', N - 1, 1)
dy = repeat(foil_col[2, :], 1, N - 1) - repeat(foil[2, 1:end-1]', N - 1, 1)
# for i=1:N-1
#     x1[:,i] = (foil_col[1,:] .- foil[1,i])
#     x2[:,i] = (foil_col[2,:] .- foil[2,i])    
# end
# dx = x1
# dy = x2
x1 = dx .* txMat + dy .* tyMat
y = dx .* -tyMat + dy .* txMat

x2 = x1 - repeat(sum(diff(foil, dims=2) .* [tx; ty], dims=1), N - 1, 1)

#Grabbed from python code
#strictly for testing though y here and z there are off 
pyx1 = [[0.12596995, -0.12568028, -0.59794013, 0.84768749, 0.6282324,
        0.12983482]
    [0.50176151, 0.25039739, -0.23996686, 0.47597968, 0.25550035,
        -0.23859873]
    [0.87194163, 0.62392359, 0.12848079, 0.11474624, -0.11978317,
        -0.61264372]
    [0.86458362, 0.62057795, 0.14221534, 0.12848079, -0.1231288,
        -0.62000172]
    [0.49053864, 0.24529443, -0.2190181, 0.49692843, 0.25039739,
        -0.2498216]
    [0.12210508, -0.12743761, -0.59072591, 0.8549017, 0.62647507,
        0.12596995]]
pyx1 = reshape(pyx1, (6, 6))
pyx1' .- x1

py_z = [1.66533454e-17, -3.12043904e-02, -5.94075000e-02, -0.00000000e+00,
    5.94075000e-02, 3.12043904e-02, -1.66533454e-17]

py_z .- foil[2, :]
py_foil = [1.00000000e+00 7.50000000e-01 2.50000000e-01 0.00000000e+00 2.50000000e-01 7.50000000e-01 1.00000000e+00
    1.66533454e-17 -3.12043904e-02 -5.94075000e-02 -0.00000000e+00 5.94075000e-02 3.12043904e-02 -1.66533454e-17]

py_col = get_mdpts(py_foil)



x1, x2, y = panel_frame(py_col, py_foil)

x1, x2, y = panel_frame(foil_col, foil)
doubletMat = doublet_inf.(x1, x2, y)
sourceMat = source_inf.(x1, x2, y)
plot(x1, st=:contourf)
plot(x2, st=:contourf)
plot(y, st=:contourf)


plot!(col[1, :], col[2, :], aspect_ratio=:equal, marker=:star, color=:red)
quiver!(col[1, :], col[2, :], quiver=(nx[1:end], ny[1:end]), length=0.1)

#define the edge off of the TE
Uinf = 1.0
Δt = 0.1
#use this def to allow for implicit kutta in the future
edge_vec = Uinf * Δt * [(tx[end] - tx[1]) / 2.0, (ty[end] - ty[1]) / 2.0]
edge = [foil[:, end] foil[:, end] .+ edge_vec foil[:, end] .+ 2 * edge_vec]

plot(foil[1, :], foil[2, :], aspect_ratio=:equal, marker=:circle)
plot!(edge[1, :], edge[2, :], aspect_ratio=:equal, marker=:star, color=:green)



sigmas = sum([Uinf, 0] .* [nx; ny], dims=1)
rhs = -sourceMat * sigmas'
doubletMat \ rhs
#this all matches to now 

#make the edge doublet work
panel_frame(foil, edge)

##CHANGE THE PANEL FRAMES -endpnts are the sources, col are targets 
_, Ns = size(source)
_, Nt = size(target)
Ns -= 1 #TE accomodations
x1 = zeros(Ns, Nt)
x2 = zeros(Ns, Nt)
y = zeros(Ns, Nt)
tx, ty, nx, ny, lens = norms(source)
txMat = repeat(tx, Nt, 1)
tyMat = repeat(ty, Nt, 1)
dx = repeat(target[1, :], 1, Ns) - repeat(source[1, 1:end-1]', Nt, 1)
dy = repeat(target[2, :], 1, Ns) - repeat(source[2, 1:end-1]', Nt, 1)
x1 = dx .* txMat + dy .* tyMat
y = -dx .* tyMat + dy .* txMat
x2 = x1 - repeat(sum(diff(source, dims=2) .* [tx; ty], dims=1), Nt, 1)

edgeInf = doublet_inf.(x1, x2, y)

edgeMat = zeros(size(doubletMat))
edgeMat[:, 1] = -edgeInf[:, 1]
edgeMat[:, end] = edgeInf[:, 1]
A = doubletMat + edgeMat
μ = A \ rhs





nf = (foil._foil' * rotation(-3 * pi / 180)')'

plot(nf[1, :], nf[2, :])
plot!(foil._foil[1, :], foil._foil[2, :])
begin
    θ = pi / 4
    for i = 1:foil.N+1
        # foil.foil = (foil._foil'*rotation(θ))'
        @show foil._foil[:, i]

        foil.foil = ([foil._foil[1, :] .- 0.5 foil._foil[2, :]] * rotation(θ))'
        foil.foil[1, :] .+= 0.5
    end
    plot(foil.foil[1, :], foil.foil[2, :], m=:dot)
end