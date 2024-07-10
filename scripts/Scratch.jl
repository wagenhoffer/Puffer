# include("../src/BemRom.jl")
using Puffer

using Plots

index = 42
motion = motions[index]


θ = motion[4]
h = motion[3]
st = motion[5]
heave_pitch = deepcopy(defaultDict)
heave_pitch[:N] = 64
heave_pitch[:Nt] = 100
heave_pitch[:Ncycles] = 6
heave_pitch[:f] = f
heave_pitch[:Uinf] = 1
heave_pitch[:kine] = :make_heave_pitch
heave_pitch[:ψ] = -pi/2
# θ0 = -0.0#deg2rad(5)
# h0 = 0.0
heave_pitch[:motion_parameters] = [h, θ]

foil, flow = init_params(; heave_pitch...)
wake = Wake(foil)

begin
    foil, flow = init_params(; heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        plot(foil, wake)            
    end
    gif(movie, "./images/handp.gif", fps = 30)
    # plot(foil, wake)
end


begin
    thetas = zeros(flow.N)
    for i = 1:flow.N 
        h = foil.kine[1](foil.f, i * flow.Δt)
        thetas[i] = foil.kine[2](foil.f, i * flow.Δt, foil.ψ)
        
    end
    @show thetas[1],thetas[end]
    plot(thetas)


end
begin
    Nt = N = 64
    moored = deepcopy(defaultDict)
    moored[:Nt] = Nt
    moored[:N] = N

    moored[:Ncycles] = 5
    moored[:f] = 0.5
    moored[:Uinf] = 1.0
    moored[:kine] = :make_heave_pitch
    moored[:pivot] = 0.25
    moored[:thick] = 0.001
    moored[:foil_type] = :make_teardrop
    θ0 = deg2rad(0)
    h0 = 0.01
    moored[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; moored...)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0,
            maximum(foil.foil[1, :]) + foil.chord * 2)
        #ZOom in on the edge
        # win = (minimum(foil.foil[1, :]') + 3*foil.chord / 4.0, maximum(foil.foil[1, :]) + foil.chord * 0.1)
        # if i%flow.N == 0
        #     spalarts_prune!(wake, flow, foil; keep=flow.N÷2)
        # end
        win = nothing
        plot(foil, wake)        
    end
    gif(movie, "./images/theo.gif", fps = 10)
end

foil, flow, wake, coeffs = run_sim(; moored...)
begin
    Nt = N = 128
    moored = deepcopy(defaultDict)
    moored[:Nt] = Nt
    moored[:N] = N

    moored[:Ncycles] = 5
    moored[:f] = 0.5
    moored[:Uinf] = 1.0
    moored[:kine] = :make_heave_pitch
    moored[:pivot] = 0.25
    moored[:thick] = 0.1
    moored[:foil_type] = :make_teardrop
    θ0 = deg2rad(8)
    h0 = 0.01
    moored[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; moored...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0,
            maximum(foil.foil[1, :]) + foil.chord * 2)
        #ZOom in on the edge
        # win = (minimum(foil.foil[1, :]') + 3*foil.chord / 4.0, maximum(foil.foil[1, :]) + foil.chord * 0.1)
        # if i%flow.N == 0
        #     spalarts_prune!(wake, flow, foil; keep=flow.N÷2)
        # end
        win = nothing
        f = plot(foil, wake; window = win)
        f
    end
    gif(movie, "./images/theo.gif", fps = 10)
end

foil, flow, wake, coeffs = run_sim(; moored...)
begin
    f = plot_current(foil, wake)
    f
    plot!(f, foil.foil[1, :], foil.foil[2, :], marker = :bar, label = "Panels")
    plot!(f, cb = :none)
    plot!(f, showaxis = false)
    plot!(f, size = (600, 200))
    plot!(f, grid = :off)
end

begin
    foil, flow = init_params(; heave_pitch...)
    k = foil.f * foil.chord / flow.Uinf
    @show k
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        phi = get_phi(foil, wake)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)
        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
        coeffs[:, i] = get_performance(foil, flow, p)
        ps[:, i] = p
        if i % flow.N == 0
            spalarts_prune!(wake, flow, foil; keep = flow.N)
        end
    end
    t = range(0, stop = flow.Ncycles * flow.N * flow.Δt, length = flow.Ncycles * flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs[1, start:end], label = "Force", lw = 3, marker = :circle)
    b = plot(t[start:end], coeffs[2, start:end], label = "Lift", lw = 3, marker = :circle)
    c = plot(t[start:end], coeffs[3, start:end], label = "Thrust", lw = 3, marker = :circle)
    d = plot(t[start:end], coeffs[4, start:end], label = "Power", lw = 3, marker = :circle)
    plot(a, b, c, d, layout = (2, 2), legend = :topleft, size = (800, 800))
end

"""
    cycle_averaged(coeffs::Array{Real, 2}, flow::FlowParams, skip_cycles::Int64 = 0, sub_cycles::Int64 = 1)

Calculate the averaged values of the given coefficients over the cycles defined by the given flow parameters.
It can optionally skip a number of initial cycles and further divide each cycle into smaller sub-cycles for more detailed averaging.

# Arguments
- `coeffs`: A 2D array of coefficients to be averaged.
- `flow`: An instance of `FlowParams` struct that defines the flow parameters including the total number of cycles (`Ncycles`) and the number of data points in each cycle (`N`).
- `skip_cycles` (optional, default `0`): The number of initial cycles to skip before beginning the averaging.
- `sub_cycles` (optional, default `1`): The number of sub-cycles into which each cycle is further divided for the averaging process.

# Returns
- `avg_vals`: A 2D array with the averaged values over each (sub-)cycle.
"""
function cycle_averaged(coeffs, flow::FlowParams, skip_cycles = 0, sub_cycles = 1)
    # Assertions for error handling
    @assert skip_cycles>=0 "skip_cycles should be a non-negative integer"
    @assert sub_cycles>0 "sub_cycles should be a positive integer"

    avg_vals = zeros(4, (flow.Ncycles - skip_cycles) * sub_cycles)
    for i in (skip_cycles * sub_cycles + 1):(flow.Ncycles * sub_cycles)
        avg_vals[:, i - skip_cycles * sub_cycles] = sum(coeffs[:,
                ((i - 1) * flow.N ÷ sub_cycles + 1):((i) * flow.N ÷ sub_cycles)],
            dims = 2)
    end
    avg_vals ./= flow.N / sub_cycles
end

# Test cases
let
    flow = FlowParams(10, 100) # 10 cycles, each with 100 data points
    coeffs = rand(4, flow.Ncycles * flow.N) # Random coefficients

    avg_vals = cycle_averaged(coeffs, flow)
    @test size(avg_vals) == (4, 10)

    avg_vals = cycle_averaged(coeffs, flow, 2)
    @test size(avg_vals) == (4, 8)
    
end

begin
    # plot a few snap shot at opposite ends of the motion
    pos = 60
    plot(foil.col[1, :], ps[:, pos] / (0.5 * flow.Uinf^2), label = "start", ylims = (-2, 2))
    plot!(foil.col[1, :],
        ps[:, pos + flow.N ÷ 2] / (0.5 * flow.Uinf^2),
        label = "half",
        ylims = (-2, 2))
end

begin
    #look at the panel velocity for a heaving foil, strictly y-comp
    θ0 = deg2rad(0)
    h0 = 0.1
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; heave_pitch...)
    t = (flow.Δt):(flow.Δt):(flow.N * flow.Ncycles * flow.Δt)
    vx = zeros(flow.N * flow.Ncycles)
    vy = zeros(flow.N * flow.Ncycles)
    for i in 1:(flow.N * flow.Ncycles)
        (foil)(flow)
        get_panel_vels!(foil, flow)
        vx[i] = foil.panel_vel[1, 1]
        vy[i] = foil.panel_vel[2, 1]
    end
    plot(t, vx, label = "Vx")
    plot!(t, vy, label = "Vy")
    vya = zeros(size(t))
    @. vya = 2π * foil.f * h0 * cos(2π * foil.f * t)
    plot!(t, vya, label = "Vya")

    @show sum(abs2, vya - vy)
end
begin
    #look at the panel velocity for a pitching foil
    θ0 = deg2rad(5)
    h0 = 0.0
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; heave_pitch...)
    vx = plot()
    vxs = []
    vy = plot()
    vys = []
    for i in 1:(flow.N)
        (foil)(flow)
        get_panel_vels!(foil, flow)
        plot!(vx, foil.col[1, :], foil.panel_vel[1, :], label = "")
        plot!(vy, foil.col[1, :], foil.panel_vel[2, :], label = "")
        push!(vxs, foil.panel_vel[1, :])
        push!(vys, foil.panel_vel[2, :])
    end
    plot(vx, vy, layout = (1, 2))
end


begin
    #look at the panel normals for a pitching foil
    θ0 = deg2rad(5)
    h0 = 0.0
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; heave_pitch...)
    normies = plot()

    normals = @animate for i in 1:(flow.N)
        (foil)(flow)

        f = quiver(foil.col[1, :],
            foil.col[2, :],
            quiver = (foil.normals[1, :], foil.normals[2, :]),
            label = "")
        f
    end
    gif(normals, "normals.gif", fps = 30)
end
begin
    #look at the panel tangents for a pitching foil
    θ0 = deg2rad(5)
    h0 = 0.0
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; heave_pitch...)
    normies = plot()

    normals = @animate for i in 1:(flow.N)
        (foil)(flow)

        f = quiver(foil.col[1, :],
            foil.col[2, :],
            quiver = (foil.tangents[1, :], foil.tangents[2, :]),
            label = "")
        f
    end
    gif(normals, "tangents.gif", fps = 30)
end

###working on Spalart vortex merge_method

# function merge_vortex(wake::Wake, foil::Foil, flow::FlowParams)
V0 = 10e-4 * flow.Uinf
D0 = 0.1 * foil.chord
#the wake has the structure of 
# n-buffer vortices ; 1st 2nd .... nth released vortices
# for multi-foil sims, skip the first n vortices (2n for lesp)
te = foil.edge[:, end]
Γj = wake.Γ[2]
dj = sqrt.(sum(abs2, wake.xy[:, 2] - te, dims = 2))
yj = sqrt.(sum(abs2, wake.xy[:, 2], dims = 1))

# for i = 3:23
Γk = wake.Γ[i]
dk = sqrt.(sum(abs2, wake.xy[:, i] - te, dims = 2))
yk = sqrt.(sum(abs2, wake.xy[:, i], dims = 1))

factor = abs(Γj * Γk) * abs(yj - yk) / abs(Γj + Γk) / (D0 + dj)^1.5 / (D0 + dk) .^ 1.5

factor < V0
#merge em

begin
    V0 = 10e-4 * flow.Uinf
    D0 = 0.1 * foil.chord
    wake = deepcopy(wake2)

    wake2 = deepcopy(wake)

    ds = sqrt.(sum(abs2, wake.xy .- te, dims = 1))
    zs = sqrt.(sum(wake.xy .^ 2, dims = 1))

    factors = abs.(wake.Γ * wake.Γ') .* abs.(zs .- zs') ./
              (abs.(wake.Γ .+ wake.Γ') .* (D0 .+ ds) .^ 1.5 .* (D0 .+ ds') .^ 1.5)

    mask = factors .< V0

    (sum(mask)) / 2.0
    k = 2
    #the k here in what to keep
    #to save the last half of a cycle of motion keep = flow.N ÷ 2
    keep = 0
    while k < flow.N * flow.Ncycles - keep
        # for j=k+1:64
        j = k + 1
        while j < foil.N + 1
            if mask[k, j]
                # only aggregate vortices generated during consistent motion
                if sign(wake.Γ[k]) == sign(wake.Γ[j])
                    wake.xy[:, j] = (abs(wake.Γ[j]) .* wake.xy[:, j] +
                                     abs(wake.Γ[k]) .* wake.xy[:, k]) ./
                                    (abs(wake.Γ[j] + wake.Γ[k]))
                    wake.Γ[j] += wake.Γ[k]
                    wake.xy[:, k] = [0.0, 0.0]
                    wake.Γ[k] = 0.0
                    mask[k, :] .= 0
                else
                    k += 1
                end
            end
            j += 1
        end
        # @show wake.xy[:,k+1]
        k += 1
        plot_current(foil, wake)
    end
    wakeorig = vortex_to_target(wake2.xy, foil.col, wake2.Γ, flow)
    wakeapprox = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)

    keepers = findall(x -> x != 0.0, wake.Γ)
    wake.Γ = wake.Γ[keepers]
    wake.xy = wake.xy[:, keepers]
    wake.uv = wake.uv[:, keepers]

    @show wakeorig .- wakeapprox

    a = plot_current(foil, wake2)
    plot!(a,
        wake.xy[1, :],
        wake.xy[2, :],
        ms = 5,
        st = :scatter,
        label = "",
        msw = 0,
        marker_z = -wake.Γ,
        color = cgrad(:jet))
    a
end

###MESSING AROUND WITH Frequency while in motion###
begin
    #scripting
    upm = deepcopy(defaultDict)
    upm[:Nt] = 64
    upm[:N] = 64
    upm[:Ncycles] = 3
    upm[:Uinf] = 1.0
    upm[:kine] = :make_ang
    upm[:pivot] = 0.0
    upm[:foil_type] = :make_naca
    upm[:thick] = 0.12
    upm[:f] = 1.0
    upm[:k] = 1.0

    foil, flow = init_params(; upm...)
    wake = Wake(foil)
    Us = zeros(2, flow.Ncycles * flow.N)
    U = (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)

    anim = Animation()
    for i in 1:(flow.Ncycles * flow.N)
        if flow.n != 1
            move_wake!(wake, flow)
            release_vortex!(wake, foil)
        end
        if flow.n % flow.N > flow.N ÷ 4
            foil.f += 0.02 #(rand()[1] -0.5) / 100.
        else
            foil.f -= 0.02 #(rand()[1] -0.5) / 100.
        end
        @show foil.f
        (foil)(flow)
        A, rhs, edge_body = make_infs(foil)
        A[getindex.(A .== diag(A))] .= 0.5
        setσ!(foil, flow)
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs * foil.σs - buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        phi = get_phi(foil, wake)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)

        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
        coeffs[:, i] = get_performance(foil, flow, p)
        ps[:, i] = p

        f = plot_current(foil, wake)
        plot!(f, ylims = (-1, 1))
        plot!(title = "$(U)")
        frame(anim, f)
    end
    gif(anim, "./images/self_prop_ang.gif", fps = 10)
end

upm = deepcopy(defaultDict)
upm[:Nt] = 64
upm[:N] = 64
upm[:Ncycles] = 2
upm[:Uinf] = 1.0
upm[:kine] = :make_heave_pitch
upm[:pivot] = 0.0
upm[:foil_type] = :make_naca
upm[:thick] = 0.12
upm[:f] = 1.0
h0 = 0.0
θ0 = deg2rad(0)
upm[:motion_parameters] = [h0, θ0]

foil, flow = init_params(; upm...)
(foil)(flow)
bearing = sum(foil.normals[:, (foil.N ÷ 2):(foil.N ÷ 2 + 1)], dims = 2) / 2.0
@show bearing
θ0 = atan(bearing[2], bearing[1])
pos = sum(foil.foil[:, (foil.N ÷ 2):(foil.N ÷ 2 + 1)], dims = 2) / 2.0
quiver(foil.foil[1, :],
    foil.foil[2, :],
    quiver = (foil.normals[1, :], foil.normals[2, :]),
    aspect_ratio = :equal)
quiver!(pos[1, :], pos[2, :], quiver = (bearing[1, :], bearing[2, :]))

begin
    a = plot()
    for n in 1:(flow.N)
        bearing = sum(foil.normals[:, (foil.N ÷ 2):(foil.N ÷ 2 + 1)], dims = 2) / 2.0
        θ0 = atan(bearing[2], bearing[1]) - π
        @show θ0, 2 * pi * flow.Δt * n
        rotate_about!(foil, turn - θ0)
        norms!(foil)
        plot!(a, foil.foil[1, :], foil.foil[2, :], aspect_ratio = :equal, label = "")
    end
    a
end

function wake_sym(wake::Wake)    
    @show norm(wake.xy[1,1:2:end] .- wake.xy[1, 2:2:end], Inf)
    @show norm(wake.xy[2,1:2:end] .+ wake.xy[2, 2:2:end], Inf)
    @show norm(wake.Γ[1:2:end] .+ wake.Γ[2:2:end], Inf)
    @show norm(wake.uv[1,1:2:end] .- wake.uv[1, 2:2:end], Inf)
    @show norm(wake.uv[2,1:2:end] .+ wake.uv[2, 2:2:end], Inf)
end

wake_sym(wake)

V0 = 1e-4 * flow.Uinf
D0 = foils[1].chord
te = [foils[1].foil[1,1] 0.0]'




k = 4
num_vorts = length(wake.Γ)
N = sum([f.N for f in foils])
x = wake.xy[1, k:end]
y = wake.xy[2, k:end]
xy = [x y]'
Γ  = wake.Γ[k:end]
ds = sqrt.(sum(abs2, xy .- te, dims = 1))
zs = sqrt.(sum(xy .^ 2, dims = 1))
idx = sortperm(y )
x = x[idx]
y = y[idx]
xy = [x y]'
Γ  = Γ[idx]
ds = ds[idx]
zs = zs[idx]
mask = abs.(Γ * Γ') .* abs.(zs .- zs') ./
       (abs.(Γ .+ Γ') .* (D0 .+ ds) .^ 1.5 .* (D0 .+ ds') .^ 1.5) .< V0

# to save the last half of a cycle of motion keep = flow.N ÷ 2    
while k < num_vorts 
    j = k + 1
    while j < N + 1
        if mask[k, j]
            # only aggregate vortiy = wake.xy[2, 4:end]ces of similar rotation
            if sign(Γ[k]) == sign(Γ[j])
                xy[:, j] = (abs(Γ[j]) .* xy[:, j] +
                                 abs(Γ[k]) .* xy[:, k]) ./
                                (abs(Γ[j] + Γ[k]))
                Γ[j] += Γ[k]
                xy[:, k] = [0.0, 0.0]
                Γ[k] = 0.0
                mask[k, :] .= 0
            else
                k += 1
            end
        end
        j += 1
    end
    k += 1
end

# clean up the wake struct
keepers = findall(x -> x != 0.0, Γ)

plot(xy[1,keepers], xy[2,keepers], ms = 4, st = :scatter, label = "", msw = 1, marker_z = -Γ[keepers], color = cgrad(:coolwarm))
plot!(wake.xy[1, :], wake.xy[2, :], ms = 5, st = :scatter, label = "", msw = 0, marker_z = -wake.Γ, color = cgrad(:coolwarm))


chord = 1 
using Plots

function visualize_schooling(height, length)
    # Define the coordinates of the swimmers in the rhombus
    xs = [0, length/2, length, length/2]
    ys = [height/2, 0, height/2, height]

    # Create a scatter plot of the swimmers
    
 
    p = scatter(xs, ys, ms = 10, st = :scatter, label = "", msw = 1, color = :blue)
    xmid = foil.chord/2.0 
    
    for (x,y) in zip(xs,ys)
        plot!(p, foil._foil[1,:] .+ x .- xmid, foil._foil[2,:] .+ y, color = :black, lw = 2)
    end

    # Connect the swimmers with lines to form the rhombus
    plot!(p, [xs[1], xs[2]], [ys[1], ys[2]], color = :black, lw = 2)
    plot!(p, [xs[2], xs[3]], [ys[2], ys[3]], color = :black, lw = 2)
    plot!(p, [xs[3], xs[4]], [ys[3], ys[4]], color = :black, lw = 2)
    plot!(p, [xs[4], xs[1]], [ys[4], ys[1]], color = :black, lw = 2)

    # Set the aspect ratio to equal for a proper visualization    
    plot!(p, aspect_ratio =:true, xlims = (-0.5, length + 0.5), ylims = (-0.5, height + 0.5))

end

# Example usage
visualize_schooling(0.5, 1.5)



begin
# compare velocity field of a single panel to that of two vortices
xs  = -1:0.1:1
ys  = -1:0.1:1
X   = xs.* ones(size(ys))'
Y   = ones(size(xs)) .* ys'
mu = [0.25 0.5 ]
targets = [X[:] Y[:]]'
source = [-0.25 0 0.25
           0.  0 -0.]
x1, x2, y = panel_frame(targets, source)
nw, nb = size(x1)
lexp = zeros((nw, nb))
texp = zeros((nw, nb))
yc = zeros((nw, nb))
xc = zeros((nw, nb))
β = [-pi ] #[atan.(1, 1)]
β = repeat(β, 1, nw)'
@. lexp = log((x1^2 + y^2) / (x2^2 + y^2)) / (4π)
@. texp = (atan(y, x2) - atan(y, x1)) / (2π)
@. xc = lexp * cos(β) - texp * sin(β)
@. yc = lexp * sin(β) + texp * cos(β)
@. lexp =  (y/(x1^2  + y^2)  - y/(x2^2 + y^2))/2π
@. texp = -(x1/(x1^2 + y^2) - x2/(x2^2 + y^2))/2π
@. xc = lexp * cos(β) - texp * sin(β)
@. yc = lexp * sin(β) + texp * cos(β)
uvP = [xc * mu'  yc * mu']'   

Γs = [-mu[1] mu[1]-mu[2] mu[2]] 
# Γs = [-mu[1] mu[1]] 
# ps = [foil.foil foil.edge]
uvΓ = vortex_to_target(source, targets, Γs, flow)
magΓ = reshape(sqrt.(sum(abs2, uvΓ, dims = 1)), (size(xs,1), size(ys,1)))
magP = reshape(sqrt.(sum(abs2, uvP, dims = 1)), (size(xs,1), size(ys,1)))
du = uvP - uvΓ
d = quiver(targets[1,:], targets[2,:],  quiver = (uvΓ[1,:], uvΓ[2,:]), aspect_ratio = :equal, label="")
e = quiver(targets[1,:], targets[2,:],  quiver = (uvP[1,:], uvP[2,:]), aspect_ratio = :equal, label="")
f = quiver(targets[1,:], targets[2,:],  quiver = (du[1,:], du[2,:]), aspect_ratio = :equal, label="")
a = plot( magΓ', st = :contour, levels = 50, color = :viridis, aspect_ratio = :equal)
b = plot( magP', st = :contour, levels = 50, color = :viridis, aspect_ratio = :equal)
c = plot((magP-magΓ)', st = :contour, levels = 50, color = :viridis, aspect_ratio = :equal)
plot(a,b,c, layout = (1,3), size = (1000, 400))
plot(e, d, f, layout = (1,3), size = (1200, 400))
end



begin
    T = Float32
    ϵ = sqrt(eps(T))
    strouhals = LinRange{T}(0.2, 0.4, 7)
    td = LinRange{Int}(0, 10, 6)
    θs = deg2rad.(td).|> T    
    hs = LinRange{T}(0.0, 0.25, 6)
    # f,a = fna[(θ, h, strou)]
    num_samps = 8

    G = inv(L|>cpu)|>gpu
    begin
        # for h in hs[1:end], θ in θs[2:end], st in strouhals[1:end]
            (θ,h, st) = fna.keys[1]
            f,a = fna[(θ,h, st)]
            test = deepcopy(defaultDict)
            test[:N] = 64
            test[:Nt] = 100
            test[:Ncycles] = 1
            test[:f] = f
            test[:Uinf] = 1
            test[:kine] = :make_heave_pitch

            test[:ψ] = -pi/2
            test[:motion_parameters] = [h, θ]
            foil, flow = init_params(; test...)
            wake = Wake(foil)
            test[:N] = num_samps
            test[:T] = Float32
            foil2, _ = init_params(; test...)
            (foil)(flow)
            (foil2)(flow)
            flow.n -= 1

            ### EXAMPLE OF AN ANIMATION LOOP
            # movie = @animate for i in 1:200
            for i = 1:200
                rhs = time_increment!(flow, foil, wake)
                # Nice steady window for plotting        
                # push!(vels, deepcopy(foil.panel_vel))
                (foil2)(flow)
                flow.n -= 1
                points = zeros(2,100)
                
                sf = sample_field(foil::Foil)                   
                n,t,l = norms(foil.edge)
                # cent  = mean(foil.foil, dims=2)
                te = foil.foil[:,end]
                θedge = atan(t[:,2]...) 
                # sf = wake.xy[:, randperm(size(wake.xy, 2))[1:100]]
                sp = stencil_points(sf;ϵ=ϵ)
                points = sf #.+ foil.edge[:,end] .+flow.Δt
                test = hcat(sf,sp)
                phi = phi_field(test,foil)
                (ori, xp, xm, yp, ym) = de_stencil(phi')
                if sum((-4*ori + xp + xm + yp + ym) .> ϵ) >1
                    println("Error in stencil")
                end
                uv = rotation(-θedge)*b2f(points, foil, flow)
                
                get_panel_vels!(foil2,flow)
                # vortex_to_target(sources, targets, Γs, flow)   
                pos = deepcopy(foil2.col')
                zeropos = deepcopy(foil2.col') .- minimum(pos,dims=1).*[1 0]
                # pos[:,1] .+= minimum(pos[:,1])
                # @show errorL2(dx,points[1,:]' .- pos[:,1])
                iv = vortex_to_target(wake.xy, foil2.col, wake.Γ, flow)
                be = 7
                pts = collect(vcat(be:be:32,32+be-1:be:64))
                iv2 = vortex_to_target(wake.xy, foil.col[:,pts], wake.Γ, flow)

                cont = reshape([zeropos foil2.normals'  iv' foil2.panel_vel' ], (num_samps,2,4,1))

                β = B_DNN.layers[2](upconv(cont|>gpu))
                # B = B_DNN.layers[2:3](upconv(cont|>gpu))|>cpu
                # excont = vcat(cont[end-1:end,:,:,:],cont,cont[1:2,:,:,:])
                # β1 = B_DNN.layers[1:2](excont|>gpu)
                # B1 = B_DNN.layers[1:3](excont|>gpu)|>cpu
                ν = G*β
            
                # errorL2(μAE[:decoder](ν)|>cpu, foil.μs)
                cont = reshape([pos foil2.normals'  iv' foil2.panel_vel' ], (num_samps,2,4))
                image = cat(cont|>mp.dev, cat(ν, β, dims=2), dims=3)

                # image = cat(cont|>mp.dev, cat(foil.μs, foil.σs   , dims=2), dims=3)
                
                plot(foil, wake)   
                # plot!(points[1,:], points[2,:], seriestype = :scatter)                         
            end
            gif(movie, "./images/full.gif", fps = 30)
            # plot(foil, wake)
        end
    end


fst, lst = foil.panel_lengths[1],foil.panel_lengths[end]
sigf, sigl = foil.σs[1],foil.σs[end]
μf, μl = foil.μs[1],foil.μs[end]
spot =( sigf*log(fst/2) + sigl*log(lst/2))/2π


wake.xy[:,1]  = foil.foil[:,1]
sten = hcat(rotation(θedge)*(wake.xy.-te), 
    stencil_points(rotation(θedge)*(wake.xy.-te );ϵ=ϵ)) .+[foil.chord 0]'
c,px,mx,py,my= de_stencil(Onet(vcat(ν,β), sten|>gpu)')   