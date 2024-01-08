# include("../src/BemRom.jl")
using Puffer

using Plots

heave_pitch = deepcopy(defaultDict)
heave_pitch[:N] = 64
heave_pitch[:Nt] = 64
heave_pitch[:Ncycles] = 2
heave_pitch[:f] = 1.0
heave_pitch[:Uinf] = 1
heave_pitch[:kine] = :make_heave_pitch
θ0 = deg2rad(5)
h0 = 0.0
heave_pitch[:motion_parameters] = [h0, θ0]

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
    gif(movie, "./images/handp.gif", fps = 10)Review
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

"""
    spalarts_prune!(wake::Wake, flow::FlowParams, foil::Foil; keep=0)

Perform a modification Spalart's wake pruning procedure for a vortex method simulation.
Spalart, P. R. (1988). Vortex methods for separated flows.
# Arguments
- `wake`: A Wake object representing the vortex wake.
- `flow`: A FlowParams object representing the flow parameters.
- `foil`: A Foil object representing the airfoil.
- `keep`: An optional argument (defaulting to 0) that determines how many of the most recent vortices
          are retained without pruning.

This function modifies the `wake` object in-place. It prunes the vortices by aggregating pairs of vortices 
that satisfy Spalart's criterion and have the same sign of circulation, indicating they were generated 
during the same phase of motion. The function updates the positions, circulation, and velocity fields of
the remaining vortices in the wake.

"""

function spalarts_prune!(wake::Wake, flow::FlowParams, foil::Foil; keep = 0)
    V0 = 10e-4 * flow.Uinf
    D0 = 0.1 * foil.chord

    ds = sqrt.(sum(abs2, wake.xy .- te, dims = 1))
    zs = sqrt.(sum(wake.xy .^ 2, dims = 1))

    mask = abs.(wake.Γ * wake.Γ') .* abs.(zs .- zs') ./
           (abs.(wake.Γ .+ wake.Γ') .* (D0 .+ ds) .^ 1.5 .* (D0 .+ ds') .^ 1.5) .< V0

    k = 2
    num_vorts = length(wake.Γ)
    #the k here in what to keep
    #to save the last half of a cycle of motion keep = flow.N ÷ 2    
    while k < num_vorts - keep
        j = k + 1
        while j < foil.N + 1
            if mask[k, j]
                # only aggregate vortices generated during consistent motion
                if sign(wake.Γ[k]) == sign(wake.Γ[j])
                    wake.xnfoils = length(foils)
                    Ns = [foil.N for foil in foils]
                    N = sum(Ns)
                    Ns = [0 cumsum(Ns)...]
                    doubletMat = zeros(N, N)
                    sourceMat = zeros(N, N)
                    edgeMat = zeros(N, N)
                    
                    buffers = zeros(nfoils, foils[1].N)
                    
                    allcols = [foil.col for foil in foils ]
            j += 1
        end
        k += 1
    end

    keepers = findall(x -> x != 0.0, wake.Γ)
    wake.Γ = wake.Γ[keepers]
    wake.xy = wake.xy[:, keepers]
    wake.uv = wake.uv[:, keepers]
    nothing
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
wake = deepcopy(wake2)

wake2 = deepcopy(wake)

#Merge the 2 vortices into the one at k
#Check possibility of merging the last vortex with the closest 20 vortices
gamma_j = curfield.tev[1].s
d_j = sqrt((curfield.tev[1].x - surf_locx)^2 + (curfield.tev[1].z - surf_locz)^2)
z_j = sqrt(curfield.tev[1].x^2 + curfield.tev[1].z^2)

for i in 2:20
    gamma_k = curfield.tev[i].s
    d_k = sqrt((curfield.tev[i].x - surf_locx)^2 + (curfield.tev[i].z - surf_locz)^2)
    z_k = sqrt(curfield.tev[i].x^2 + curfield.tev[i].z^2)

    fact = abs(gamma_j * gamma_k) * abs(z_j - z_k) /
           (abs(gamma_j + gamma_k) * (D0 + d_j)^1.5 * (D0 + d_k)^1.5)

    if fact < V0
        #Merge the 2 vortices into the one at k
        curfield.tev[i].x = (abs(gamma_j) * curfield.tev[1].x +
                             abs(gamma_k) * curfield.tev[i].x) /
                            (abs(gamma_j + gamma_k))
        curfield.tev[i].z = (abs(gamma_j) * curfield.tev[1].z +
                             abs(gamma_k) * curfield.tev[i].z) /
                            (abs(gamma_j + gamma_k))
        curfield.tev[i].s += curfield.tev[1].s

        popfirst!(curfield.tev)

        break
    end
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
turn = pi / 7

"""

nfoils = length(foils)
Ns = [foil.N for foil in foils]
N = sum(Ns)
Ns = [0 cumsum(Ns)...]
doubletMat = zeros(N, N)
sourceMat = zeros(N, N)
edgeMat = zeros(N, N)

buffers = zeros(nfoils, N)


allpanels = hcat([f.foil for f in foils]...)
alledges = hcat([f.edge for f in foils]...)
x1, x2, y = panel_frame(allcols, allpanels)
ymask = abs.(y) .> ϵ
y = y .* ymask
doublet_inf.(x1, x2,y)
source_inf.(x1, x2, y)

x1, x2, y = panel_frame(allcols, alledges)
edgeInf = doublet_inf.(x1, x2, y)

x1, x2, y = panel_frame(allcols, foils[1].edge)
edgeInf = doublet_inf.(x1, x2, y)




nothing
"""



V0 = 10e-4 * flow.Uinf
D0 = 0.1 * foils[1].chord
te = [foils[1].foil[1,1] 0.0]'


mask = abs.(wake.Γ * wake.Γ') .* abs.(zs .- zs') ./
       (abs.(wake.Γ .+ wake.Γ') .* (D0 .+ ds) .^ 1.5 .* (D0 .+ ds') .^ 1.5) .< V0

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
keepers = findall(x -> x != 0.0, wake.Γ)