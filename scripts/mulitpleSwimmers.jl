using Puffer
using Plots

begin
    pos1 = deepcopy(defaultDict)
    pos1[:kine] = :make_heave_pitch
    pos1[:motion_parameters] = [0.0, -π / 20]
    foil1, flow = init_params(; pos1...)
    # copy the first foil
    foil2 = deepcopy(foil1)
    foil3 = deepcopy(foil1)
    foil4 = deepcopy(foil1)
    #alter the second foil - absolute position
    foil2.foil[1, :] .+= 2.0 * foil1.chord
    foil2.LE = [minimum(foil2.foil[1, :]), 0.0]
    norms!(foil2)
    set_collocation!(foil2)
    move_edge!(foil2, flow)
    foil2.edge[1, end] = 2.0 * foil2.edge[1, 2] - foil2.edge[1, 1]

    foil3.foil[1, :] .+= 1.0 * foil1.chord
    foil3.foil[2, :] .+= 1.0
    foil3.LE = [minimum(foil3.foil[1, :]), foil3.foil[2, (foil3.N ÷ 2 + 1)]]
    norms!(foil3)
    set_collocation!(foil3)
    move_edge!(foil3, flow)
    foil3.edge[1, end] = 2.0 * foil3.edge[1, 2] - foil3.edge[1, 1]

    foil4.foil[1, :] .+= 1.0 * foil1.chord
    foil4.foil[2, :] .-= 1.0
    foil4.LE = [minimum(foil4.foil[1, :]), foil4.foil[2, (foil4.N ÷ 2 + 1)]]
    norms!(foil4)
    set_collocation!(foil4)
    move_edge!(foil4, flow)
    foil4.edge[1, end] = 2.0 * foil4.edge[1, 2] - foil4.edge[1, 1]

    #vector of foils
    foils = [foil1, foil2, foil3, foil4]
    @show typeof(foils)

    flow.Ncycles = 5
    flow.N = 50
    flow.Uinf = 2.0

    #wake init
    N = length(foils)
    xy = zeros(2, N)
    Γs = zeros(N)
    uv = zeros(2, N)
    for (i, foil) in enumerate(foils)
        xy[:, i] = foil.edge[:, end]
    end
    wake = Wake(foils)
    do_kinematics!(foils, flow)
end

function cancel_buffer_Γ!(wake::Wake, foils::Vector{Foil{T}}) where {T <: Real}
    for (i, foil) in enumerate(foils)
        wake.xy[:, i] = foil.edge[:, 2]
        wake.Γ[i] = -foil.μ_edge[end]
    end
end

plt = plot()
for f in foils
    plot!(plt, f.foil[1, :], f.foil[2, :])
end
plt

begin
    steps = flow.N * flow.Ncycles

    movie = @animate for t in 1:steps
        A, rhs, edge_body = make_infs(foils) #̌#CHECK
        [setσ!(foil, flow) for foil in foils] #CHECK
        σs = [] #CHECK
        buff = []
        for (i, foil) in enumerate(foils)
            foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
            normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
            foil.σs -= normal_wake_ind[:]
            push!(σs, foil.σs...)
            push!(buff, (edge_body[:, i] * foil.μ_edge[1])...)
        end
        μs = A \ (-rhs * σs - buff)
        for (i, foil) in enumerate(foils)
            foil.μs = μs[((i - 1) * foil.N + 1):(i * foil.N)]
        end
        set_edge_strength!.(foils)
        cancel_buffer_Γ!(wake, foils)
        wake_self_vel!(wake, flow)
        totalN = sum(foil.N for foil in foils)
        phis = zeros(totalN)
        ps = zeros(totalN)
        old_mus = zeros(3, totalN)
        old_phis = zeros(3, totalN)
        coeffs = zeros(length(foils), 4, steps)
        for (i, foil) in enumerate(foils)
            body_to_wake!(wake, foil, flow)
            phi = get_phi(foil, wake)
            phis[((i - 1) * foils[i].N + 1):(i * foils[i].N)] = phi
            p = panel_pressure(foil,
                flow,
                old_mus[:, ((i - 1) * foils[i].N + 1):(i * foils[i].N)],
                old_phis[:, ((i - 1) * foils[i].N + 1):(i * foils[i].N)],
                phi)
            ps[((i - 1) * foils[i].N + 1):(i * foils[i].N)] = p
            coeffs[i, :, 1] .= get_performance(foil, flow, p)
        end
        old_mus = [μs'; old_mus[1:2, :]]
        old_phis = [phis'; old_phis[1:2, :]]
        wake.xy = sdf_fence(wake, foils, flow; mask = [0, 1, 0, 0] .|> Bool)
        f = plot_current(foils, wake;)
        # move_wake!(wake, flow)
        for foil in foils
            release_vortex!(wake, foil)
        end
        do_kinematics!(foils, flow)
       
        f
    end
    gif(movie, "multi.gif", fps = 30)
end

function time_increment!(flow::FlowParams, foils::Vector{Foil}, wake::Wake)
    A, rhs, edge_body = make_infs(foils) #̌#CHECK
    [setσ!(foil, flow) for foil in foils] #CHECK
    σs = [] #CHECK
    buff = []
    for (i, foil) in enumerate(foils)
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        foil.σs -= normal_wake_ind[:]
        push!(σs, foil.σs...)
        push!(buff, (edge_body[:, i] * foil.μ_edge[1])...)
    end
    μs = A \ (-rhs * σs - buff)
    for (i, foil) in enumerate(foils)
        foil.μs = μs[((i - 1) * foil.N + 1):(i * foil.N)]
    end
    set_edge_strength!.(foils)
    cancel_buffer_Γ!(wake, foils)
    wake_self_vel!(wake, flow)
    totalN = sum(foil.N for foil in foils)
    phis = zeros(totalN)
    ps = zeros(totalN)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    for (i, foil) in enumerate(foils)
        body_to_wake!(wake, foil, flow)
        phi = get_phi(foil, wake)
        phis[((i - 1) * foils[i].N + 1):(i * foils[i].N)] = phi
        p = panel_pressure(foil,
            flow,
            old_mus[:, ((i - 1) * foils[i].N + 1):(i * foils[i].N)],
            old_phis[:, ((i - 1) * foils[i].N + 1):(i * foils[i].N)],
            phi)
        ps[((i - 1) * foils[i].N + 1):(i * foils[i].N)] = p
        coeffs[i, :, 1] .= get_performance(foil, flow, p)
    end
    old_mus = [μs'; old_mus[1:2, :]]
    old_phis = [phis'; old_phis[1:2, :]]

    # move_wake!(wake, flow)
    for foil in foils
        release_vortex!(wake, foil)
    end
    do_kinematics!(foils, flow)
    nothing
end

######## STARTING TO IMPLEMENT A FENCE ######## 
# Modify move_wake! to have a flag for fence ->flowParams
using RegionTrees
begin
    dest = wake.xy + wake.uv .* flow.Δt
    dir = dest .- wake.xy
end
vortices = vcat(wake.xy, dest)
plot(wake.xy[1, :], wake.xy[2, :], st = :scatter, color = :blue, ms = 2)
plot!(dest[1, :], dest[2, :], st = :scatter, color = :red, ms = 1)

# We need to construct a RegionTrees to encapsulate the flow field. 
xmin = minimum([wake.xy[1, :] dest[1, :]])
xmax = maximum([wake.xy[1, :] dest[1, :]])
ymin = minimum([wake.xy[2, :] dest[2, :]])
ymax = maximum([wake.xy[2, :] dest[2, :]])
xcom = (xmax + xmin) / 2.0
ycom = (ymax + ymin) / 2.0
width = xmax - xmin
height = ymax - ymin
panels = zeros(2, length(foils) * (foils[1].N + 1))
for (i, f) in enumerate(foils)
    panels[:, ((i - 1) * f.N + 1):(i * f.N)] = f.foil
end
root = Cell(SVector(xmin, ymin), SVector(width, height), vortices)

import RegionTrees: AbstractRefinery, needs_refinement, refine_data
using IntervalSets

struct MyRefinery <: AbstractRefinery
    tolerance::Float64
end

# These two methods are all we need to implement
function needs_refinement(r::MyRefinery, cell)
    maximum(cell.boundary.widths) > r.tolerance
end
function refine_data(r::MyRefinery, cell::Cell, indices)
    # boundary = child_boundary(cell, indices)
    # "child with widths: $(boundary.widths)"
    curr = child_boundary(root, indices)
    intervals = [ClosedInterval(curr.origin[i], curr.origin[i] + curr.widths[i])
                 for i in 1:2]
    hcat([vortices[:, i] for i in 1:size(vortices)[2] if
              vortices[1, i] in intervals[1] && vortices[2, i] in intervals[2]]...)
end
r = MyRefinery(maximum(abs.(dir)) * 40.0)

adaptivesampling!(root, r)

count = 0
for leaf in allleaves(root)
    @show leaf.data |> size
    count += 1
end
count

panels[:, 1]
node = findleaf(root, panels[:, 1])

plt = plot(xlim = (xmin, xmax), ylim = (ymin, ymax), legend = nothing)
for leaf in allleaves(root)
    v = hcat(collect(vertices(leaf.boundary))...)
    @show getindex(leaf)
    plot!(plt, v[1, [1, 2, 4, 3, 1]], v[2, [1, 2, 4, 3, 1]])
end
plt

function plot_ind(plt, index)
    v = hcat(collect(vertices(getindex(root, [index])[1].boundary))...)
    plot!(plt, v[1, [1, 2, 4, 3, 1]], v[2, [1, 2, 4, 3, 1]], color = :red, lw = 5)
    plt
end

function get_cell(index)
    indices = []
    while index != 0
        q, r = index ÷ 4, index % 4
        push!(indices, [q, r])
        index = (index - 1) ÷ 4
    end
    indices
end

curr = child_boundary(root, 1)
intervals = [ClosedInterval(curr.origin[i], curr.origin[i] + curr.widths[i]) for i in 1:2]
hcat([vortices[:, i] for i in 1:size(vortices)[2] if
          vortices[1, i] in intervals[1] && vortices[2, i] in intervals[2]]...)
for i in 1:size(vortices)[2]
    all(vortices[j, :] in intervals for j in 1:2)
end
filter(pos -> pos[1] in intervals[1] && pos[2] in intervals[2], vortices[1:2, :])
v_f = hcat([vortices[:, i] for i in 1:size(vortices)[2] if
                vortices[1, i] in intervals[1] && vortices[2, i] in intervals[2]]...)
