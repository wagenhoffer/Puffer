using Puffer
using Plots

function create_foils(num_foils, starting_positions)
    # Ensure starting_positions has correct dimensions
    if size(starting_positions) != (2, num_foils)
        error("starting_positions must be a 2xN array, where N is the number of foils")
    end

    pos1 = deepcopy(defaultDict)
    pos1[:kine] = :make_heave_pitch
    pos1[:motion_parameters] = [0.1, 0.0]
    foil1, flow = init_params(; pos1...)

    foils = Vector{typeof(foil1)}(undef, 4)
    for i in 1:num_foils
        foil = deepcopy(foil1)
        foil.foil[1, :] .+= starting_positions[1, i] * foil1.chord
        foil.foil[2, :] .+= starting_positions[2, i]
        foil.LE = [minimum(foil.foil[1, :]), foil.foil[2, (foil.N ÷ 2 + 1)]]
        norms!(foil)
        set_collocation!(foil)
        move_edge!(foil, flow;startup=true)
        foil.edge[1, end] = 2.0 * foil.edge[1, 2] - foil.edge[1, 1]
        foils[i] =  foil
    end

    foils, flow
end



begin
    num_foils = 4
    starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    foils, flow = create_foils(num_foils, starting_positions)
    wake = Wake(foils)
    (foils)(flow)
end

for (i, foil) in enumerate(foils)
   @show foil.edge[:, end]
end


begin 
    steps = flow.N * flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake, old_mus, old_phis)
    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake, old_mus, old_phis)        
        f1TEx = foils[1].foil[1, end] .+ (-0.25, 0.25)
        f1TEy = foils[1].foil[2, end] .+ (-0.25, 0.25)
        plot(foils, wake;xlims=f1TEx, ylims= f1TEy)        
        # plot!(foils[1].edge[1,:],foils[1].edge[2,:], color = :green, lw = 2,label="")
    end
    gif(movie, "newMulti.gif", fps = 30)
end



begin
    # do it with gory detal
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
        f = plot(foils, wake)
        # wake.xy = sdf_fence(wake, foils, flow; mask = [0, 1, 0, 0] .|> Bool)
       
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
