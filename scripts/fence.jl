include("../src/BemRom.jl")
using Plots
using RegionTrees
import RegionTrees: AbstractRefinery, needs_refinement, refine_data
using IntervalSets


heave_pitch = deepcopy(defaultDict)
heave_pitch[:N] = 50
heave_pitch[:Nt] = 64
heave_pitch[:Ncycles] = 1
heave_pitch[:f] = 1.0
heave_pitch[:Uinf] = 1.0
heave_pitch[:kine] = :make_heave_pitch
θ0 = deg2rad(10)
h0 = 0.0
heave_pitch[:motion_parameters] = [h0, θ0]

begin
    foil, flow = init_params(;heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    
     for i in 1:flow.Ncycles*flow.N*1.75
        time_increment!(flow, foil, wake)                    
    end
    plot_current(foil,wake)
end
###MOVE the wake to in front of the foil
wake.xy[1,:] .-= 2.1
plot_current(foil,wake)
######## STARTING TO IMPLEMENT A FENCE ######## 
# Modify move_wake! to have a flag for fence ->flowParams

begin  
    time_increment!(flow, foil, wake)
    dest = wake.xy + wake.uv .* flow.Δt
    dir  = dest .- wake.xy
end

# Place the vortices into a large container with origin above destination
vortices = vcat(wake.xy,dest)
plot(wake.xy[1,:], wake.xy[2,:], st=:scatter, color=:blue, ms=2)
plot!(dest[1,:], dest[2,:], st=:scatter, color=:red, ms=1)

# We need to construct a RegionTrees to encapsulate the flow field. 
xmin   = minimum([wake.xy[1,:] dest[1,:]])
xmax   = maximum([wake.xy[1,:] dest[1,:]])
ymin   = minimum([wake.xy[2,:] dest[2,:]])
ymax   = maximum([wake.xy[2,:] dest[2,:]])
xcom   = (xmax + xmin) /2.0
ycom   = (ymax + ymin) /2.0
width  = xmax -xmin
height = ymax - ymin

#for mulitple foils
panels = zeros(2,length(foils)*(foils[1].N+1))
for (i,f) in enumerate(foils)
    panels[:,(i-1)*f.N+1:i*f.N] = f.foil
end
#single foil
panels = foil.foil




struct MyRefinery <: AbstractRefinery
    tolerance::Float64
end

# These two methods are all we need to implement
function needs_refinement(r::MyRefinery, cell)
    maximum(cell.boundary.widths) > r.tolerance
end

function refine_data(r::MyRefinery, cell::Cell, indices)

    curr = child_boundary(cell,indices)
    intervals = [ClosedInterval(curr.origin[i],curr.origin[i]+curr.widths[i]) for i=1:2]
    out = zeros(4,0)
    if !isempty(cell.data)
        for i=1:size(cell.data)[2] 
            if cell.data[1,i] in intervals[1] && cell.data[2,i] in intervals[2]
                out = hcat(out, cell.data[:,i])
            end
        end
    end
    out
end
r = MyRefinery(maximum(abs.(dir))*10.0)
root = Cell(SVector(xmin, ymin), SVector(width,height), vortices)
adaptivesampling!(root, r)


#looks at some metrics of the tree, how many levels and how many vortices per leaf
lcount = 0
vcount = 0 
for leaf in allleaves(root)
    # @show leaf.data|>size 
    lcount += 1
    vcount += size(leaf.data)[2]
end
lcount
vcount
@assert size(root.data)[2] == length(wake.Γ)
@assert vcount == length(wake.Γ)

begin
    plt = plot()
    
    for leaf in allleaves(root)    
            v = hcat(collect(vertices(leaf.boundary))...)
            plot!(plt, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], label="")
            plot!(plt, leaf.data[1,:],leaf.data[2,:], st=:scatter, label="")
            # plot!(plt, leaf.data[3,:],leaf.data[4,:], st=:scatter, label="")
            # plt      
    end
    plt
end
#test how many leaves does a vortex travel by a panel
center(cell::Cell)  = cell.boundary.origin .+ cell.boundary.widths ./2.0
panels[:, 1]
for i in 1:size(panels)[2]
    node = findleaf(root, panels[:,i])
    if !isempty(node.data)
        cent = center(node)
        plot!(plt, [cent[1]], [cent[2]], st=:scatter, marker=:star,ms=5., label="")        
    end 
end
plt

plot!(plt,foil.foil[1,:],foil.foil[2,:],marker=:hex,ms=.5, label="")

function find_intersection(p0, p1, p2, p3)
    """ p0->p1 and p2->p3 
        returns intersection_point of the lines """
    s10_x = p1[1] - p0[1]
    s10_y = p1[2] - p0[2]
    s32_x = p3[1] - p2[1]
    s32_y = p3[2] - p2[2]
    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0
        return nothing # collinear
    end
    denom_is_positive = denom > 0
    s02_x = p0[1] - p2[1]
    s02_y = p0[2] - p2[2]
    s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == denom_is_positive
        return nothing # no collision
    end
    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denom_is_positive
        return nothing # no collision
    end
    if (s_numer > denom) == denom_is_positive || (t_numer > denom) == denom_is_positive
        return nothing # no collision
    end
    # collision detected
    t = t_numer / denom
    intersection_point = [p0[1] + t * s10_x, p0[2] + t * s10_y]
    return intersection_point
end
#testing on find_intersection
@assert [0.0, 0.5] == find_intersection([0,0],[0,1], [-0.5, 0.5], [0.5, 0.5] )
@assert [0.0, 0.5] == find_intersection([-0.5, 0.5], [0.5, 0.5], [0,0],[0,1])
@assert [0.5, 0.5] == find_intersection([0.0, 0.0],  [1.0, 1.0], [0,1],[1,0])
@assert [0.0, -0.5] == find_intersection([0,0],[0,-1], [-0.5, -0.5], [0.5, -0.5] )
@assert [0.0, -0.5] == find_intersection([-0.5, -0.5], [0.5, -0.5], [0,0],[0,-1])
@assert [-0.5, -0.5] == find_intersection([0.0, 0.0],  [-1.0, -1.0], [0,-1],[-1,0])
@assert [0,0] == find_intersection([0.0, 0.0],  [0, 1.0], [0,0], [-1,0])
@assert nothing == find_intersection([0.0, 0.0],  [0, 1.0], [1,0], [2,0])

#run through the control block above, but now look for intersections
plt = plot()
for i in 1:size(foil.foil)[2]-1
    node = findleaf(root, foil.foil[:,i])
    if !isempty(node.data)
        node.data[3:4,:] = node.data[1:2,:] .+ [0.05,0]
        # start by only looking to the next panel, filter to yield indexing for multiple intersections
        sects = filter(y-> !isnothing(y),
                    map( x -> find_intersection(foil.foil[:,i], foil.foil[:,i+1], x[1], x[2]),
                     [(node.data[1:2,i],node.data[3:4,i]) for i =1:size(node.data)[2]]))
        if !isempty(sects)
            @show(sects)
            plot!(plt, [sects[1][1]], [sects[1][2]], st=:scatter, marker=:hex, ms=5., label="") 
            v = hcat(collect(vertices(node.boundary))...)
            plot!(plt, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], label="")
            plot!(plt, node.data[1,:],node.data[2,:], st=:scatter, label="origin")
            plot!(plt, node.data[3,:],node.data[4,:], st=:scatter, label="end")
            plot!(plt,[foil.foil[:,i] foil.foil[:,i+1]][1,:],[foil.foil[:,i] foil.foil[:,i+1]][2,:], label="" )
            plot!(plt,)
            vdist = node.data[3:4,:] - node.data[1:2,:] 
            # s2i   = sects[1] - node.data[1:2,:]
            s2i   = (sects[1] + foil.normals[:,i] * flow.δ) - node.data[1:2,:]
            rest  = norm(vdist) - norm(s2i)
            @show rest
            node.data[3:4] =sects[1] + foil.tangents[:,i] *rest
            plot!(plt, node.data[3,:],node.data[4,:], st=:scatter, label="new end")

        end
   
    end 
end
plt