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
        dest = wake.xy + wake.uv .* flow.Δt
                         
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
    out = zeros(Int64,0)
    if !isempty(cell.data)
        for i=1:length(cell.data) 
            if wake.xy[1,cell.data[i]] in intervals[1] && wake.xy[2,cell.data[i]] in intervals[2]
                push!(out, cell.data[i])
            end
        end
    end
    out
end
r = MyRefinery(maximum(abs.(dir))*10.0)
root = Cell(SVector(xmin, ymin), SVector(width,height), collect(1:length(wake.Γ)))
adaptivesampling!(root, r)


#looks at some metrics of the tree, how many levels and how many vortices per leaf
lcount = 0
vcount = 0 
eles = []
for leaf in allleaves(root)
    # @show leaf.data|>size 
    lcount += 1
    vcount += length(leaf.data)
    push!(eles, leaf.data...)
end
lcount
vcount
@assert length(root.data) == length(wake.Γ)
@assert vcount == length(wake.Γ)
@assert sort(eles) == 1:length(wake.Γ)
begin
    plt = plot()
    
    for leaf in allleaves(root)    
            v = hcat(collect(vertices(leaf.boundary))...)
            plot!(plt, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], label="")
            plot!(plt, wake.xy[1,leaf.data], wake.xy[2,leaf.data], st=:scatter, label="")
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
begin
    plt = plot()
    dest = wake.xy .+ [0.1,0]
    eles = []
    for i in 1:size(foil.foil)[2]-1
        #make a collect then iterate over it later
        node = findleaf(root, foil.foil[:,i])
        v = hcat(collect(vertices(node.boundary))...)
        plot!(plt, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], label="",lw=2)
        plot!(plt, wake.xy[1,node.data], wake.xy[2,node.data], st=:scatter, label="")
        push!(eles, node.data...)
        if length(node.data)>0
            #cookup a destination
    
            # start by only looking to the next panel, filter to yield indexing for multiple intersections
            sects = map( x -> find_intersection(foil.foil[:,i], foil.foil[:,i+1], x[1], x[2]),
                        [[wake.xy[:,node.data], dest[:,node.data]] for i =1:length(node.data)])
            for (ind, sect) in enumerate(sects)                
                if !isnothing(sect)
                    plot!(plt, [sects[1][1]], [sects[1][2]], st=:scatter, marker=:hex, ms=5., label="") 
                    v = hcat(collect(vertices(node.boundary))...)
                    plot!(plt, v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], label="")
                    plot!(plt, wake.xy[1,node.data], wake.xy[2,node.data], st=:scatter, label="origin")
                    plot!(plt, dest[1,node.data], dest[2,node.data], st=:scatter, label="end")
                    plot!(plt,[foil.foil[:,i] foil.foil[:,i+1]][1,:],[foil.foil[:,i] foil.foil[:,i+1]][2,:], label="" )
                    plot!(plt,)
                    vdist = wake.xy[:,node.data[ind]] - dest[:,node.data[ind]] 
                    s2i   = sects[1] - wake.xy[:,node.data[ind]]
                    # s2i   = (sect + foil.normals[:,i] * flow.δ) - dest[:,node.data[ind]] 
                    rest  = abs(norm(vdist) - norm(s2i))
                    @show norm(vdist), norm(s2i), rest
                    dest[:,node.data[ind]] = sect + foil.tangents[:,i+1] *rest
                    plot!(plt, [dest[1,node.data]], [dest[2,node.data]], st=:scatter,marker=:hex,color=:red, label="new end")
                end

            end
    
        end 
    end
    plot!(plt,foil.foil[1,:],foil.foil[2,:],label="")
    plt
end



function fence!(dest,tree::Cell,foil::Foil, wake::Wake)
    for i in 1:size(foil.foil)[2]-1
        #TODO: make a collect then iterate over it later
        node = findleaf(tree, foil.foil[:,i])       
        if length(node.data)>0    
            # start by only looking to the next panel, filter to yield indexing for multiple intersections
            sects = map( x -> find_intersection(foil.foil[:,i], foil.foil[:,i+1], x[1], x[2]),
                        [[wake.xy[:,node.data], dest[:,node.data]] for i =1:length(node.data)])
            for (ind, sect) in enumerate(sects)                
                if !isnothing(sect)
                    vdist = wake.xy[:,node.data[ind]] - dest[:,node.data[ind]] 
                    tointersetion   = sects[1] - wake.xy[:,node.data[ind]]                    
                    rest  = abs(norm(vdist) - norm(tointersetion))                    
                    dest[:,node.data[ind]] = sect + foil.tangents[:,i+1] *rest                    
                end
            end    
        end 
    end
    nothing    
end

function make_qt(wake::Wake)
    xmin   = minimum([wake.xy[1,:] dest[1,:]])
    ymin   = minimum([wake.xy[2,:] dest[2,:]])
    width  = xmax -xmin
    height = ymax - ymin
    root = Cell(SVector(xmin, ymin), SVector(width,height), collect(1:length(wake.Γ)))
    adaptivesampling!(root, r)
    root
end

fence!(dest,root, foil,wake)


begin
    foil, flow = init_params(;heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    
     for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)   
        dest = wake.xy + wake.uv .* flow.Δt
        tree = make_qt(wake)                
        fence!(dest,tree,foil,wake)
    end
    plot_current(foil,wake)
    wake.xy[1,:] .-=2.0
    movie = @animate for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)   
        dest = wake.xy + wake.uv .* flow.Δt
        tree = make_qt(wake)                
        # fence!(dest,tree,foil,wake)
        f = plot_current(foil,wake)
        f
    end
    gif(movie,"fence.gif")
end