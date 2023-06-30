include("../src/BemRom.jl")
using Plots
# using RegionTrees
# import RegionTrees: AbstractRefinery, needs_refinement, refine_data
# using IntervalSets
using Dierckx


function make_splines(foil, N)
    top = Spline1D(foil[1,N÷2 + 1:end], foil[2,N÷2+1:end])   
    bottom = Spline1D(foil[1,N÷2+1:-1:1], foil[2,N÷2+1:-1:1])
    top,bottom 
end

function make_splines(foil)
    # assumes traveling to the left
    _ , N  = findmin(foil[1,:])
    top    = Spline1D(foil[1,N:end],  foil[2,N:end])   
    bottom = Spline1D(foil[1,N:-1:1], foil[2,N:-1:1])
    top,bottom 
end

make_splines(foil::Foil) = make_splines(foil.foil)


function foilsdf(foil::Foil)
    #make splines
    top, bottom = make_splines(foil)
    """
    isinside(xy)
    xy is a pair to test if inside of the splines constructed
    should allow for passing of vectors via 
    
    """
    function isinside(xy)
        #make a range of acceptable xs        
        if minimum(foil.foil[1,:]) <= xy[1] <= maximum(foil.foil[1,:])
            #find the smallest distance in y if accX is true
            thetop = top(xy[1])
            thebot = bottom(xy[1])
            mid = (thetop + thebot)./2.0
            if xy[2] <= mid
                return thebot  - xy[2] 
            else
                return xy[2] .- thetop
            end
        else 
            return 0.0
        end
    end
end

function foilsdf(foil::Foil, splines)
    #make splines
    top, bottom = splines
    """
    isinside(xy)
    xy is a pair to test if inside of the splines constructed
    should allow for passing of vectors via 
    
    """
    function isinside(xy)
        #make a range of acceptable xs        
        if minimum(foil.foil[1,:]) <= xy[1] <= maximum(foil.foil[1,:])
            #find the smallest distance in y if accX is true
            thetop = top(xy[1])
            thebot = bottom(xy[1])
            mid = (thetop + thebot)./2.0
            if xy[2] <= mid
                return thebot  - xy[2] 
            else
                return xy[2] .- thetop
            end
        else 
            return 0.0
        end
    end
end

function foilsdf(nfoil, N, splines)
    #make splines
    top,bottom = splines
    
    """
    isinside(xy)
    xy is a pair to test if inside of the splines constructed
    should allow for passing of vectors via 
    
    """
    function isinside(xy)
        #make a range of acceptable xs        
        if minimum(nfoil[1,:]) <= xy[1] <= maximum(nfoil[1,:])
            #find the smallest distance in y if accX is true
            thetop = top(xy[1])
            thebot = bottom(xy[1])
            mid = (thetop + thebot)./2.0
            if xy[2] <= mid
                return thebot  - xy[2] 
            else
                return xy[2] .- thetop
            end
        else 
            return 0.0
        end
    end
end

function foilsdf(nfoil, N)
    #make splines
    top,bottom =make_splines(nfoil,N)
    
    """
    isinside(xy)
    xy is a pair to test if inside of the splines constructed
    should allow for passing of vectors via 
    
    """
    function isinside(xy)
        #make a range of acceptable xs        
        if minimum(nfoil[1,:]) <= xy[1] <= maximum(nfoil[1,:])
            #find the smallest distance in y if accX is true
            thetop = top(xy[1])
            thebot = bottom(xy[1])
            mid = (thetop + thebot)./2.0
            if xy[2] <= mid
                return thebot  - xy[2] 
            else
                return xy[2] .- thetop
            end
        else 
            return 0.0
        end
    end
end

minsmax(foil) = map(x -> minimum(nfoil[1,:]) <= x <= maximum(nfoil[2,:]), dest[1,:])

function sdf_fence(wake::Wake, foil::Foil, flow::FlowParams; dest = nothing)
    # Estimate the final position of the vortices
    dest = isnothing(dest) ? wake.xy + wake.uv * flow.Δt : dest
    # Calculate the next position of the foil
    nfoil = next_foil_pos(foil, flow)
    _ , N  = findmin(nfoil[1,:])
    # Generate splines for the top and bottom paths of the foil
    top, bottom = make_splines(nfoil)
    # Construct signed distance functions (SDFs) for the foil
    sdf = foilsdf(nfoil, foil.N, (top, bottom))
    # Define the mid plane between splines
    mid(x) = (top(x) + bottom(x)) / 2.0
    
    # Define vortex motion as a line with slopes and intercepts
    ms = (dest[2, :] .- wake.xy[2, :]) ./ (dest[1, :] .- wake.xy[1, :])
    bs = ms .* dest[1, :] .- dest[2, :]
    # Check if the final position is inside the foil
    xinside = map(x -> minimum(nfoil[1,:]) <= x <= maximum(nfoil[1,:]), dest[1,:])
    inside = xinside + [sdf(dest[:, i]) for i in 1:size(dest)[2]] .< 0
    
    # Start the looping process, using a quadtree to reduce the computational load
    iters = 1
    
    while sum(inside) > 0 && iters < 10
        for i in findall(x -> x == 2, inside)
            flip = 1  # Variable to flip the direction of the tangent vortex
            # Check if the vortex is on the top or bottom of the foil
            if dest[2, i] >= mid(wake.xy[1, i])
                # Construct the spline path for the top of the foil
                tPath = Spline1D(nfoil[1, N+1:end], nfoil[2, N+1:end] - ms[i] .* nfoil[1, N+1:end] .+ bs[i])
                if wake.xy[1, i] < dest[1, i]
                    xint = filter(x -> wake.xy[1, i] <= x <= dest[1, i], roots(tPath))
                else
                    xint = filter(x -> wake.xy[1, i] >= x >= dest[1, i], roots(tPath))
                    flip = -1
                end
                # Check if there is an intersection point between the vortex and the foil
                if !isempty(xint)
                    yint = top(xint)
                    whichPanel = findlast(xint .>= nfoil[1, N+1:end]) + N
                else
                    whichPanel = findlast(dest[1, i] .>= nfoil[1, N:end]) + N
                end
            else
                # Construct the spline path for the bottom of the foil
                bPath = Spline1D(nfoil[1, N+1:-1:1], nfoil[2, N+1:-1:1] - ms[i] .* nfoil[1, N+1:end] .+ bs[i])
                if wake.xy[1, i] < dest[1, i]
                    xint = filter(x -> wake.xy[1, i] <= x <= dest[1, i], roots(bPath))                    
                else
                    xint = filter(x -> wake.xy[1, i] >= x >= dest[1, i], roots(bPath))
                    flip = -1
                end
                if !isempty(xint)
                    yint = bottom(xint)
                    whichPanel = findfirst(xint .>= nfoil[1, 1:N + 1]) - 1
                else
                    whichPanel = findfirst(dest[1, i] .>= nfoil[1, 1:N + 1]) - 1
                end
            end
            # Update the motion of the vortex based on the intersection with the foil
            if !isempty(xint)
                #make all tangents direct to TE
                tangents = [-foil.tangents[:,1:foil.N÷2] foil.tangents[:,foil.N÷2+1:end]]
                leg1 = [xint..., yint...] .- wake.xy[:, i] .+ flow.δ / 2.0 .* foil.normals[:, whichPanel]
                leg2 = dest[:, i] .- [xint..., yint...] .+ flow.δ / 2.0 .* foil.normals[:, whichPanel]
                mag = norm(leg2) * flip
                fin = [xint..., yint...] + mag .* tangents[:, whichPanel] .+ flow.δ / 2.0 .* foil.normals[:, whichPanel]
            # If the vortex is outside the last foil but within the next time step, move it according to the local panel's motion
            elseif sdf(dest[:, i]) < 0.0
                fin = dest[:, i] + nfoil[:, whichPanel] - foil.foil[:, whichPanel] .+ flow.δ / 2.0 .* foil.normals[:, whichPanel]
            end
            dest[:, i] = fin
            inside = [sdf(dest[:, i]) for i in 1:size(dest)[2]] .< 0
            inside += map(x -> minimum(nfoil[1,:]) <= x <= maximum(nfoil[1,:]), dest[1,:])
            iters += 1
       
        end
    end
    @assert count(x-> x ==2, inside) == 0 
    dest    
end

"""
    sdf_fence(wake::Wake, foils::Vector,flow::FlowParams;dest=nothing, mask=nothing)

    mask -> bit mask defaults to all foils get a fence, can select which get fences
"""
function sdf_fence(wake::Wake, foils::Vector, flow::FlowParams; mask=nothing) 
    dest =  wake.xy + wake.uv * flow.Δt    
    mask = isnothing(mask) ? ones(Bool, length(foils)) : mask
    for i in findall(mask)
        dest = sdf_fence(wake, foils[i], flow; dest=dest)              
    end
    dest
end


begin
    foils = [(foil.foil[1,:].+1.5)... foil.foil[1,:]...;
             foil.foil[2,:]...   foil.foil[2,:]...]
    plot(foils[1,:], foils[2,:])
end

begin
    heave_pitch = deepcopy(defaultDict)
    heave_pitch[:N] = 64
    heave_pitch[:Nt] = 64
    heave_pitch[:Ncycles] = 1
    heave_pitch[:f] = 2.0
    heave_pitch[:Uinf] = 1.0
    heave_pitch[:kine] = :make_heave_pitch
    θ0 = deg2rad(1)
    h0 = 0.01
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(;heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    
     for i in 1:flow.Ncycles*flow.N*2
        time_increment!(flow, foil, wake)   
        dest = wake.xy + wake.uv .* flow.Δt
                         
    end
    #We only care if the vortex ends up inside of the foil after motion
    nfoil = next_foil_pos(foil, flow)
    a = plot_current(foil,wake)
    wake.xy .-= [2, 0.0]    
    a = plot_current(foil,wake)
end

begin
    @show flow.n
    movie = @animate for i in 1:flow.N*3       
        dest = sdf_fence!(wake, foil, flow)
        wake.xy = dest     
        time_increment!(flow, foil, wake)          
        a = plot_current(foil,wake)
        a
    end
    gif(movie, "fence.gif")
end


begin
    @show flow.n
    movie = @animate for i in 1:flow.N*3
        #for the next time step
        nfoil = next_foil_pos(foil, flow)
        #make splines
        topcurr, bottomcurr = make_splines(foil)
        top, bottom = make_splines(nfoil,foil.N)
        # construct SDFs
        sdfcurr = foilsdf(foil, (topcurr, bottomcurr))                
        sdf = foilsdf(nfoil, foil.N, (top, bottom))

        # define the neutral axis        
        mid(x) = (top(x)+bottom(x))/2.0
        #initial guess at where vortices final position
        dest = wake.xy + wake.uv*flow.Δt

        #define vortex motion as line; slopes and intercepts
        ms = (dest[2,:] .- wake.xy[2,:]) ./(dest[1,:] .- wake.xy[1,:])
        bs = ms.*dest[1,:] - dest[2,:]
        
        inside = [sdf(dest[:,i]) for i in 1:size(dest)[2]] .< 0 
        insidecur = sum([sdfcurr(wake.xy[:,i]) for i in 1:size(dest)[2]] .< 0 )
               
        #start looping -> quadtree will reduce this load 
        # top and bottom paths for root finding
        iters = 1
        while sum(inside)>0 & iters < 10
            for i in findall(x-> x ==1, inside)                                
                # flip the tangent vortex is traveling to the left (-x dir)
                flip = 1
                #top of foil
                if dest[2,i] >= mid(wake.xy[1,i]) 
                    tPath = Spline1D(nfoil[1,foil.N÷2 + 1:end], nfoil[2,foil.N÷2+1:end]  - ms[i].*nfoil[1,foil.N÷2 + 1:end] .+ bs[i])
                    if wake.xy[1,i] < dest[1,i]
                        xint =  filter(x-> wake.xy[1,i] <= x <= dest[1,i], roots(tPath))
                    else
                        xint =  filter(x-> wake.xy[1,i] >= x >= dest[1,i], roots(tPath))
                        flip = -1
                    end
                    #make sure that there is an intercept, might be pancaked between time-steps
                    if !isempty(xint)
                        yint = top(xint)
                        whichPanel = findlast(xint .>= nfoil[1,foil.N÷2+1:end]) + foil.N÷2
                        # print("Intersection $(xint), $(yint)")
                    else
                        whichPanel = findlast(dest[1,i] .>=nfoil[1,foil.N÷2+1:end]) + foil.N÷2 
                    end
                else #bottom of the foil
                    bPath = Spline1D(nfoil[1,foil.N÷2+1:-1:1],  nfoil[2,foil.N÷2+1:-1:1] - ms[i].*nfoil[1,foil.N÷2 + 1:end] .+ bs[i])                
                    if wake.xy[1,i] < dest[1,i]
                        xint = filter(x-> wake.xy[1,i]<= x <= dest[1,i], roots(bPath))
                        flip = -1
                    else
                        xint = filter(x-> wake.xy[1,i]>= x >= dest[1,i], roots(bPath))
                    end                
                    if !isempty(xint)
                        yint = bottom(xint)
                        whichPanel = findfirst(xint .>= nfoil[1,1:foil.N÷2+1])-1
                        # print("Intersection $(xint), $(yint)")
                    else
                        whichPanel = findfirst(dest[1,i] .>= nfoil[1,1:foil.N÷2+1])-1
                    end
                end
                #intersection with nfoil, adjust motion
                if !isempty(xint)
                    leg1 = [xint..., yint...] .- wake.xy[:,i] .+ flow.δ/2.0 .*foil.normals[:,whichPanel] 
                    leg2 = dest[:,i] .- [xint..., yint...] .+ flow.δ/2.0 .*foil.normals[:,whichPanel]
                    mag = norm(leg2)*flip 
                    #if on the top and moves right to left <- flip sign or if on bottom and moves left to right flip sign
                    fin = [xint..., yint...] + mag.*foil.tangents[:,whichPanel] .+ flow.δ/2.0 .*foil.normals[:,whichPanel]
                    # println("ends up at $(fin) ")
                # pancaked - outside the last foil but travels within the next time step ->
                # move it how much the local panel has moved
                # elseif sdf(wake.xy[:,i])<0.0 
                elseif sdf(dest[:,i])<0.0 
                    fin = dest[:,i] + nfoil[:,whichPanel] - foil.foil[:,whichPanel] .+ flow.δ/2.0 .*foil.normals[:,whichPanel]

                end
                dest[:,i] = fin 
                inside = [sdf(dest[:,i]) for i in 1:size(dest)[2]] .< 0 
                iters +=1
            end
            @show iters
            # no vortices are allowed inside of the foil after motion
            @assert sum(inside) == 0
        end        
        a = plot_current(foil,wake) 

            
        wake.xy = dest     
        time_increment!(flow, foil, wake)  
        # end

    
        plot!(a, dest[1,:], dest[2,:], st=:scatter, ms = 1)    
        plot!(nfoil[1,:],nfoil[2,:],size=(1000,800))
        # plot!(xlims=(-0.45,-0.4),ylims=(-0.025,0.125))     
        a
    end
    gif(movie, "fence.gif")
end


begin
    plot(nfoil[1,:], top.(nfoil[1,:]))
    plot!(nfoil[1,:], bottom.(nfoil[1,:]))
    plot!(foil.foil[1,:], topcurr.(foil.foil[1,:]))
    plot!(foil.foil[1,:], bottomcurr.(foil.foil[1,:]))
    @show sdf(dest[:,13])
    plot!([dest[1,13]],[dest[2,13]],st=:scatter)

end

begin 
    tangents = [-foil.tangents[:,1:foil.N÷2] foil.tangents[:,foil.N÷2+1:end]]
    # quiver(foil.foil[1,1:32],foil.foil[2,1:32], quiver=(-foil.tangents[1,1:32],-foil.tangents[2,1:32]))
    # quiver!(foil.foil[1,32:end],foil.foil[2,32:end], quiver=(foil.tangents[1,32:end],foil.tangents[2,32:end]))
    quiver(foil.foil[1,:],foil.foil[2,:], quiver=(tangents[1,:], tangents[2,:]), label="new")
end



findall(in(right), left )
plot!(x, top(x) )
plot!(x, bottom(x) )
plot!(x, (top(x) + bottom(x))./2.0)

plot([sdf(wake.xy[:,i]) for i in 1:size(wake.xy)[2]])
plot!([xy[1]], [mid], st=:scatter)


x = -1.1:0.01:0
plot(x,top.(x))
plot!(x,bottom.(x), aspect_ratio = :equal)

xp,yp = -1.5, 0.05
top(xp)
plot!([xp], [yp], st=:scatter)


#make a range of acceptable xs
accX(x) = minimum(foil.foil[1,:])<= x <= maximum(foil.foil[1,:])

accX(xp)



top(1)
plot(top.(wake.xy[1,:]) + top.(dest[1,:]))
plot!(bottom.(wake.xy[1,:]) - bottom.(dest[1,:]))
plot(wake.xy[1,:], wake.xy[2,:], st=:scatter, color=:blue)
plot!(dest[1,:], dest[2,:], st=:scatter, color=:red)
plot!(foil.foil[1,:],foil.foil[2,:], aspect_ratio=:equal)
dest[2,:]
begin
    strt = [-0.85, 0.125]     
    stp  = [-0.75, -0.05]
    #define the line for vortex motion
    m = (stp[2] - strt[2]) /(stp[1]-strt[1])
    b = m*stp[1] - stp[2]
    # top and bottom paths for root finding
    tPath = Spline1D(foil.foil[1,foil.N÷2 + 1:end], foil.foil[2,foil.N÷2+1:end] - m.*foil.foil[1,foil.N÷2 + 1:end].+ b)
    bPath = Spline1D(foil.foil[1,foil.N÷2+1:-1:1], foil.foil[2,foil.N÷2+1:-1:1] - m.*foil.foil[1,foil.N÷2 + 1:end].+ b)
    #mid plane
    mid = (top.(foil.foil[1,foil.N÷2 + 1:end]) + bottom(foil.foil[1,foil.N÷2 + 1:end])) ./ 2.0
    midd(x) = (top(x)+bottom(x))/2.0
    #filter to ensure its on the particle path
    flip = 1
    if strt[2] >= midd(strt[1]) 
        if strt[1] < stp[1]
            xint =  filter(x-> strt[1]<= x <= stp[1], roots(tPath))
        else
            xint =  filter(x-> strt[1]>= x >= stp[1], roots(tPath))
            flip = -1
        end
        yint = top(xint)
        whichPanel = findlast(xint .>=foil.foil[1,foil.N÷2+1:end])+foil.N÷2
    else
        if strt[1] < stp[1]
            xint = filter(x-> strt[1]<= x <= stp[1], roots(bPath))
            flip = -1
        else
            xint = filter(x-> strt[1]>= x >= stp[1], roots(bPath))
        end
        yint = bottom(xint)
        whichPanel = findfirst(xint .>= foil.foil[1,1:foil.N÷2+1])-1
    end
    a = plot(pos[1,:], pos[2,:], aspect_ratio=:equal,label="")
    plot!(foil.foil[1,foil.N÷2 + 1:end], mid,label="")
    plot!([strt[1], stp[1]], [strt[2], stp[2]],label="")
    plot!([xint], [yint],st=:scatter, label="intercept")
    plot!([foil.foil[1,whichPanel:whichPan+1]], [foil.foil[2,whichPanel:whichPan+1]],st=:scatter,label="Panel Ends")
    
    leg1 = [xint..., yint...] .+ flow.δ/2.0 .*foil.normals[:,whichPanel] .- strt
    leg2 = stp .- [xint..., yint...] .+ flow.δ/2.0 .*foil.normals[:,whichPanel]
    mag = norm(leg2)*flip 

    #if on the top and moves right to left <- flip sign or if on bottom and moves left to right flip sign

    fin = mag.*foil.tangents[:,whichPanel]+[xint..., yint...] .+ flow.δ/2.0 .*foil.normals[:,whichPanel]

    plot!(a,[xint fin[1]]', [yint fin[2]]', marker=:star, label="final",size=(1000,800))
    
    # quiver!(a, foil.foil[1,whichPanel:whichPan+1],foil.foil[2,whichPanel:whichPan+1], 
    # quiver=(foil.tangents[1,whichPanel:whichPan+1],foil.tangents[2,whichPanel:whichPan+1]))
    a
end


function compute_signed_distance(x, y)
    t_c = 0.12
    m = 0  # Camber line for symmetric airfoil

    # Maximum thickness distribution (z_t) function
    function z_t(x)
        return (0.12/0.2) * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x^2 + 0.2843 * x^3 - 0.1015 * x^4)
    end

    # Compute Euclidean distance to camber line
    d_e = abs(y - m)

    # Compute thickness distribution
    d_t = z_t(x)

    # Compute signed distance
    d = y >= m ? y - m - d_t : - m + d_t

    return d
end

# Example usage
x = 0.5  # x-coordinate of the point
y = 0.01  # y-coordinate of the point
signed_distance = compute_signed_distance(x, y)
println("Signed distance:", signed_distance)

X = -2:0.1:2 
Y = deepcopy(X)
contourf(X,Y,compute_signed_distance.(X,Y))

heave_pitch = deepcopy(defaultDict)
heave_pitch[:N] = 64
heave_pitch[:Nt] = 64
heave_pitch[:Ncycles] = 1
heave_pitch[:f] = 1.0
heave_pitch[:Uinf] = 1.0
heave_pitch[:kine] = :make_heave_pitch
θ0 = deg2rad(5)
h0 = 0.0
heave_pitch[:motion_parameters] = [h0, θ0]

begin
    heave_pitch = deepcopy(defaultDict)
    heave_pitch[:N] = 64
    heave_pitch[:Nt] = 64
    heave_pitch[:Ncycles] = 1
    heave_pitch[:f] = 1.0
    heave_pitch[:Uinf] = 1.0
    heave_pitch[:kine] = :make_heave_pitch
    θ0 = deg2rad(5)
    h0 = 0.0
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(;heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    
     for i in 1:flow.Ncycles*flow.N*.5
        time_increment!(flow, foil, wake)   
        dest = wake.xy + wake.uv .* flow.Δt
                         
    end
    a = plot_current(foil,wake)
    wake.xy .-= [1.46, 0]
    a = plot_current(foil,wake)
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

function fence!(dest ,tree::Cell, pos , wake::Wake)
    tangents, normals,_ =norms(pos)
    for i in 2:size(pos)[2]-2

        node = findleaf(tree, pos[:,i])       
        if length(node.data)>0    
            # start by only looking to the next panel, filter to yield indexing for multiple intersections
            sects = map( x -> find_intersection(pos[:,i], pos[:,i+1], x[1], x[2]),            
                        [[wake.xy[:,node.data], dest[:,node.data]] for i =1:length(node.data)])
            more = map( x -> find_intersection(pos[:,i-1], pos[:,i+1], x[1], x[2]),
                    [[wake.xy[:,node.data], dest[:,node.data]] for i =1:length(node.data)])                
            sects = hcat(sects,more)
            more = map( x -> find_intersection(pos[:,i-1], pos[:,i+2], x[1], x[2]),
                    [[wake.xy[:,node.data], dest[:,node.data]] for i =1:length(node.data)])                
            sects = hcat(sects,more)
            for (ind, sect) in enumerate(sects)                
                if !isnothing(sect)
                    index = ind % length(node.data) +1
                    vdist = wake.xy[:,node.data[index]] - dest[:,node.data[index]] 
                    tointersetion   = (sect + flow.δ*normals[:,i]/2.0) - wake.xy[:,node.data[index]]                    
                    rest  = abs(norm(vdist) - norm(tointersetion))                    
                    dest[:,node.data[index]] = [1.0, 0.0] #(sect + flow.δ*normals[:,i]/2.0) + normals[:,i] *rest                    
                end
            end    
        end 
    end
    nothing    
end

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



struct distanceRefinery <: AbstractRefinery
    tolerance::Float64
end

# These two methods split the domain into the tree
function needs_refinement(r::distanceRefinery, cell)
    maximum(cell.boundary.widths) > r.tolerance
end

function refine_data(r::distanceRefinery, cell::Cell, indices)

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


function make_qt(wake::Wake,r::distanceRefinery )
    xmin, ymin = minimum(wake.xy, dims=2)
    xmax, ymax = maximum(wake.xy, dims=2)
    width  = xmax -xmin
    height = ymax - ymin
    root = Cell(SVector(xmin, ymin), SVector(width,height), collect(1:length(wake.Γ)))
    adaptivesampling!(root, r)
    root
end
begin
    foil, flow = init_params(;heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    
     for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)   
        dest = wake.xy + wake.uv .* flow.Δt
        dir = dest.-wake.xy
        r = distanceRefinery(maximum(abs.(dir))*10.0)
        tree = make_qt(wake,r)                
        fence!(dest,tree,foil,wake)
    end
    plot_current(foil,wake)
    wake.xy[1,:] .-= 2.0
    global pos = next_foil_pos(foil,flow)
    movie = @animate for i in 1:3*flow.Ncycles*flow.N/4
        pos = next_foil_pos(foil,flow)
        # move_wake!(wake, flow)   
        release_vortex!(wake, foil)
        # pos = next_foil_pos(foil,flow)
        move_foil!(foil, pos)
        solve_n_update!(flow, foil, wake) 
        dest = wake.xy + wake.uv .* flow.Δt
        r = distanceRefinery(maximum(abs.(dir))*5.0)
        tree = make_qt(wake,r)          
        pos = next_foil_pos(foil,flow)
        # fence!(dest,tree,foil,wake)      
        fence!(dest,tree,pos,wake)
        wake.xy = dest
        f = plot_current(foil,wake)
        f
    end
    gif(movie,"fence.gif")
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

