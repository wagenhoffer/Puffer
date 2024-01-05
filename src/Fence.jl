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
    top, bottom = make_splines(nfoil,N)
    
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


#TODO: refactor this into smaller functions for testing
"""
    sdf_fence(wake::Wake, foil::Foil, flow::FlowParams; dest = nothing)

    dest -> the destination of the vortices
    foil -> the foil to fence
    flow -> the flow parameters
    dest -> the destination of the vortices
"""
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
    inside = map(x-> sdf(dest[:,x]) < 0, axes(dest,2) )
    inside = inside .&& xinside
    # Start the looping process, using a quadtree to reduce the computational load
    iters = 1

    while sum(inside) > 0 && iters < 10        
        deez = findall(x -> x == 1, inside)

        for i in deez
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
                bPath = Spline1D(nfoil[1, N:-1:1], nfoil[2, N:-1:1] - ms[i] .* nfoil[1,  N:-1:1] .+ bs[i])
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
                # leg1 = [xint..., yint...] .- wake.xy[:, i] .+ flow.δ .* foil.normals[:, whichPanel]
                leg2 = dest[:, i] .- [xint..., yint...] .+ flow.δ .* foil.normals[:, whichPanel]
                mag = norm(leg2) * flip
                fin = [xint..., yint...] + mag .* tangents[:, whichPanel] .+ flow.δ .* foil.normals[:, whichPanel]
            # If the vortex is outside the last foil but within the next time step, move it according to the local panel's motion
            elseif sdf(dest[:, i]) < 0.0
                fin = dest[:, i] + nfoil[:, whichPanel] - foil.foil[:, whichPanel] .+ flow.δ .* foil.normals[:, whichPanel]
            end
            dest[:, i] = fin
            
            xinside = map(x -> minimum(nfoil[1,:]) <= x <= maximum(nfoil[1,:]), dest[1,:])    
            inside = map(x-> sdf(dest[:,x]) < 0, axes(dest,2) )
            inside = inside .&& xinside
            
            iters += 1
       
        end
    end
    @assert count(x-> x ==2, inside) == 0 
    dest    
end

function process_intersection(xint, path, dest, i, nfoil, N)
    if !isempty(xint)
        yint = path(xint)
        whichPanel = findlast(xint .>= nfoil[1, N+1:end]) + N
    else
        whichPanel = findlast(dest[1, i] .>= nfoil[1, N:end]) + N
        yint = nfoil[2, whichPanel] #TODO: WTF is this?
    end
    return yint, whichPanel
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