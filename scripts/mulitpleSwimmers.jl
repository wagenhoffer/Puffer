using Puffer
using Plots
using LinearAlgebra

"""
create_foils(num_foils, starting_positions, kine; kwargs...)

Create multiple foils with specified starting positions and kinematics.

# Arguments
- `num_foils`: Number of foils to create.
- `starting_positions`: Matrix of starting positions for each foil.
- `kine`: Kinematics for the foils.
- `kwargs`: Additional keyword arguments.

# Returns
- `foils`: Array of created foils.
- `flow`: Flow value.

# Example
An example of usage can be found in multipleSwimmers.jl
"""
function create_foils(num_foils, starting_positions, kine; kwargs...) 
    pos = deepcopy(defaultDict)       
    pos[:kine] = kine
    foils = Vector{Foil{pos[:T]}}(undef, num_foils)
    flow = 0
    for i in 1:num_foils

        for (k,v) in kwargs
            if size(v,1) == 1
                pos[k] = v
            else
                v = v[i,:]
                if size(v,1) == 1
                    pos[k] = v[1]
                else
                    pos[k] = v
                end
            end
        end   
        # @show pos     
        foil, flow = init_params(; pos...)      
        @show flow          
        foil.foil[1, :] .+= starting_positions[1, i] * foil.chord
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



@time begin
    num_foils = 2
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [0.0 1.5 ; 0.0 0.0 ]
    phases = [pi/2, -pi/2]
    θ0 = deg2rad(10)
    h0 = 0.1
    motion_parameters = [h0 θ0 ; -h0 θ0]

    foils, flow = create_foils(num_foils, starting_positions, :make_heave_pitch;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 5, Nt = 64, kine=[:make_heave_pitch, :make_heave_pitch]);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    time_increment!(flow, foils, wake)

    for t in 2:steps
        time_increment!(flow, foils, wake; mask=[false, true])    
        xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        # plot(foils, wake)#; xlims=xlims, ylims=ylims)
    end
    # gif(movie, "newMulti.gif", fps = 30)
end

begin
    num_foils = 4
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [0.0 1.5 3.0 1.5; 0.0 0.5 0.0 -0.5 ]
    phases = [pi/2, -pi/2, -pi/2, pi/2]
    
    θ0 = deg2rad(5)
    h0 = 0.1
    motion_parameters = [h0 θ0 ; -h0 θ0; -h0 θ0 ; h0 θ0]

    foils, flow = create_foils(num_foils, starting_positions, :make_heave_pitch;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 5, Nt = 64);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    time_increment!(flow, foils, wake)

    @time movie = @animate for t in 2:steps
        time_increment!(flow, foils, wake; mask=[false, true, true, true])    
        xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        plot(foils, wake)#; xlims=xlims, ylims=ylims)
    end
    gif(movie, "newMulti.gif", fps = 30)
end

begin
    num_foils = 3
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [0.0 1.5 1.5; 0.0 0.5 -0.5 ]
    phases = [pi/2, -pi/2,  pi/2]
    θ0 = deg2rad(5)
    h0 = 0.1
    motion_parameters = [0.0 0.0 ; -h0 θ0; h0 θ0]

    foils, flow = create_foils(num_foils, starting_positions, :make_heave_pitch;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 5, Nt = 64);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake)

    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake; mask=[false, true, true])    
        xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        plot(foils, wake)#; xlims=xlims, ylims=ylims)
    end
    gif(movie, "newMulti.gif", fps = 30)
end
begin
    num_foils = 2
    # starting_positions = [2.0 1.0 1.0 0.0; 0.0 1.0 -1.0 0.0]
    starting_positions = [1.5 0.0;  0.15 0.15 ]
    phases = [-pi, 0.0]    
    fs = [ 2.0, 2.0]
    ks = [ 1.0, 1.0]
    a0 = 0.1
    motion_parameters = [a0 for i in 1:num_foils]

    foils, flow = create_foils(num_foils, starting_positions, :make_wave;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 5,
             k= ks,  Nt = 64, f = fs);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake)

    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake; mask=[false, true])    
        # xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        # ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        #  if t%(flow.N÷2) == 0
        #     spalarts_prune!(wake, flow, foils; te =[foils[1].foil[1,1] 0.0]' )
        # end
        # (foils)(flow)
        plot(foils, wake; size=(1200,800))#; xlims=xlims, ylims=ylims)
    end
    gif(movie, "newMulti.gif", fps = 30)
end




function school_of_four(height, width, chord)
    # Define the coordinates of the swimmers in the rhombus 
    xs = [0, width/2, width, width/2] .- chord/2.0
    ys = [height/2, 0, height/2, height]
    [xs ys]'
end

begin
    num_foils = 4
    starting_positions = school_of_four(1.0, 2.0, 1.0)
    phases = [pi/2, pi/2, pi/2, pi/2]    
    fs = [ 1.0, 1.0, 1.0, 1.0]
    ks = [ 1.0, 1.0, 1.0, 1.0]
    a0 = 0.1
    motion_parameters = [a0 for i in 1:num_foils]

    foils, flow = create_foils(num_foils, starting_positions, :make_ang;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 5,
             k= ks,  Nt = 64, f = fs);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake)

    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake; mask=[false, false, true, false])    
        xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        #  if t%(flow.N÷2) == 0
        #     spalarts_prune!(wake, flow, foils; te =[foils[1].foil[1,1] 0.0]' )
        # end
        plot(foils, wake)#; xlims=xlims, ylims=ylims)
    end
    gif(movie, "newMulti.gif", fps = 30)
end

begin
    num_foils = 4
    h = 2.0  
    α = atan(h)  
    r = h/sin(α)
    # θ = 3pi/4
    starting_positions = [0.0  0.0 1.25 1.25; -0.25 0.25 -0.25 0.25]
    phases = [pi/2, pi/2,pi/2, pi/2]    
    fs     = [ 1.0, -1.0, 1.0, -1.0]
    ks     = [ -1.0, 1.0, -1.0, 1.0]
    a0 = 0.1
    motion_parameters = [a0 for i in 1:num_foils]

    foils, flow = create_foils(num_foils, starting_positions, :make_ang;
             motion_parameters=motion_parameters, ψ=phases, Ncycles = 4,
             k= ks,  Nt = 64, f = fs);
    wake = Wake(foils)
    @show [foil.f for foil in foils]
    (foils)(flow)
    steps = flow.N *flow.Ncycles
    totalN = sum(foil.N for foil in foils)
    kuttas = zeros(num_foils, 2, steps)
    old_mus = zeros(3, totalN)
    old_phis = zeros(3, totalN)
    coeffs = zeros(length(foils), 4, steps)
    coeffs[:,:,1] = time_increment!(flow, foils, wake)

    movie = @animate for t in 2:steps
        coeffs[:,:,t] = time_increment!(flow, foils, wake; mask=[false, false, false, false])    
        xlims = foils[2].foil[1,1] .+ (-0.25, 0.25)
        ylims = foils[2].foil[2,1] .+ (-0.5, 0.25)
        #  if t%(flow.N÷2) == 0
        #     spalarts_prune!(wake, flow, foils; te =[foils[1].foil[1,1] 0.0]' )
        # end
        plot(foils, wake)#; xlims=xlims, ylims=ylims)
    end
    gif(movie, "newMulti.gif", fps = 30)
end