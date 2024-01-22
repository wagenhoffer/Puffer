# "this script will test using variable frequencies across a cycle of motion"


using Puffer
using Plots

ang = deepcopy(defaultDict)
ang[:N] = 64
ang[:Nt] = 64
ang[:Ncycles] = 2
ang[:f] = 1.0
ang[:Uinf] = 1
ang[:kine] = :make_wave
a0 = 0.1
ang[:motion_parameters] = [a0]


begin
    foil, flow = init_params(; ang...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        if i %flow.N < 32
            foil.f = 1.0
        else
            foil.f = 2.0
        end

        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0,
            maximum(foil.foil[1, :]) + foil.chord * 2)


        win = nothing
        f = plot(foil, wake)
        f
    end
    gif(movie, "./images/handp.gif", fps = 30)
end

# """we need to find a way to increase the frequency while maintaing a smooth motion
# ensure that panel velociies are smooth when we change the n value
begin
    
    movie = @animate for n in 1:128
        if n < 32
            Δt = flow.Δt
        elseif n<64
            Δt = flow.Δt*2
        else
            Δt = flow.Δt/2
        end
        out = foil.kine.(foil.col[1, :], foil.f, foil.k, n * Δt, foil.ψ)
        p = plot(out[1:32],  label = "")
        title!("$n")
        ylims!(-0.1,0.1)
        p
    end
    
    gif(movie, "./images/variable_freq.gif", fps = 10)
end

begin
    
    movie = @animate for n in 1:128
        if n < 32
            f = foil.f
        elseif n<64
            f = foil.f*2
        else
            f = foil.f/2
        end
        out = foil.kine.(foil.col[1, :], f, foil.k, n * flow.Δt,  foil.ψ)
        p = plot(out[1:32],  label = "")
        title!("$n")
        ylims!(-0.1,0.1)
        p
    end
    
    gif(movie, "./images/variable_freq.gif", fps = 10)
end

begin
    num = 128
    var = sin.(2π.*LinRange(0,1,num))

    movie = @animate for (n,f) in enumerate(var)
        (foil)(flow)
        foil.f = f
        # need to ensure that the velocity from panels makes sense and then
        # we can move to pressure
        out = foil.kine.(foil.col[1, :], foil.f, foil.k, n * flow.Δt,   foil.ψ)
        p = plot(out[1:32],  label = "")
        title!("$n")
        ylims!(-0.1,0.1)
        p
    end
    
    gif(movie, "./images/variable_freq.gif", fps = 10)
end
