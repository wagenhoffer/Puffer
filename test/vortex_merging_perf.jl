using Puffer
using Plots


heave_pitch = deepcopy(defaultDict)
heave_pitch[:N] = 64
heave_pitch[:Nt] = 64
heave_pitch[:Ncycles] = 4
heave_pitch[:f] = 1.
heave_pitch[:Uinf] = 1
heave_pitch[:kine] = :make_heave_pitch
θ0 = deg2rad(5)
h0 = 0.0
heave_pitch[:motion_parameters] = [h0, θ0]

foil, flow = init_params(;heave_pitch...)
wake = Wake(foil)

begin
    foil, flow = init_params(;heave_pitch...)
    foil2, flow2 = init_params(;heave_pitch...)
    wake = Wake(foil)
    wake2 = Wake(foil2)
    
    (foil)(flow)
    (foil2)(flow2)
    movie = @animate for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)
        time_increment!(flow2, foil2, wake2)
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)        
        # only aggregate a few times a cycle
        if i%(flow.N÷2)== 0
        # if i > 32
            spalarts_prune!(wake, flow, foil; keep = flow.N)
        end
        win=nothing
        f = plot_current(foil, wake; window=win)
        g = plot_current(foil2, wake2; window=win)
        
        plot!(f, ylims=(-1,1))
        plot!(g, ylims=(-1,1))
        plot(f,g, layout = (2,1))        
    end
    gif(movie,  "./images/vortex_merging.gif", fps=10)
end


function gen_coeffs(heave_pitch, spalart=true)
    foil, flow = init_params(;heave_pitch...)   
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    
    for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)
        phi =  get_phi(foil, wake)                                   
        p = panel_pressure(foil, flow,  old_mus, old_phis, phi)        
        old_mus = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] = get_performance(foil, flow, p)
        ps[:,i] = p
        if spalart && i%(flow.N÷1) == 0
            spalarts_prune!(wake, flow, foil; keep=flow.N)
        end
    end
    coeffs    
end

begin
    coeffs_on = gen_coeffs(heave_pitch, true)    
    coeffs_off = gen_coeffs(heave_pitch, false)    
    L2= norm(coeffs_on .-coeffs_off, 2)./norm(coeffs_on, 2)
    @test L2 < 1e-2
end

begin
    t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs_on[1,start:end], label="Force off"  ,lw = 3, marker=:none)
        plot!(a, t[start:end], coeffs_off[1,start:end], label="Force on"  ,lw = 0, marker=:circle)
    b = plot(t[start:end], coeffs_off[2,start:end], label="Lift off"   ,lw = 3, marker=:none)
        plot!(b, t[start:end], coeffs_on[2,start:end], label="Lift on"   ,lw = 0, marker=:circle)
    c = plot(t[start:end], coeffs_on[3,start:end], label="Thrust off" ,lw = 3, marker=:none)
        plot!(c, t[start:end], coeffs_off[3,start:end], label="Thrust on" ,lw = 0, marker=:circle)
    d = plot(t[start:end], coeffs_on[4,start:end], label="Power off"  ,lw = 3, marker=:none)
        plot!(d, t[start:end], coeffs_off[4,start:end], label="Power on"  ,lw = 0, marker=:circle)
    
    plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(800,800))
end