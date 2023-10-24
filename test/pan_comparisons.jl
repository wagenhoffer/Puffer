using Puffer
# include(joinpath("..","src","Puffer.jl"))
using Plots
using SpecialFunctions
using Plots
# experimental data scraped from paper below
# h0/c = 0.75 αmax = 15°

cta = [0.20213065147399903 0.25042614936828617 0.3029829502105713 0.3498579502105713 0.40241475105285646
       0.22763582942121677 0.3107534358798949 0.3938710423385726 0.4769886487972504 0.5878122790867963]
ηa= [0.2 0.2517482346754808 0.30069927509014427 0.35104895958533655 0.40139855018028847
    0.6886363220214844 0.6409089660644534 0.6000000000000001 0.5863636779785157 0.5522726440429686]
begin
# """
# Boundary-element method for the prediction of performance of flapping foils with leading-edge separation
# Y Pan, X Dong, Q Zhu, DKP Yue - Journal of Fluid Mechanics, 2012
# DOI: https://doi.org/10.1017/jfm.2012.119
# """
    pan = deepcopy(defaultDict)
    # pan[:Nt] = Nt
    pan[:N] = 64

    pan[:Ncycles] = 5

    pan[:f] = 0.5
    pan[:Uinf] = 0.4
    pan[:kine] = :make_heave_pitch
    pan[:pivot] = 1.0/3.0
    pan[:thick] = 0.2
    pan[:foil_type] = :make_naca
    θ0 = 0.00 #deg2rad(0.01)
    h0 = 0.01
    ψ = 0.0
    pan[:motion_parameters] = [h0, θ0]


    foil, flow, wake, coeffs = run_sim(; pan...)
    coeffs ./= (0.5*flow.Uinf^2)
    q∞ = 0.5*flow.Uinf^2
    St = 2*h0*foil.f/flow.Uinf
    k = π*foil.f*foil.chord/flow.Uinf
    # @show St, k
    τ =	 collect(flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt) .*foil.f
    cl = zeros(size(τ))
    theo(k) = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
    ck = theo(k)
    @. cl = -2*π^2*St*abs.(ck)*cos(2π.*τ + angle(ck)) - π^2*St*k*sin(2π.*τ)

    shift = 0
    plot(τ[flow.N:end], coeffs[2, flow.N+shift:end+shift], marker=:circle, label="BEM Lift")
    plt = plot!(τ[flow.N:end], cl[flow.N:end], label="Theo Lift",lw=3)
end
begin
    pan = deepcopy(defaultDict)
    pan[:N] = 64
    pan[:Nt] = 64
    pan[:Ncycles] = 5
    pan[:f] = 0.5
    pan[:Uinf] = 0.4
    pan[:kine] = :make_heave_pitch
    θ0 = deg2rad(5)
    h0 = 0.0
    pan[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; pan...)
    wake = Wake(foil)
    (foil)(flow)    
    movie = @animate for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)        
        win = (minimum(foil.foil[1, :]') + 3.0*foil.chord / 4.0, maximum(foil.foil[1, :]) + foil.chord * 0.25)
        win = nothing
        f = plot_current(foil, wake; window=win)
        f
    end
    gif(movie, "./images/h_$(h0)_p_$(rad2deg(θ0)).gif", fps=30)
end

begin
    """numerical approach to finding θ given αmax and Strouhal number
    αmax = max_t |α^h(t) + θ(t)|
    α^h = -tan^-1(̇h(t)/U)
    ̇h = 2πfh0cos(ωt)"""
    St = 0.2
    h0 = 0.1
    αmax = deg2rad(15)
    θmax = αmax - atan(π*St)
    pan = deepcopy(defaultDict)
    # pan[:Nt] = Nt
    pan[:N] = 64

    pan[:Ncycles] = 5
    pan[:f] = 1.0
    pan[:Uinf] = 2.0*pan[:f]*h0/St
    pan[:kine] = :make_heave_pitch
    pan[:pivot] = 1.0/3.0
    pan[:thick] = 0.12
    pan[:foil_type] = :make_naca
    θ0 = 0#θmax #deg2rad(0.01)

    ψ = 0.0
    pan[:motion_parameters] = [h0, θ0]

    foil, flow = init_params(;pan...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)
        #ZOom in on the edge

        win=nothing
        f = plot_current(foil, wake;window=win)
        f
    end
    gif(movie, "./images/pan15_St2.gif", fps=10)
end
begin
    """ convergence plots"""
    plt = plot()
    for n in [32,64,128,256]
        pan[:N] = n
        foil, flow, wake, coeffs = run_sim(; pan...)
        coeffs ./= (0.5*flow.Uinf^2)
        plot!(plt,τ[flow.N:end], coeffs[2, flow.N+shift:end+shift], label="$n")
    end
    plt

    plt = plot()
    for n in [32,64,128,256]

        pan[:N] = 64
        pan[:Nt] = n
        foil, flow, wake, coeffs = run_sim(; pan...)
        coeffs ./= (0.5*flow.Uinf^2)
        τ =	 collect(flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt) .*foil.f
        # pan[:Nt] = Nt
        pan[:N] = 64
    
        pan[:Ncycles] = 15
        pan[:f] = 0.5
        pan[:Uinf] = 0.4
        pan[:kine] = :make_heave_pitch
        pan[:pivot] = 1.0/3.0
        pan[:thick] = 0.12
        pan[:foil_type] = :make_naca
        θ0 = 0.00 #deg2rad(0.01)
        h0 = 0.01
        ψ = 0.0
        pan[:motion_parameters] = [h0, θ0]
        plot!(plt,τ[flow.N:end], coeffs[2, flow.N+shift:end+shift], label="$n")
    end
    plt
end

begin 
    """Figure 9a,g"""
    Strous = 0.2:0.05:0.5
    Cts = zeros(size(Strous))
    pan = deepcopy(defaultDict)
    # pan[:Nt] = Nt
    pan[:N] = 64
    pan[:Ncycles] = 5
    pan[:f] = 0.5
    pan[:Uinf] = 1.0
    pan[:kine] = :make_heave_pitch
    pan[:pivot] = 1.0/3.0
    pan[:thick] = 0.12
    pan[:foil_type] = :make_naca
    θ0 = deg2rad(15)
    h0 = 1.0
    
    pan[:motion_parameters] = [h0, θ0]

    for (i,strou) in enumerate(Strous)
        pan[:f] = strou*pan[:Uinf]/2.0/h0
        foil, flow, wake, coeffs = run_sim(; pan...)
        Cts[i] = sum(coeffs[3,flow.N:end])./(flow.Ncycles-1)./flow.N
    end
    plot(Strous, Cts)
        
end