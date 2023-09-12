include("../src/BemRom.jl")

using Plots
function plot_teng(coeffs, flow;ylims=(-0.8, 0.8))
	t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = length(t)-2*flow.N

    plt = plot(t[start:end], coeffs[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
    plot!(t[start:end], -coeffs[3,start:end]*50, label="Thrust" ,lw = 3, marker=:circle,ylims= ylims)	
    plt
end

begin
    """fig 5.11 ωcU∞ = 4.3 h =0.018"""
	teng = deepcopy(defaultDict)
	teng[:Nt] = 64
	teng[:N] = 64	
    teng[:thick] = 0.15
	teng[:Ncycles] = 5
	teng[:f] = 4.3/(2π)
    teng[:Uinf] = 1.0
	teng[:kine] = :make_heave_pitch
	teng[:pivot] = 0.0
    
	θ0 = deg2rad(0)
	h0 = 0.018
	teng[:motion_parameters] = [h0, θ0]


	foil, flow, wake, coeffs = run_sim(; teng...)
	coeffs ./= (0.5*flow.Uinf^2)
    p = plot_teng(coeffs, flow;ylims=(-0.8,0.8))
	savefig(p, "./images/teng5_11_comparison.png")
	p
end

begin
    """fig 5.12 ωcU∞ = 17.0  h =0.018"""
	teng = deepcopy(defaultDict)
	teng[:Nt] = 64
	teng[:N] = 64
    teng[:thick] = 0.15
	teng[:Ncycles] = 4
	teng[:f] = 17.0/(2π)
    teng[:Uinf] = 1.0
	teng[:kine] = :make_heave_pitch
	teng[:pivot] = 0.0
    
	θ0 = deg2rad(0)
	h0 = 0.018
	teng[:motion_parameters] = [h0, θ0]


	foil, flow, wake, coeffs = run_sim(; teng...)
	coeffs ./= (0.5*flow.Uinf^2)
    p = plot_teng(coeffs, flow; ylims =(-9,9))
	savefig(p, "./images/teng5_12_comparison.png")
	p
end
	
