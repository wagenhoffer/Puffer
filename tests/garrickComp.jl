# using BemRom
include("../src/BemRom.jl")
using SpecialFunctions
using Plots

begin
	garrick = deepcopy(defaultDict)
	garrick[:Nt] = 150
	garrick[:Ncyc] = 5
	garrick[:f] = 0.5
    garrick[:Uinf] = 0.85
	garrick[:kine] = :make_heave_pitch
	θ0 = deg2rad(1)
	h0 = 0.10
	garrick[:motion_parameters] = [h0, θ0]


	foil, flow, wake, coeffs = run_sim(; garrick...)
end

begin
	RHO = flow.ρ = 1.0
	b = foil.chord / 2.0
	
	t = flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt
	t = LinRange(0,flow.Ncycles/foil.f,flow.Ncycles*flow.N)
	
	ω = 2π * foil.f
	k = ω * b / flow.Uinf
    ϕ = -pi/2*(flow.Uinf/foil.f)
	ϕ = -pi/2
	ϕ = 0.0
	@show ϕ
	a = -1.0 #-1 is rotation about leading edge
	theo = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
	theod(k) = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
	F, G = real(theo), imag(theo)
	R = π * θ0 * (k^2 / 2 * (0.125 + a^2) + (0.5 + a) * (F - (0.5 - a) * k * G))
	I = -π * θ0 * (k / 2 * (0.5 - a) - (0.5 + a) * (G + (0.5 - a) * k * F))
	Φ = atan(I, R)
	gar = sqrt.(R .^ 2 + I .^ 2) # *  0.5* flow.ρ* flow.Uinf^2* foil.chord* 
	Cl = gar .* exp.(-1im * ω * t)
	M = gar .* exp.(1im * ω * t .+ Φ)
	den = foil.chord^2 * foil.f^2 * (θ0 * foil.chord + h0)
	
	inst_lift = zeros(size(t))
	@. inst_lift =
		-RHO * b^2 * (flow.Uinf * pi * θ0 * ω * cos(ω * t + ϕ) - pi * h0 * ω^2 * sin(ω * t) +
					  pi * b * a * θ0 * ω^2 * sin(ω * t + ϕ)) - 2.0 * pi * RHO * flow.Uinf * b * F * (flow.Uinf * θ0 * sin(ω * t + ϕ) +
											  h0 * ω * cos(ω * t) + b * (0.5 - a) * θ0 * ω * cos(ω * t + ϕ)) -
		2.0 * pi * RHO * flow.Uinf * b * G * (flow.Uinf * θ0 * cos(ω * t + ϕ) - h0 * ω * sin(ω * t) - b * (0.5 - a) * θ0 * ω * sin(ω * t + ϕ))
    @show k, ϕ , k/ϕ
	
	plot(t[1:end-24], coeffs[2, 25:end], marker = :circle, lw = 0, ms = 3, label = "BEM  \$C_L\$")
	plot!(t, -inst_lift, label = "Garrick  \$C_L\$",lw=3)
	# plot!(t, real(M), label="Garrick  \$C_M\$")
   
end
#Katz and Plotkin eqn 13.37a
alpha = θ0*sin.(ω*t .+ π/2)
heave = h0*sin.(ω*t)
alphadot = θ0*ω*cos.(ω*t .+ π/2)
alphadotdot = -θ0*ω^2*sin.(ω*t .+ π/2)
hdot = h0*ω*cos.(ω*t)
hdotdot = -h0*ω^2*sin.(ω*t)
kk = ω*foil.chord/2/flow.Uinf
lift = π*flow.ρ*flow.Uinf*foil.chord*theod(k)*(flow.Uinf*alpha - hdot + 0.75*foil.chord*alphadot) + 
	   π*flow.ρ*foil.chord^2/4.0 *(flow.Uinf*alphadot - hdotdot + foil.chord/2.0*alphadotdot)

plot(t, real(lift), label="Katz and Plotkin  \$C_L\$",lw=3)
plot!(t[25:end], coeffs[2, 25:end], marker = :circle, lw = 0, ms = 3, label = "BEM  \$C_L\$")