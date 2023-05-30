# using BemRom
include("../src/BemRom.jl")
using SpecialFunctions
using Plots

begin
	garrick = deepcopy(defaultDict)
	garrick[:Nt] = 150
	garrick[:Ncycles] = 5
	garrick[:f] = 1.5
    garrick[:Uinf] = 2.0
	garrick[:kine] = :make_heave_pitch
	θ0 = deg2rad(5)
	h0 = 0.0
	garrick[:motion_parameters] = [h0, θ0]


	foil, flow, wake, coeffs = run_sim(; garrick...)
end

begin
	RHO = RHO = 1.0
	b = foil.chord / 2.0
	
	# t = flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt
	t = LinRange(0,flow.Ncycles/foil.f,flow.Ncycles*flow.N)
	
	ω = 2π * foil.f
	k = ω *b / flow.Uinf
    # ϕ = -pi*(flow.Uinf/foil.f)/4.0
	ϕ = pi
	# ϕ = 0.0
	@show ϕ
	a = -1.0 #-1 is rotation about leading edge
	theo = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
	theod(k) = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
	F, G = real(theo), imag(theo)
	R = π * θ0 * (k^2 / 2 * (0.125 + a^2) + (0.5 + a) * (F - (0.5 - a) * k * G))
	I = -π * θ0 * (k / 2 * (0.5 - a) - (0.5 + a) * (G + (0.5 - a) * k * F))
	Φ = atan(I, R)
	gar = sqrt.(R .^ 2 + I .^ 2) # *  0.5* RHO* flow.Uinf^2* foil.chord* 
	Cl = gar .* exp.(1im * ω * t)
	M = gar .* exp.(1im * ω * t .+ Φ)
	den = foil.chord^2 * foil.f^2 * (θ0 * foil.chord + h0)
	
	inst_lift = zeros(size(t))
	@. inst_lift =
		-RHO * b^2 * (flow.Uinf * pi * θ0 * ω * cos(ω * t + ϕ) - pi * h0 * ω^2 * sin(ω * t) +
					  pi * b * a * θ0 * ω^2 * sin(ω * t + ϕ)) - 2.0 * pi * RHO * flow.Uinf * b * F * (flow.Uinf * θ0 * sin(ω * t + ϕ) +
											  h0 * ω * cos(ω * t) + b * (0.5 - a) * θ0 * ω * cos(ω * t + ϕ)) -
		2.0 * pi * RHO * flow.Uinf * b * G * (flow.Uinf * θ0 * cos(ω * t + ϕ) - h0 * ω * sin(ω * t) - b * (0.5 - a) * θ0 * ω * sin(ω * t + ϕ))
    @show k, ϕ , k/ϕ
	
	plot(t[25:end], coeffs[2, 25:end], marker = :circle, lw = 0, ms = 3, label = "BEM  \$C_L\$")
	plot!(t, inst_lift, label = "Garrick  \$C_L\$",lw=3)
	# plot!(t, real(M), label="Garrick  \$C_M\$")
   
end
begin
	#Katz and Plotkin eqn 13.73a
	phi = ϕ = π/2
	alpha = θ0*sin.(ω*t .+ phi)
	heave = h0*sin.(ω*t)
	alphadot = θ0*ω*cos.(ω*t .+ phi)
	alphadotdot = -θ0*ω^2*sin.(ω*t .+ phi)
	hdot = h0*ω*cos.(ω*t)
	hdotdot = -h0*ω^2*sin.(ω*t)
	kk = ω*foil.chord/2/flow.Uinf
	lift = π*RHO*flow.Uinf*foil.chord*theod(k)*(flow.Uinf*alpha - hdot + 0.75*foil.chord*alphadot) + 
		π*RHO*foil.chord^2/4.0 *(flow.Uinf*alphadot - hdotdot + foil.chord/2.0*alphadotdot)

	plot(t, real(lift), label="Katz and Plotkin  \$C_L\$",lw=3)
	plot!(t[25:end], coeffs[2, 25:end], marker = :circle, lw = 0, ms = 3, label = "BEM  \$C_L\$")
end




#Katz and Plotkin 13.74a
M0 = - π*RHO*foil.chord^2/4*(-b*hdotdot+3*flow.Uinf*b/2*alphadot + 9/32*foil.chord^2*alphadotdot +flow.Uinf*theod(k)*(-hdot +flow.Uinf*alpha+3*b/2*alphadot))

plot(real(M0.*alpha))
# plot(t[25:end], coeffs[2, 25:end], marker = :circle, lw = 0, ms = 3, label = "BEM  \$C_L\$")
km = foil.f*foil.chord/2/flow.Uinf
td = theod(km)
F,G = real(td), imag(td)
C_t_Jones_and_Platzer_Pitch_Only = pi*k^2*θ0^2*((F^2 + G^2)*(1/k^2+(0.5-a)^2)+(0.5 - F)*(0.5-a)-F/k^2-(0.5+a)*G/k)
Ct = 3*pi^3/32 - pi^3/8*(3*F/2 - G/(2*pi*km) + F /(pi*k)^2 - (F^2+G^2)*(1/(pi*km)^2 + 9/4))
plot(Ct.*sin.(ω*t), label="C_t Garrick")
plot!(coeffs[3,25:end]/(0.5*θ0^2*foil.f), label="C_t BEM")

#RIPPING OFF MOORED
m = exp.(1im*ω*t)
test_thing = imag( ( ( 3. * pi ^3 * (1 - td)/ 4.) + 1im*(9. * pi ^4 *km/16. + pi^2*td/2.0/km)) * m)
Power_star = imag(test_thing.*m)

P_star = imag( ( ( pi^2/2. - td/km^2) - 1im*( pi/2.0/km + 3.0*pi*td/2.0/km ) )*pi*m )
alpha_P_star = imag(P_star.*m)
S_star = imag( sqrt(2.)/2.0*( 2.0*td/km + 1im*( 3.0*pi*td - pi ))*m )

inst_C_T_moored = pi/2.0*S_star.^2 + alpha_P_star # use the imaginary part--associated with sin(omega*t) motion.
plot(t[25:end], coeffs[3, 25:end]/foil.f^2/θ0^2, marker = :circle, lw = 0, ms = 3, label = "BEM  \$C_T\$")
plot!(t, inst_C_T_moored, label="Garrick  \$C_T\$",lw=3)