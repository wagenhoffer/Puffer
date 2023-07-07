# using BemRom
include("../src/BemRom.jl")
using SpecialFunctions
using Plots

begin
	garrick = deepcopy(defaultDict)
	garrick[:Nt] = 150
	garrick[:N] = 150
	garrick[:Ncycles] = 5
	garrick[:f] = 0.5
    garrick[:Uinf] = 1.0
	garrick[:kine] = :make_heave_pitch
	garrick[:pivot] = 0.25
	θ0 = deg2rad(0)
	h0 = 0.001
	garrick[:motion_parameters] = [h0, θ0]


	foil, flow, wake, coeffs = run_sim(; garrick...)
	coeffs ./= (0.5*RHO*flow.Uinf^2)
end
	
function plot_coeffs(coeffs, flow)
	t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs[1,start:end], label="Force"  ,lw = 3, marker=:circle)
    b = plot(t[start:end], coeffs[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
    c = plot(t[start:end], coeffs[3,start:end], label="Thrust" ,lw = 3, marker=:circle)
    d = plot(t[start:end], coeffs[4,start:end], label="Power"  ,lw = 3, marker=:circle)
    p = plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(800,800))
	p 
end


begin
	RHO = RHO = 1.0
	b = foil.chord / 2.0
	
	# t = flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt
	t = LinRange(0,flow.Ncycles/foil.f,flow.Ncycles*flow.N)
	
	ω = 2π * foil.f
	k = ω *b / (2*flow.Uinf)
    
	ϕ = pi/2
	ϕ = 0.0
	@show ϕ
	a = -1.0 #-1 is rotation about leading edge
	theo = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
	theod(k) = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
	F, G = real(theo), imag(theo)
	R = π * θ0 * (k^2 / 2 * (0.125 + a^2) + (0.5 + a) * (F - (0.5 - a) * k * G))
	I = -π * θ0 * (k / 2 * (0.5 - a) - (0.5 + a) * (G + (0.5 - a) * k * F))
	ϕ = Φ = atan(I, R)
	gar = sqrt.(R .^ 2 + I .^ 2) # *  0.5* RHO* flow.Uinf^2* foil.chord
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
	
	plot(t[25:end], coeffs[2, 25:end]/2.0, marker = :circle, lw = 0, ms = 3, label = "BEM  \$C_L\$")
	plot!(t, inst_lift, label = "Garrick  \$C_L\$",lw=3)   
end

begin
	t =	 collect(flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt) 
	# load up values to match their defns
	ω = 2π*foil.f
	k = ω*foil.chord/2/flow.Uinf
	#Katz and Plotkin eqn 13.73a
	phi = ϕ = 0.#-π/2 #angle(theod(k))
	alpha = θ0*sin.(ω*t .+ phi)
	heave = h0*sin.(ω*t)
	alphadot = θ0*ω*cos.(ω*t .+ phi)
	alphadotdot = -θ0*ω^2*sin.(ω*t .+ phi)
	hdot = h0*ω*cos.(ω*t)
	hdotdot = -h0*ω^2*sin.(ω*t)
	
	lift = π*RHO*flow.Uinf*foil.chord*theod(k)*(flow.Uinf*alpha - hdot + 0.75*foil.chord*alphadot) + 
		π*RHO*foil.chord^2/4.0 *(flow.Uinf*alphadot - hdotdot + foil.chord/2.0*alphadotdot)

	plot(t, real(lift), label="Katz and Plotkin  \$C_L\$",lw=3)
	plot!(t[25:end], coeffs[2, 25:end], marker = :circle, lw = 0, ms = 3, label = "BEM  \$C_L\$")
end




 plot_current(foil,wake)

 begin
	garrick = deepcopy(defaultDict)
	garrick[:Nt] = 64
	garrick[:N] = 256
	garrick[:Ncycles] = 5
	garrick[:f] = 0.5
    garrick[:Uinf] = 1.0
	garrick[:kine] = :make_heave_pitch
	garrick[:pivot] = 0.25
	garrick[:thick] = 0.001
	θ0 = deg2rad(0)
	h0 = 0.001
	garrick[:motion_parameters] = [h0, θ0]


	foil, flow, wake, coeffs = run_sim(; garrick...)
	plot_current(foil, wake)
	plot_coeffs(coeffs/(0.5*flow.Uinf^2),flow)
 end

 begin
	# Change thickness to 0.1%
	#pure heaving compare - Quinn, moored
	St = 2*h0*foil.f/flow.Uinf
	k = π*foil.f*foil.chord/flow.Uinf
	@show St, k
	τ =	 collect(flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt) .*foil.f
	cl = zeros(size(τ))
	ck = theod(k)
	@. cl = -2*π^2*St*abs.(ck)*cos(2π.*τ + angle(ck)) - π^2*St*k*sin(2π.*τ)
	
	plot(τ, cl, label="Theo Lift",lw=3)
	plot!(τ[flow.N:end], coeffs[2, flow.N:end]./q∞, marker=:circle, label="BEM Lift")
 end

 begin
	""" Oscillation Frequency and Amplitude Effects on the Wake of a Plunging Airfoil
	J. Young∗ and J. C. S. Lai """
	q∞ = 0.5*flow.Uinf^2
	α = rad2deg(θ0)
	k = π*foil.f*foil.chord/flow.Uinf
	kα = k*α
	theo = theod(k)
	F, G = real(theo), imag(theo)
	A = (F^2 + G^2)
	B = sqrt(F^2 + G^2 + k*G +k^2/4)
	Ctmean= 4*π*(k*h0)^2*A
	# Ctmean= 4*π*(k*h0/2.0)^2*A # single side amp?
	Clpeak = 4*π*(k*h0)*B
	#fig4
	println("k = $k kh = $(k*h0)")
    println("Ctmean = $Ctmean \t Clpeak = $Clpeak")
	println("\t$(mean(coeffs[3,flow.N:end]/q∞)) \t $(maximum(coeffs[2,flow.N:end]/q∞))")
 end

begin
	ks = LinRange(0.1, 4.0/π, 10).*π
	#until k = 4 is good accordging to the paper
	
	cts = zeros(2,length(ks))
	cls = zeros(2,length(ks))

	for (i,k) in enumerate(ks)
	# for (i,freq) in enumerate(ctmvsk[:,1])
		
		garrick[:f] = k./π
		garrick[:thick] = 0.12
		θ0 = deg2rad(0)	
		# h0 = 0.01			
		h0 = 0.3/k  
		garrick[:motion_parameters] = [h0, θ0]
		
		foil, flow, wake, coeffs = run_sim(; garrick...)
		q∞ = 0.5*flow.Uinf^2
		cts[2,i] = mean(coeffs[3,flow.N:end]/q∞)
		cls[2,i] = maximum(coeffs[2,flow.N:end]/q∞)
		k = π*foil.f*foil.chord/flow.Uinf	
		theo = theod(k)
		F, G = real(theo), imag(theo)
		A = (F^2 + G^2)
		B = sqrt(F^2 + G^2 + k*G +k^2/4)
		Ctmean= 4*π*(k*h0)^2*A
		cts[1,i] = Ctmean
		Clpeak = 4*π*(k*h0)*B
		cls[1,i] = Clpeak 
	end
	# why 1.5?
	thrust = plot(ks, cts[1,:], marker=:utri, label="Garrick")
	plot!(thrust, ks , cts[2,:], marker =:circle, label="BEM")
	xlabel!(thrust, "k")
	ylabel!(thrust, "C_Tm}")
	lift = plot(ks ,cls[1,:], label="Garrick")
	plot!(lift,ks , cls[2,:]./1.5, marker =:circle, label="BEM")
	xlabel!(lift, "k")
	ylabel!(lift, "C_{Lp}")
	plot(thrust, lift, layout=(1,2), size=(800,400))
end

begin
	""" Fig 15 and 16 recreation
	Fig. 15 Mean thrust coefficient vs reduced frequency, N–S (Re =
	2 × 104, laminar), UPM, and Garrick14 analysis results, NACA 0012,
	plunging motion, kh = 0.3
	Fig. 16 Peak lift coefficient vs reduced frequency, N–S (Re = 2 × 104,
	laminar), UPM, and Garrick14 analysis results, NACA 0012, plunging
	motion, kh = 0.3"""
	ctmvsk= [0.216  0.632
			0.48 0.461
			1.007 0.355
			2.014 0.302
			3.022 0.282
			3.981 0.273
			4.988 0.268
			5.995 0.266
			7.962 0.257
			10.024 0.25
			12.038 0.25
			16.019 0.245
			19.928 0.239]
	plot(ctmvsk[:,1], ctmvsk[:,2], marker=:dtri)
	clpvsk = [  0.13 3.12
				0.42 2.8
				0.88 2.63
				1.87 4.28
				2.96 6.25
				3.9 8.06
				4.94 9.87
				5.97 11.35
				7.9 14.97
				9.92 18.75
				11.9 22.53
				16 29.11
				19.92 36.68]
	plot(clpvsk[:,1], clpvsk[:,2], marker=:dtri)
 end

begin	
"""try to make young aiaa2004 Fig 4."""
	young = deepcopy(defaultDict)
	young[:Nt] = 64
	young[:N] = 128
	young[:Ncycles] = 5
	young[:f] = 0.5
    young[:Uinf] = 1.0
	young[:kine] = :make_heave_pitch
	young[:pivot] = 0.25
	young[:thick] = 0.12
	θ0 = deg2rad(2)
	h0 = 0.0
	young[:motion_parameters] = [h0, θ0]


	foil, flow, wake, coeffs = run_sim(; young...)
	plot_current(foil, wake)
	plot_coeffs(coeffs/(0.5*flow.Uinf^2),flow)

end