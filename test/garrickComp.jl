using Puffer

using SpecialFunctions
using Statistics
using Plots
using Test
using LinearAlgebra

theo(k) = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))

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


function moored_teardrop(;Nt = 64, N = 64)
	moored = deepcopy(defaultDict)
	moored[:Nt] = Nt
	moored[:N] = N

	moored[:Ncycles] = 5
	moored[:f] = 0.5
	moored[:Uinf] = 1.0
	moored[:kine] = :make_heave_pitch
	moored[:pivot] = 0.25
	moored[:thick] = 0.001
	moored[:foil_type] = :make_teardrop
	θ0 = deg2rad(0)
	h0 = 0.01
	moored[:motion_parameters] = [h0, θ0]


	foil, flow, wake, coeffs = run_sim(; moored...)
	coeffs ./= (0.5*flow.Uinf^2)


	# Change thickness to 0.1%
	#pure heaving compare - Quinn, moored
	q∞ = 0.5*flow.Uinf^2
	St = 2*h0*foil.f/flow.Uinf
	k = π*foil.f*foil.chord/flow.Uinf
	# @show St, k
	τ =	 collect(flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt) .*foil.f
	cl = zeros(size(τ))
	ck = theo(k)
	@. cl = -2*π^2*St*abs.(ck)*cos(2π.*τ + angle(ck)) - π^2*St*k*sin(2π.*τ)

	shift = -2
	plot(τ[flow.N:end], coeffs[2, flow.N+shift:end+shift], marker=:circle, label="BEM Lift")
	plt = plot!(τ[flow.N:end], cl[flow.N:end], label="Theo Lift",lw=3,ylims=(-0.25,0.25))
	[cl[flow.N:end], coeffs[2,flow.N+shift:end+shift]], plt
end

 function young1516()
	""" Fig 15 and 16 recreation
	Fig. 15 Mean thrust coefficient vs reduced frequency, N–S (Re =
	2 × 104, laminar), UPM, and Garrick14 analysis results, NACA 0012,
	plunging motion, kh = 0.3
	Fig. 16 Peak lift coefficient vs reduced frequency, N–S (Re = 2 × 104,
	laminar), UPM, and Garrick14 analysis results, NACA 0012, plunging
	motion, kh = 0.3"""
	upm = deepcopy(defaultDict)
	upm[:Nt]        = 64
	upm[:N]         = 64
	upm[:Ncycles]   = 5
	upm[:f]         = 0.5
    upm[:Uinf]      = 1.0
	upm[:kine]      = :make_heave_pitch
	upm[:pivot]     = 0.0
	upm[:foil_type] = :make_naca
	upm[:thick]     = 0.12
	θ0 = deg2rad(0)	
	ks = LinRange(0.1, 4.0/π, 8).*π
	#until k = 4 is good accordging to the paper
	
	cts = zeros(2,length(ks))
	cls = zeros(2,length(ks))

	for (i,k) in enumerate(ks)
	# for (i,freq) in enumerate(ctmvsk[:,1])
		upm[:f] = k./π				
		h0 = 0.3/k  
		upm[:motion_parameters] = [h0, θ0]
		
		foil, flow, wake, coeffs = run_sim(; upm...)
		q∞ = 0.5*flow.Uinf^2
		cts[2,i] = mean(coeffs[3,flow.N:end]/q∞)
		cls[2,i] = maximum(coeffs[2,flow.N:end]/q∞)
		k = π*foil.f*foil.chord/flow.Uinf	
		theod = theo(k)
		F, G = real(theod), imag(theod)
		A = (F^2 + G^2)
		B = sqrt(F^2 + G^2 + k*G +k^2/4)
		Ctmean= 4*π*(k*h0)^2*A		
		Clpeak = 4*π*(k*h0)*B
		cts[1,i] = Ctmean
		cls[1,i] = Clpeak 
	end

	thrust = plot(ks, cts[1,:], marker=:utri, label="Garrick")
	plot!(thrust, ks , cts[2,:], marker =:circle, label="BEM")
	xlabel!(thrust, "k")
	ylabel!(thrust, "C_Tm}")
	lift = plot(ks ,cls[1,:], label="Garrick")
	plot!(lift,ks , cls[2,:], marker =:circle, label="BEM")
	xlabel!(lift, "k")
	ylabel!(lift, "C_{Lp}")
	plot(thrust, lift, layout=(1,2), size=(800,400))
end

"""try to make young aiaa2004 Fig 4."""
function young4()
	young = deepcopy(defaultDict)
	young[:Nt] = 64
	young[:N] = 64
	young[:Ncycles] = 6
	young[:f] = 8.0/π
    young[:Uinf] = 1.0
	young[:kine] = :make_heave_pitch
	young[:pivot] = 0.25
	young[:thick] = 0.12
	θ0 = deg2rad(2)
	h0 = 0.0
	young[:motion_parameters] = [h0, θ0]

	
	foil, flow, wake, coeffs = run_sim(; young...)
	k = π*foil.f/flow.Uinf
	@show rad2deg(θ0), k
	plot_current(foil, wake)
	plot_coeffs(coeffs./(0.5*flow.Uinf^2), flow)
end

function young17()
	"""try to make young aiaa2004 Fig 17.
	
	"""
	young = deepcopy(defaultDict)
	young[:Nt] = 64
	young[:N] = 64
	young[:Ncycles] = 3
	
	young[:Uinf] = 1.0
	young[:kine] = :make_heave_pitch
	young[:pivot] = 0.25
	young[:thick] = 0.12
	q∞ = 0.5*young[:Uinf]^2
	ks = [2 4 8].|>Float64
	cps = zeros(length(ks), young[:N])
	young[:motion_parameters] = [0.0, 0.0]
	young[:f] = young[:Uinf]/π
	foil, flow = init_params(;young...)
	for (i,k) in enumerate(ks)
		
		θ0 = deg2rad(0)
		h0 = 0.3/k
		young[:motion_parameters] = [h0, θ0]
		young[:f] = k*young[:Uinf]/π
		
		foil, flow = init_params(;young...)
		
		
		wake = Wake(foil)
		(foil)(flow)
		#data containers
		old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
		phi = zeros(foil.N)
		coeffs = zeros(4,flow.Ncycles*flow.N)
		p = zeros(foil.N)
		### EXAMPLE OF AN PERFROMANCE METRICS LOOP
		for i in 1:(flow.Ncycles-1)*flow.N + flow.N÷4 -1
			time_increment!(flow, foil, wake)
			phi =  get_phi(foil, wake)                                   
			p = panel_pressure(foil, flow,  old_mus, old_phis, phi)        
			old_mus = [foil.μs'; old_mus[1:2,:]]
			old_phis = [phi'; old_phis[1:2,:]]
			coeffs[:,i] = get_performance(foil, flow, p)
	
		end
		cps[i,:] = p/q∞
	end
	# cps ./= q∞
	plot(foil.col[1,:], cps[1, :], label="k = 2")
	plot!(foil.col[1,:], cps[2, :], label="k = 4")
	plot!(foil.col[1,:], cps[3, :], label="k = 8")
	plot!(ylims=(-11,11),yflip=true)
end


####Run each of the cases
# young4 is suspect
####

if @isdefined VISUALIZE
	if VISUALIZE
		_, plt = moored_teardrop()
		savefig(plt,"./images/theodorsen_teardrop_comparison.png")
		plt = young4()
		savefig(plt,"./images/young4_comparison.png")
		plt = young1516()
		savefig(plt,"./images/young15_16_comparison.png")
		plt = young17()
		savefig(plt,"./images/young17_comparison.png")
	end
end


@testset "Theodorsen Comparison on Moored's configuration" begin
	# Define a tolerance for the error
	tolerance = 0.06 #<_ probably high
    # Perform the simulation
    data, plt= moored_teardrop()
    cl, cl_sim = data
    # Check the results against the experimental data
    @test length(cl) == length(cl_sim)    
	@test norm(cl-cl_sim,2)/norm(cl,2) < tolerance

end
