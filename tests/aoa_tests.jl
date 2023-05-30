# using BemRom
include("../src/BemRom.jl")

using Plots
begin
	#DO a sweep and compare with experimental results
	# https://turbmodels.larc.nasa.gov/NACA0012_validation/CL_Gregory_expdata.dat
	gregory = [0.0943692 0.0257134
			1.97212   0.231092
			2.8876    0.333789
			5.93866   0.658169
			9.03657   0.984876
			12.1339   1.29286
			13.1421   1.36745
			14.127    1.44673
			15.1349   1.51429
			15.6742   1.55392
			16.0955   1.56081
			17.1059   0.956775
			17.9695   0.89568]
	cps = []  
	gregComp = deepcopy(defaultDict)
	gregComp[:f] = 1.0
	gregComp[:Uinf] = 5000.0
	gregComp[:kine] = :no_motion
	gregComp[:Ncycles] = 1

	coeffs = zeros(size(gregory)[1])

	for i ∈ 1:size(gregory)[1]
		gregComp[:aoa] = -(gregory[i, 1] * pi / 180)
		@show gregComp[:aoa]
		foil, flow, wake, perf = run_sim(; gregComp...)

		coeffs[i] = perf[2]
		a = plot_current(foil, wake)
		a
	end
	#normalize the coefficients
	coeffs ./= (0.5 * gregComp[:Uinf]^2)
	err = abs.(coeffs - gregory[:, 2]) ./ gregory[:, 2]

	a = plot(gregory[:, 1], coeffs, label = "BEM", shape = :circle, color = :blue)
	plot!(a, gregory[:, 1], gregory[:, 2], label = "Gregory", shape = :circle, color = :red)
	plot!(a, xlabel = "Angle of Attack (deg)", ylabel = "C_L", legend = :bottomright)
	b = plot(gregory[:, 1], err,
		label = "L_2 Error", shape = :circle, color = :blue,
		yscale = :log10)
	plot!(b, xlabel = "Angle of Attack (deg)", ylabel = "L_2 Error", legend = :bottomright, yticks = [1e-3, 1e-2, 1e-1, 1e0])

	plot(a, b, layout = (2, 1), size = (800, 800))
end


begin
	#look at the coefficicent of pressure for a single angle of attack
	movie  = @animate for α = 1:8
		aoaComp = deepcopy(defaultDict)
		aoaComp[:f] = 0.1
		aoaComp[:Uinf] = 20.0
		aoaComp[:kine] = :no_motion
		aoaComp[:Ncycles] = 1
		aoaComp[:aoa] = -(α* pi / 180)

		foil, flow = init_params(;aoaComp...)
		foil._foil = (foil._foil' * rotation(aoaComp[:aoa])')'
		wake = Wake(foil)
		(foil)(flow)
		#data containers
		old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
		phi = zeros(foil.N)
		cps = []
		for i in 1:flow.Ncycles*flow.N
			time_increment!(flow, foil, wake)
			phi =  get_phi(foil, wake)
			p, old_mus, old_phis = panel_pressure(foil, flow, old_mus, old_phis, phi)
			# coeffs[:,i] .= get_performance(foil, flow, p)
		
			push!(cps, p)
		end
		mid = foil.N ÷2 +1

		f = plot(foil.col[1,1:mid], cps[end][1:mid]./(0.5*flow.Uinf^2),marker=:circle, label="Bottom")
		plot!(foil.col[1,mid:end], cps[end][mid:end]./(0.5*flow.Uinf^2),marker=:circle, label="Top")
		plot!(xlabel="x/c", ylabel="C_p", legend=:bottomright, title!("C_p vs x/c for α = $(α)"))
		plot!(ylim=(-1.5,5.0))
		f
	end
	gif(movie, "aoa.gif", fps = 2)
end



