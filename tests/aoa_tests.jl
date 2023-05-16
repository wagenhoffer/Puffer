# using BemRom
include("../src/BemRom.jl")

using Plots
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
gregComp[:f] = 0.1
gregComp[:Uinf] = 500.0
gregComp[:kine] = :no_motion
gregComp[:Ncycles] = 10

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
error = abs.(coeffs - gregory[:, 2]) ./ gregory[:, 2]

a = plot(gregory[:, 1], coeffs, label = "BEM", shape = :circle, color = :blue)
plot!(a, gregory[:, 1], gregory[:, 2], label = "Gregory", shape = :circle, color = :red)
plot!(a, xlabel = "Angle of Attack (deg)", ylabel = "C_L", legend = :bottomright)
b = plot(gregory[:, 1], error,
	label = "L_2 Error", shape = :circle, color = :blue,
	yscale = :log10)
plot!(b, xlabel = "Angle of Attack (deg)", ylabel = "L_2 Error", legend = :bottomright, yticks = [1e-3, 1e-2, 1e-1, 1e0])

plot(a, b, layout = (2, 1), size = (800, 800))
