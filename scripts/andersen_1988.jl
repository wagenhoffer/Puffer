using Puffer
using Plots

anderson = deepcopy(defaultDict)
anderson[:N] = 200
anderson[:Nt] = 50
anderson[:Ncycles] = 5
anderson[:pivot] = 1/3
anderson[:kine] = :make_heave_pitch
ks = 0.3:0.2:2.1
h1 = 0.25 

for k in ks
    α1 = deg2rad(15) - atan(2*k*h1)
    anderson[:motion_parameters] = [h1, α1]
    anderson[:f] = k*anderson[:Uinf]/(π*anderson[:chord])
    foil, flow = init_params(; anderson...)
    @show k
    wake = Wake(foil)
    
    steps = flow.N *flow.Ncycles        
    old_mus = zeros(3, foil.N)
    old_phis = zeros(3, foil.N)    
    coeffs = zeros(4, steps)

    (foil)(flow)

    for i in 1:flow.Ncycles * flow.N                    
        rhs = time_increment!(flow, foil, wake)
        phi = get_phi(foil, wake)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)
        coeffs[:,i] = get_performance(foil, flow, p)
        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
    end
end