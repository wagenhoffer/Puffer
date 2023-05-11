using BemRom
using SpecialFunctions
using Plots 

N_STEP = 150
N_CYC = 3

tau = range(0, length= N_STEP * N_CYC, stop=N_CYC)
θ0 = π / 20
h0 = 0 
foil, flow, wake, coeffs = run_sim(; steps=N_STEP*N_CYC, f=1.0, 
                motion=:make_heave_pitch, motion_parameters=[h0, θ0], nfoil=64)


begin      
        t = flow.Δt:flow.Δt:flow.N*N_CYC*flow.Δt
        ω = 2π * foil.f
        k = ω * foil.chord / flow.Uinf
        a = -1 #-1 is rotation about leading edge
        theo = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
        F, G = real(theo), imag(theo)
        R =  π * θ0 * (k^2 / 2 * (0.125 + a^2) + (0.5 + a) * (F - (0.5 - a) * k * G))
        I = -π * θ0 * (k / 2 * (0.5 - a) - (0.5 + a) * (G + (0.5 - a) * k * F))
        Φ = atan(I, R)
        gar = sqrt.(R .^ 2 + I .^ 2) # *  0.5* flow.ρ* flow.Uinf^2* foil.chord* 
        Cl = gar .* exp.(1im * ω * tau)
        M = gar .* exp.(1im * ω * t .+ Φ)
        den = foil.chord^3* foil.f^2* θ0*4
        plot(t, real(Cl), label="Garrick  \$C_L\$")
        plot!(t[50:end], coeffs[2, 50:end]/den, label ="BEM  \$C_L\$" )
        # plot!(t, real(M), label="Garrick  \$C_M\$")
end