using BemRom
using SpecialFunctions
using Plots 

N_STEP = 150
N_CYC = 4


θ0 = deg2rad(2.5)
h0 = 0.0
foil, flow, wake, coeffs = run_sim(; steps=N_STEP*N_CYC, f=1.0, 
                motion=:make_heave_pitch, motion_parameters=[h0, θ0], nfoil=64);
begin      

        RHO = flow.ρ = 1.0
        b = foil.chord/2.0
        PHI = 0.0
        t = flow.Δt:flow.Δt:flow.N*N_CYC*flow.Δt
        inst_lift = zeros(size(t))
        ω = 2π * foil.f
        k = ω * b / flow.Uinf
        a = -1.0 #-1 is rotation about leading edge
        theo = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
        F, G = real(theo), imag(theo)
        R =  π * θ0 * (k^2 / 2 * (0.125 + a^2) + (0.5 + a) * (F - (0.5 - a) * k * G))
        I = -π * θ0 * (k / 2 * (0.5 - a) - (0.5 + a) * (G + (0.5 - a) * k * F))
        Φ = atan(I, R)
        gar = sqrt.(R .^ 2 + I .^ 2) # *  0.5* flow.ρ* flow.Uinf^2* foil.chord* 
        Cl = gar .* exp.(-1im * ω * t)
        M = gar .* exp.(1im * ω * t .+ Φ)
        den = foil.chord^2* foil.f^2* (θ0*foil.chord + h0)
        @. inst_lift = -RHO*b^2*(flow.Uinf * pi* θ0  * ω  * cos(ω*t + PHI) - pi* h0 * ω ^2 * sin(ω *t ) +
                 pi* b * a * θ0  * ω^2 * sin(ω*t + PHI)) - 2.0*pi*RHO*flow.Uinf*b*F*(flow.Uinf*θ0 *sin(ω*t + PHI) + 
                h0*ω*cos(ω*t ) + b*(0.5-a)*θ0 *ω*cos(ω*t + PHI)) -
                2.0*pi*RHO*flow.Uinf*b*G*(flow.Uinf*θ0 *cos(ω*t + PHI) - h0*ω*sin(ω*t)- b*(0.5 - a)*θ0 *ω*sin(ω*t + PHI))

        plot(t, inst_lift, label="Garrick  \$C_L\$")
        plot!(t[75:end-75], coeffs[2, 150:end], marker=:circle, lw=0,ms=3,  label ="BEM  \$C_L\$" )
        # plot!(t, real(M), label="Garrick  \$C_M\$")
end

begin 
c_t = pi*k^2*θ0^2*((F^2 + G^2)*(1/k^2+(0.5-a)^2)+(0.5 - F)*(0.5-a)-F/k^2-(0.5+a)*G/k)
plot(t,c_t.*cos.(ω*t)/den/θ0)
plot!(t[50:end], coeffs[3,50:end])
end


L = zeros(size(t))
U = 1
@. L = -RHO * b^2 * (U * pi * θ0* ω * cos(ω * t + PHI) - pi * h0 * ω^2 * sin(ω * t) + pi * b * a * θ0* ω^2 * sin(ω * t + PHI)) - 2. * pi * RHO * U * b * F * (U * θ0* sin(ω * t + PHI) + h0 * ω * cos(ω * t) + b * (0.5 - a) * θ0* ω * cos(ω * t + PHI)) - 2. * pi * RHO * U * b * G * (U * θ0* cos(ω * t + PHI) - h0 * ω * sin(ω * t) - b * (0.5 - a) * θ0* ω * sin(ω * t + PHI))
# Path: scripts/garrickComp.jl