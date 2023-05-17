using Symbolics
@variables  t, ω, θ0, h0, a, b, F, G, R, I, Φ, U, ρ, k, den, C, ppii
h = h0*sin(ω*t)
θ = θ0*sin(ω*t + Φ)
D = Differential(t)
hd = D(h)
hdd =D(hd)
θd = D(θ)
θdd = D(θd)
# L = -ρ * b^2 * (U * pi * θ0* ω * cos(ω * t + Φ) - pi * h0 * ω^2 * sin(ω * t) + pi * b * a * θ0* ω^2 * sin(ω * t + Φ)) - 2. * pi * RHO * U * b * F * (U * θ0* sin(ω * t + Φ) + h0 * ω * cos(ω * t) + b * (0.5 - a) * θ0* ω * cos(ω * t + Φ)) - 2. * pi * RHO * U * b * G * (U * θ0* cos(ω * t + Φ) - h0 * ω * sin(ω * t) - b * (0.5 - a) * θ0* ω * sin(ω * t + Φ))   
L = ppii*ρ*b^2*(hdd + U*θd -b*a*θdd) +2*ppii*ρ*U*b*C*(hd +U*θ + b*(0.5-a)*θd)
Lqs = ppii*ρ*b^2*(hdd + U*θd -b*a*θdd) +2*ppii*ρ*U*b*(hd +U*θ + b*(0.5-a)*θd)