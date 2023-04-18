
using DelimitedFiles
using SpecialFunctions
using Plots

# Load input parameters
C = 1. 
N_STEP = 150 
N_CYC = 3
F = 1
tau = range(0, length=N_STEP*N_CYC, stop=5)

# Define constants
b = 0.5 * C
a = -1.0
w = 2.0 * pi *F
t = tau
RHO = 1000.
U = abs(1)
THETA_MAX = P["THETA_MAX"]
HEAVE_MAX = P["HEAVE_MAX"]
PHI = P["PHI"]

k = w * b / U
k2 = pi *F * C / U
St = 2.0 *F * P["HEAVE_MAX"] / abs(P["V0"])

F = (j1(k)*(j1(k)+y0(k)) + y1(k)*(y1(k)-j0(k))) / ((j1(k)+y0(k))^2 + (y1(k)-j0(k))^2)
G = -(y1(k)*y0(k) + j1(k)*j0(k)) / ((j1(k)+y0(k))^2 + (y1(k)-j0(k))^2)

L = -RHO * b^2 * (U * pi * THETA_MAX * w * cos(w * t + PHI) - pi * HEAVE_MAX * w^2 * sin(w * t) + pi * b * a * THETA_MAX * w^2 * sin(w * t + PHI)) -
    2.0 * pi * RHO * U * b * F * (U * THETA_MAX * sin(w * t + PHI) + HEAVE_MAX * w * cos(w * t) + b * (0.5 - a) * THETA_MAX * w * cos(w * t + PHI)) -
    2.0 * pi * RHO * U * b * G * (U * THETA_MAX * cos(w * t + PHI) - HEAVE_MAX * w * sin(w * t) - b * (0.5 - a) * THETA_MAX * w * sin(w * t + PHI))

Cl = real.(L) / (0.5 * P["RHO"] * abs(P["V0"])^2 * C[1])

# Call rigid_bem2d function
include("rigid_bem2d.jl")
rigid_bem2d.rigid(P)
expCsv = readdlm(P["OUTPUT_DIR"] * "forces0.csv", ',')

# Determine error
indx = (P["N_CYC"] - 1) * P["N_STEP"]
diff = abs.(expCsv[indx:end, 3] - Cl[indx:end])
println("Maximum Error = ", maximum(diff))

# Plot the results
plot(expCsv[:, 1] / (1.0 /F) - (P["N_CYC"] - 1), expCsv[:, 3], label="BEM", markershape=:circle)
plot!(expCsv[:, 1] / (1.0 /F) - (P["N_CYC"] - 1), Cl, label="Theodorsen")
xlabel!(L"$\tau = t/T$")
ylabel!(L"$C_l$")
xmin = (P["N_CYC"] - 1) * P["N_STEP"] * P
