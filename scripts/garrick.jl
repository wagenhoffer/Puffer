
using SpecialFunctions
using Plots

# Define local versions of variables from parameter list 
n_step = 150
n_cyc = 3
theta_max = π/20.0
heave_max = 0.1
ϕ = -π/2.0
ρ = 1000.
t = np.asarray(P['T']) #NATE?
α = theta_max #NATE?
# α
(ϕ_1, ϕ_2) = (ϕ,0)
tau = LinRange(0,5,n_step*n_cyc+1)
C = 1.0
f = 1.0
V0 = -1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Define constants used in paper                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

b = 0.5 * C            # b is defined as half the reference chord length
p = 2.0 * π *f      # Frequency of Oscillations in radians
v = abs(V0)
a = -1.0                      # Coordinate for axis of rotation (pitch)
k = p * b / v                 # Equation 2, Page 420, Reduced Frequency
k_from_moored = f*C/v
# Vector inits 
alpha_dot = zeros(size(t))
h_dot     = zeros(size(t))
Q         = zeros(size(t))
S         = zeros(size(t))
Inst_Lift = zeros(size(t))
M_a       = zeros(size(t))

@. alpha_dot = theta_max*cos(p*t + ϕ_1)*p
@. h_dot = heave_max*cos(p*t + ϕ_1)*p

# Real component of complex C(k), utilizes Bessel Functions of the First and Second Kind
F = (besselj1(k) * (besselj1(k) + bessely0(k)) + bessely1(k) * (bessely1(k) - besselj0(k)))/((besselj1(k) + bessely0(k))^2 + (bessely1(k) - besselj0(k))^2)
# Complex component of complex C(k), utilizes Bessel Functions of the First and Second Kind
G = - (bessely1(k)*bessely0(k) + besselj1(k)*besselj0(k))/((besselj1(k) + bessely0(k))^2 + (bessely1(k) - besselj0(k))^2)

C = F + 1im*G

# Q, Condition for smooth flow at leading edge, Equation 18, Page 422
@. Q = v*α+ h_dot + b*(0.5-a)*alpha_dot

# Constant 'M' defined on page 423
M = 2.0 * F * (v*theta_max*cos(ϕ_1) - heave_max*p*sin(ϕ_2) - b*(0.5 - a)*theta_max*p*sin(ϕ_1)) - 
    2.0 * G * (v*theta_max*sin(ϕ_1)+heave_max*p*cos(ϕ_2)+b*(0.5 - a)*theta_max*p*cos(ϕ_1)) + 
    b*theta_max*p*sin(ϕ_1)

# Constant 'N' defined on page 424
N = 2.0*F*(v*theta_max*sin(ϕ_1)+heave_max*p*cos(ϕ_2)+b*(0.5 - a)*theta_max*p*cos(ϕ_1)) + 
    2.0*G*(v*theta_max*cos(ϕ_1) - heave_max*p*sin(ϕ_2) - b*(0.5 - a)*theta_max*p*sin(ϕ_1)) - 
    b*theta_max*p*cos(ϕ_1)

# Define "a" constants - Page 424
a_1 = F^2 + G^2
a_2 = b^2*((F^2 + G^2)*(1/k^2+(0.5 - a)^2)+0.25-(0.5 - a)*F - 1/k*G)
a_4 = b*((F^2 + G^2)*(-1/k*sin(ϕ_2-ϕ_1)+(0.5-a)*cos(ϕ_2-ϕ_1))-F/2.0*cos(ϕ_2-ϕ_1)+G/2.0*sin(ϕ_2-ϕ_1))

# Define "b" constants - Page 424
b_2 = b^2*(-a/2.0-F/k^2+(0.5 - a)*G/k)
b_4 = b/2.0*((0.5+G/k)*cos(ϕ_2 - ϕ_1)+F/k*sin(ϕ_2-ϕ_1))

# Define "B" constants - Page 422
B_1 = F
B_2 = b^2*(0.5*(0.5-a)-(a+0.5)*(F*(0.5 - a)+G/k))
B_4 = b/2.0*((0.5-2*a*F+G/k)*cos(ϕ_2-ϕ_1)-(F/k - G)*sin(ϕ_2-ϕ_1))

# S represents the limit given on page 422. Evaluated based on equation 25 on page 423
@. S = sqrt(2.)/2.0*(M*sin(p*t) + N*cos(p*t))
# Full S equation
S_compare = imag(sqrt(2.)/2.0*(2.0*(F + 1im*G)*Q-b*alpha_dot))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Performance Calculations                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Instantaneous Lift Equation, Equation (8), Page 421
@. Inst_Lift = -RHO*b^2*(v * π * theta_max * p * cos(p*t + ϕ_1) - π * heave_max * p^2 * sin(p*t + ϕ_2) + 
    π * b * a * theta_max * p^2 * sin(p*t + ϕ_1)) - 2.0*π*RHO*v*b*F*(v*theta_max*sin(p*t + ϕ_1) + 
    heave_max*p*cos(p*t + ϕ_2) + b*(0.5-a)*theta_max*p*cos(p*t + ϕ_1)) - 
    2.0*π*RHO*v*b*G*(v*theta_max*cos(p*t + ϕ_1) - heave_max*p*sin(p*t + ϕ_2) - 
    b*(0.5 - a)*theta_max*p*sin(p*t + ϕ_1))

# Instantaneous Thrust Force, Equation (13), Page 422
P_x = π*RHO.*S_compare.^2 .+ α.*Inst_Lift
P_x_bar_compare = mean(P_x)

# Average Thrust Force, Equation (29), Page 425, VALIDATED BY JONES AND PLATZER
P_x_bar = π*RHO*b*p^2*(a_1*heave_max^2+(a_2+b_2)*theta_max^2+2*(a_4+b_4)*theta_max*heave_max)
C_t_bar = P_x_bar/(0.5 * RHO * abs(V0)^2 * C *b)
C_t_Jones_and_Platzer_Pitch_Only = π*k^2*theta_max^2*((F^2 + G^2)*(1/k^2+(0.5-a)^2)+(0.5 - F)*(0.5-a)-F/k^2-(0.5+a)*G/k)
C_t_from_s = π*RHO*S_compare.^2 ./(0.5 * RHO * v ^2 * C * b)
C_t_from_P = α*Inst_Lift/(0.5 * RHO * v^2 * C * b)

avg_s_thrust = π*RHO/4.0 *(M^2 + N^2)
avg_s_thrust_compare = mean(π*RHO*S.^2)
avg_s_thrust_compare_1 = mean(π*RHO*S_compare.^2)
avg_P_thrust = π*RHO*b*p^2*(b_2*theta_max^2+2*b_4*theta_max*heave_max)
avg_P_thrust_compare = mean(α*Inst_Lift)
P_x_bar_add_components = +avg_s_thrust + avg_P_thrust

# Average Work done in unit time, Equation (15), Page 422
W_bar = π*RHO*b^2*p^3/k*(B_1*heave_max^2 + B_2*theta_max^2 + 2*B_4*theta_max*heave_max)

# Average Power and Average Power Coefficient
Power_bar = W_bar/maximum(t)
C_power_bar = Power_bar/(0.5 * RHO * v^3 * C * b)

# Efficiency
eta = C_t_bar/C_power_bar

# Instantaneous Moment about the Leading Edge, Equation 9, Page 421
@. M_a = -RHO*b^2*(π*(0.5 - a)*v*b*theta_max*p*cos(p*t+ϕ_1)-π*b^2*(.125+a^2)*theta_max*p^2*sin(p*t+ϕ_1)+
      a*π*b*heave_max*p^2*sin(p*t+ϕ_2)) + 2.0*RHO*v*b^2*π*(a + 0.5)*F*(v*theta_max*sin(p*t + ϕ_1) +
      heave_max*p*cos(p*t + ϕ_2) + b*(0.5 - a)*theta_max*p*cos(p*t + ϕ_1)) + 
      2.0*RHO*v*b^2*π*(a + 0.5)*G*(v*theta_max*cos(p*t + ϕ_1)
      - heave_max*p*sin(p*t + ϕ_2)-b*(0.5-a)*theta_max*p*sin(p*t + ϕ_1))

# Instantaneous Power from Pitching ONLY
Pitching_Power = -M_a .* alpha_dot

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculating Coefficients                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
indx = (P['N_CYC'] - 1) * P['N_STEP']
# Instantaneous Coefficients
Cl = np.real(Inst_Lift) / (RHO * C^3 * f^2 * P['THETA_MAX']^2 * b)
C_l_bar = mean(Cl[indx:-1])
Ct = np.real(P_x) / (RHO * C^3 * f^2 * P['THETA_MAX']^2 * b)
Ct_average_test = mean(Ct)
Cp = np.real(Pitching_Power) / (RHO * C^3 * f^2 * P['THETA_MAX']^2 * v)
# Cp = np.real(Pitching_Power) / (0.5 * RHO * v^3 * P['C'] * b)
Cp_average_test = mean(Cp[indx:-1])



# Run Simulation --kill this shit and rewrite it all - aka just run our code
rigid_bem2d.rigid(P)

# Collect Results
expCsv = np.genfromtxt(P['OUTPUT_DIR']+'forces0.csv', delimiter=',')
exp_T = expCsv[:,3]*(0.5 * RHO * v^2 * C * b)
exp_C_T = expCsv[indx:-1,3]


time = np.asanyarray(P['T'])/(1. / f) -(P['N_CYC']-1)
########################################################################################################################
# Prof Moored's Inst. CT Results
m = cos(2.0*  π*time) + 1im*sin(2.0*π*time)

a_dot = 1im*P['THETA_MAX'] * p * m
a_d_dot = -P['THETA_MAX'] * p^2 * m

Moment_alpha = imag(- RHO * b^2 * (π * (0.5 - a) * v * b * a_dot + π * b^2 * (0.125 + a^2) * a_d_dot) + 2 * RHO * v * b^2 * π * (a + 0.5) * C * Q )
Power_alpha = imag(Moment_alpha * -1im * P['THETA_MAX'] * p * m)
Norm_Power_alpha = Power_alpha/(RHO * C^3 * f^2 * P['THETA_MAX']^2 * v)


test_thing = imag( ( ( 3. * π ^3 * (1 - C)/ 4.) + 1im*(9. * π ^4 *k_from_moored/16. + π^2*C/2./k_from_moored)) * m)
Power_star = imag(test_thing*m)

P_star = imag( ( ( π^2/2. - C/k_from_moored^2) - 1im*( π/2./k_from_moored + 3.*π*C/2./k_from_moored ) )*π*m )
alpha_P_star = imag(P_star*m)
S_star = imag( sqrt(2.)/2.0*( 2.0*C/k_from_moored + 1im*( 3.*π*C - π ))*m )

inst_C_T_moored = π/2.0*S_star^2 + alpha_P_star # use the imaginary part--associated with sin(omega*t) motion.

#######################################################################################################################


#Plot the results
figure = plt.figure(1)
figure.add_subplot(1, 1, 1) # Change background color here
plt.tick_params(labelsize=20)
plot(time, expCsv[:,2],'o',linewidth=3)
plot(time, P_star,linewidth=3)
plt.ylabel(r'$ C_L $', fontsize=20)

xmin = (P['N_CYC'] - 1) * P['N_STEP'] * P['DEL_T'] /(1. / f)  - (P['N_CYC']-1)
xmax = P['N_CYC'] * P['N_STEP'] * P['DEL_T'] /(1. / f) - (P['N_CYC']-1)
ymin = minimum(P_star[-250:-5]) + 0.125*minimum(P_star[-250:-5])
ymax = maximum(P_star[-250:-5]) + 0.125*maximum(P_star[-250:-5])
plt.axis([xmin, xmax, ymin, ymax])

figure = plt.figure(2)
figure.add_subplot(1, 1, 1) # Change background color here
plt.tick_params(labelsize=20)
plot(time, expCsv[:,3],'o',linewidth=3)
plot(time, inst_C_T_moored,'-',linewidth=3)
plt.legend(('BEM','Garrick - "S" component','Garrick - "P" component','Garrick Total'), fontsize=10)
plt.ylabel(r'$ C_T $', fontsize=20)
xmin = (P['N_CYC'] - 1) * P['N_STEP'] * P['DEL_T'] /(1. / f)  - (P['N_CYC']-1)
xmax = P['N_CYC'] * P['N_STEP'] * P['DEL_T'] /(1. / f) - (P['N_CYC']-1)
ymin = minimum(inst_C_T_moored[-100:-1]) + 0.125*minimum(inst_C_T_moored[-100:-1])
ymax = maximum(inst_C_T_moored[-100:-1]) + 0.125*maximum(inst_C_T_moored[-100:-1])
plt.axis([xmin, xmax, ymin, ymax])

plot(time, expCsv[:,4],'o',linewidth=3)
plot(time, Cp,linewidth=3)

plt.ylabel(r'$ C_{power} $', fontsize=20)
xmin = (P['N_CYC'] - 1) * P['N_STEP'] * P['DEL_T'] /(1. / f)  - (P['N_CYC']-1)
xmax = P['N_CYC'] * P['N_STEP'] * P['DEL_T'] /(1. / f) - (P['N_CYC']-1)
ymin = minimum(Cp[-100:-1]) + 0.125*minimum(Cp[-100:-1])
ymax = maximum(Cp[-100:-1]) + 0.125*maximum(Cp[-100:-1])
plt.axis([xmin, xmax, ymin, ymax])






