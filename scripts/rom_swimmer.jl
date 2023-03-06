using LinearAlgebra
using Plots
using BenchmarkTools
using StaticArrays

########################################################################
"""
Reduced order model swimmer
requirements: 
1. vortex based
2. encodes transient traveling wave motion
3. includes a pressure solver -> finds efficiency, thrust, power, lift, etc …
    a. Neural Net pressure solver, the remainder of the system is linear per time step and
       this can be avoided and sped up sig if done right
4. easy to setup discrete motions -> faster, slower, left, right
5. invertable? or just attach a FMM code for acceleration
6. Write a version of Tianjuan's BEM to scrape inviscid data from 
7. POD/PCA or DMD on the vortex distro to define a rom
8. ROM -> P(NN) -> Hydrodynamic performance metrics  
9. Emit a vortex at reg interval

P(Γ((x,y)(t))) we want to reduce Γ while still getting pressure along the body

"""
"""
#rows left right
vort = [-1 1 
         0.5  -0.5]
# cols i, rows x y                  
pos  = [0.0 0.0
        0.5 0.0]  

δ = 0.25
θ = π/2.0
θr = θ - π/2.0
θl = θ + π/2.0  
left = [(pos[1,:] .+ δ*cos(θl))'
        (pos[2,:] .+ δ*sin(θl))']
right = [(pos[1,:] .+ δ*cos(θr))'
         (pos[2,:] .+ δ*sin(θr))']        

sources = [left right]

function streamfunction(sources,gammas,targets)
    """
    """
    pot = zeros((size(targets[1])...))

    for i in 1:size(sources)[2]

        dx = targets[1] .- sources[1,i]
        dy = targets[2] .- sources[2,i]
        @. pot += gammas[i] *log(sqrt(dx^2+dy^2))            
    end
    pot./(2π)
end


xs = LinRange(-2,2,31)
ys = LinRange(-2,2,31)

X = repeat(reshape(xs, 1, :), length(ys), 1)
Y = repeat(ys, 1, length(xs))
targets = [X,Y]

stream = streamfunction(sources,vort, targets)



plot(collect(xs),collect(ys), stream, st=:contourf,c=:redsblues)
plot!(pos[1,:],pos[2,:],seriestype=:scatter, label="top")
plot!(left[1,:],left[2,:],seriestype=:scatter,label="left")
plot!(right[1,:],right[2,:],seriestype=:scatter,label="right")   
"""

a0 = 0.1
a = [0.367,0.323,0.310]
f = k = 1

amp(x,a) = a[1] + a[2]*x + a[3]*x^2
h(x,f,k,t) = a0*amp(x,a)*sin(2π*(k*x - f*t))
h(x,t) = f,k -> h(x,f,k,t)
#no used
# ang = x -> h(x,a)

plot(h.(LinRange(0,1,64),1.0,0.5,1.0))

begin 
    n = 25
    T = LinRange(0, 1, n)
    x = LinRange(0,1,25)
    anim = @animate for t in T
        plot(x,h.(x,t), xlim=(-0.1,1.1), ylim=(-2*a0,2*a0),aspect_ratio=:equal)
    end
    gif(anim,"ang.gif", fps=12)
end

begin
    plot()
    for t in LinRange(0, 1, 10)
        plot!(x,h.(x,t), 
            xlim=(-0.1,1.1), ylim=(-2*a0,2*a0),
            aspect_ratio=:equal, label="")
    end
    plot!()
end



# begin 
#     n = 25
#     T = LinRange(0, 1, n)
    
#     anim = @animate for t in T
#         vorts = vort .* (sin(2π*t))/2.
#         @show vort
#         stream = streamfunction(sources,vorts,targets)
#         plot(collect(xs),collect(ys), stream, st=:contourf,c=:redsblues)        
#         plot!(left[1,:],left[2,:],seriestype=:scatter,label="")
#         plot!(right[1,:],right[2,:],seriestype=:scatter,label="")  
#     end
#     gif(anim,"simple.gif", fps=12)
# end

#Deforming NACA0012 Foil
function make_naca(N;chord=1,T=0.12)
    # N = 7
    an = [0.2969, -0.126, -0.3516, 0.2843, -0.1036]
    # T = 0.12 #thickness
    yt(x_)  = T/0.2*(an[1]*x_^0.5 + an[2]*x_ + an[3]*x_^2 + an[4]*x_^3 + an[5]*x_^4)
    #neutral x
    x = (1 .- cos.(LinRange(0, pi, (N+2)÷2)))/2.0
    foil = [[x[end:-1:1];  x[2:end]]'
            [-yt.(x[end:-1:1]);  yt.(x[2:end])]']

            # yt.(x[end:-1:1])
    # foil_col = (foil[:,2:end] + foil[:,1:end-1])./2
    foil.*chord
end
function make_waveform(a0 = 0.1, a = [0.367,0.323,0.310] )
    a0 = 0.1
    a = [0.367,0.323,0.310]
    f = k = 1

    amp(x,a) = a[1] + a[2]*x + a[3]*x^2
    h(x,f,k,t) = a0*amp(x,a)*sin(2π*(k*x - f*t))
    # h(x,t) = f,k -> h(x,f,k,t)
    h
end
make_waveform()
get_collocation(foil) = (foil[:,2:end] + foil[:,1:end-1])./2

# plot(foil[1,:], foil[2,:], aspect_ratio=:equal, marker=:circle)
# plot!(foil_col[1,:],foil_col[2,:], aspect_ratio=:equal, marker=:star,color=:red)

#traveling wave
foil[2,:] .+ h.(foil[1,:],0)

abstract type Body end
    
mutable struct FlowParams
    Δt #time-step
    Uinf
    ρ #fluid density
    N #number of time steps
    n #current time step
end
mutable struct Foil <: Body
    kine  #kinematics: heave or traveling wave function
    f #wave freq
    k # wave number
    N # number of elements
    _foil #coordinates in the Body Frame
    foil  # Absolute fram
    col  # collocation points
    μs #doublet strengths
    edge
    chord
    normals
    tangents
    panel_lengths
end
function norms(foil)
    dxdy = diff(foil, dims=2)
    lengths = sqrt.(sum(abs2,diff(foil,dims=2),dims=1))
    tx = dxdy[1,:]'./lengths
    ty = dxdy[2,:]'./lengths
    # tangents x,y normals x, y  lengths
    return [tx; ty], [-ty; tx], lengths
end  

source_inf(x1, x2, z) = (x1*log(x1^2 + z^2) - x2*log(x2^2 + z^2) - 2*(x1-x2)
                        + 2*z*(atan(z,x2) -atan(z,x1)))/(4π)
doublet_inf(x1, x2, z) = -(atan(z,x2) - atan(z,x1))/(2π)

function panel_frame(target,source)
    _, Ns = size(source)
    _, Nt = size(target)
    Ns -= 1 #TE accomodations
    x1 = zeros(Ns, Nt)
    x2 = zeros(Ns, Nt)
    y = zeros(Ns, Nt)
    tx,ty,nx,ny,lens = norms(source)
    txMat = repeat(tx, Nt, 1)
    tyMat = repeat(ty, Nt, 1)
    dx = repeat(target[1,:],1,Ns) - repeat(source[1,1:end-1]',Nt,1)
    dy = repeat(target[2,:],1,Ns) - repeat(source[2,1:end-1]',Nt,1)
    x1 = dx.*txMat + dy.*tyMat
    y  = -dx.* tyMat + dy.*txMat
    x2 = x1 - repeat(sum(diff(source,dims=2).*[tx;ty],dims=1),Nt,1)
    return x1, x2, y
end

function init_params()
    N = 128
    chord = 1.0
    naca0012 = make_naca(N+1;chord=chord)
    col = get_collocation(naca0012)
    ang = make_waveform()
    fp = FlowParams(0.01, 1, 1000., 100)
    txty, nxny, ll = norms(naca0012)
    edge_vec = fp.Uinf*fp.Δt*[(txty[1,end] - txty[1,1])/2.,(txty[2,end] - txty[2,1])/2.]
    edge = [naca0012[:,end]  (naca0012[:,end] .+ edge_vec) (naca0012[:,end] .+ 2*edge_vec) ] 
    Foil(ang,1,1,N,naca0012,naca0012,col,zeros(N-1),edge,1,txty,nxny,ll) , fp
end

foil, flow = init_params()

function (foil::Foil)(fp::FlowParams) 
    foil.foil += foil.h.(foil.foil,foil.f, foil.k, fp.n*fp.Δt)
    fp.n += 1
    
end

function panel_frame(target,source)
    _, Ns = size(source)
    _, Nt = size(target)
    Ns -= 1 #TE accomodations
    x1 = zeros(Ns, Nt)
    x2 = zeros(Ns, Nt)
    y = zeros(Ns, Nt)
    ts,ns,lens = norms(source)
    txMat = repeat(ts[1,:]', Nt, 1)
    tyMat = repeat(ts[2,:]', Nt, 1)
    dx = repeat(target[1,:],1,Ns) - repeat(source[1,1:end-1]',Nt,1)
    dy = repeat(target[2,:],1,Ns) - repeat(source[2,1:end-1]',Nt,1)
    x1 = dx.*txMat + dy.*tyMat
    y  = -dx.* tyMat + dy.*txMat
    x2 = x1 - repeat(sum(diff(source,dims=2).*ts,dims=1),Nt,1)
    x1, x2, y
end

begin 
    n = 25
    T = LinRange(0, 1, n)
    
    anim = @animate for t in T        
        plot(foil[1,:], foil[2,:] .+ h.(foil[1,:],t),label="",aspect_ratio=:equal)        
    end
    gif(anim,"simple.gif", fps=12)
end



function foil_dx_dy(N,foil)
    dx = zeros(N,N)
    dy = zeros(N,N)
    for i=1:N
        @inbounds dx[:, i] = foil[1,:] .- foil[1,i]
        @inbounds dy[:, i] = foil[2,:] .- foil[2,i]
    end
    [dx,dy]
end
dx,dy = foil_dx_dy(N,foil)
# Find distance matrices -> first with on swimmer and then with several
#target - source
function this(N,foil)
    dx = zeros(N,N)
    dy = zeros(N,N)
    for i=1:N
        @inbounds dx[:, i] = foil[1,:] .- foil[1,i]
        @inbounds dy[:, i] = foil[2,:] .- foil[2,i]
    end
    [dx,dy]
end
function that(N, foil)
    dx = repeat(foil[1,:],1,N) - repeat(foil[1,:]',N,1)
    dy = repeat(foil[2,:],1,N) - repeat(foil[2,:]',N,1)
    nothing
end

@btime this(N,foil) #N=2001 35.652 ms (8008 allocations: 184.69 MiB)
@btime that(N,foil) #N=2001 118.981 ms (20 allocations: 183.35 MiB)

function norms(foil)
    dxdy = diff(foil, dims=2)
    lengths = sqrt.(sum(abs2,diff(foil,dims=2),dims=1))
    tx = dxdy[1,:]'./lengths
    ty = dxdy[2,:]'./lengths
    # tangents x,y normals x, y  lengths
    return tx, ty, -ty, tx, lengths
end  

tx,ty, nx, ny, lengths = norms(foil)
@assert [nx,ny]⋅[tx,ty] == 0.0
plot(foil[1,:], foil[2,:], aspect_ratio=:equal, marker=:circle)
plot!(foil_col[1,:],foil_col[2,:], aspect_ratio=:equal, marker=:star,color=:red)
quiver!(foil_col[1,:], foil_col[2,:], quiver=(nx[1:end],ny[1:end]))

plot(foil[1,:], foil[2,:], aspect_ratio=:equal, marker=:circle)
quiver!(foil_col[1,:], foil_col[2,:], quiver=(tx[1:end], ty[1:end]))


begin 
    move = copy(foil)
    move[2,:] += h.(foil[1,:], 0.25)
    tx,ty,nx,ny,ll = norms(move)
    col = get_collocation(move)

    plot(move[1,:], move[2,:], aspect_ratio=:equal, marker=:circle)
    plot!(col[1,:], col[2,:], aspect_ratio=:equal, marker=:star,color=:red)
    quiver!(col[1,:], col[2,:], quiver=(nx[1:end],ny[1:end]),length=0.1)
end  

source_inf(x1, x2, z) = (x1*log(x1^2 + z^2) - x2*log(x2^2 + z^2) - 2*(x1-x2)
                        + 2*z*(atan(z,x2) -atan(z,x1)))/(4π)
doublet_inf(x1, x2, z) = -(atan(z,x2) - atan(z,x1))/(2π)

tx,ty,nx,ny,ll = norms(foil)
##CHANGE THE PANEL FRAMES -endpnts are the sources, col are targets 
x1 = zeros(N-1, N-1)
x2 = zeros(N-1, N-1)
y = zeros(N-1, N-1)
txMat = repeat(tx,N-1,1)
tyMat = repeat(ty,N-1,1)
dx = repeat(foil_col[1,:],1,N-1) - repeat(foil[1,1:end-1]',N-1,1)
dy = repeat(foil_col[2,:],1,N-1) - repeat(foil[2,1:end-1]',N-1,1)
# for i=1:N-1
#     x1[:,i] = (foil_col[1,:] .- foil[1,i])
#     x2[:,i] = (foil_col[2,:] .- foil[2,i])    
# end
# dx = x1
# dy = x2
x1 = dx.*txMat + dy.*tyMat
y =  dx.*-tyMat+ dy.*txMat

x2 = x1 - repeat(sum(diff(foil,dims=2).*[tx;ty],dims=1),N-1,1)

#Grabbed from python code
#strictly for testing though y here and z there are off 
pyx1 = [[ 0.12596995, -0.12568028, -0.59794013,  0.84768749,  0.6282324 ,
0.12983482]
[ 0.50176151,  0.25039739, -0.23996686,  0.47597968,  0.25550035,
-0.23859873]
[ 0.87194163,  0.62392359,  0.12848079,  0.11474624, -0.11978317,
-0.61264372]
[ 0.86458362,  0.62057795,  0.14221534,  0.12848079, -0.1231288 ,
-0.62000172]
[ 0.49053864,  0.24529443, -0.2190181 ,  0.49692843,  0.25039739,
-0.2498216 ]
[ 0.12210508, -0.12743761, -0.59072591,  0.8549017 ,  0.62647507,
0.12596995]] 
pyx1 = reshape(pyx1,(6,6))
pyx1' .-x1

py_z = [ 1.66533454e-17, -3.12043904e-02, -5.94075000e-02, -0.00000000e+00,
5.94075000e-02,  3.12043904e-02, -1.66533454e-17]

py_z .- foil[2,:]
py_foil = [1.00000000e+00  7.50000000e-01  2.50000000e-01 0.00000000e+00  2.50000000e-01   7.50000000e-01  1.00000000e+00;
           1.66533454e-17 -3.12043904e-02 -5.94075000e-02 -0.00000000e+00  5.94075000e-02  3.12043904e-02 -1.66533454e-17]

py_col = get_collocation(py_foil)           



x1,x2,y  = panel_frame(py_col, py_foil)

x1,x2,y = panel_frame(foil_col,foil)
doubletMat = doublet_inf.(x1,x2,y)
sourceMat = source_inf.(x1,x2,y)
plot(x1, st=:contourf)
plot(x2, st=:contourf)
plot(y, st=:contourf)


plot!(col[1,:], col[2,:], aspect_ratio=:equal, marker=:star,color=:red)
quiver!(col[1,:], col[2,:], quiver=(nx[1:end],ny[1:end]),length=0.1)

#define the edge off of the TE
Uinf = 1. 
Δt = 0.1
#use this def to allow for implicit kutta in the future
edge_vec = Uinf*Δt*[(tx[end] - tx[1])/2.,(ty[end] - ty[1])/2.]
edge = [foil[:,end] foil[:,end] .+ edge_vec foil[:,end] .+ 2*edge_vec ] 

plot(foil[1,:], foil[2,:], aspect_ratio=:equal, marker=:circle)
plot!(edge[1,:],edge[2,:], aspect_ratio=:equal, marker=:star,color=:green)



sigmas = sum([Uinf,0] .* [nx;ny], dims = 1)
rhs = - sourceMat*sigmas'
doubletMat\rhs
#this all matches to now 

#make the edge doublet work
panel_frame(foil,edge)

##CHANGE THE PANEL FRAMES -endpnts are the sources, col are targets 
_, Ns = size(source)
_, Nt = size(target)
Ns -= 1 #TE accomodations
x1 = zeros(Ns, Nt)
x2 = zeros(Ns, Nt)
y = zeros(Ns, Nt)
tx,ty,nx,ny,lens = norms(source)
txMat = repeat(tx, Nt, 1)
tyMat = repeat(ty, Nt, 1)
dx = repeat(target[1,:],1,Ns) - repeat(source[1,1:end-1]',Nt,1)
dy = repeat(target[2,:],1,Ns) - repeat(source[2,1:end-1]',Nt,1)
x1 = dx.*txMat + dy.*tyMat
y  = -dx.* tyMat + dy.*txMat
x2 = x1 - repeat(sum(diff(source,dims=2).*[tx;ty],dims=1),Nt,1)

edgeInf = doublet_inf.(x1,x2,y)

edgeMat = zeros(size(doubletMat))
edgeMat[:,1] = -edgeInf[:,1]
edgeMat[:,end] = edgeInf[:,1]
A = doubletMat + edgeMat
μ = A\rhs


function make_infs(foil :: Foil)
    x1,x2,y    = panel_frame(foil.col, foil.foil)
    doubletMat = doublet_inf.(x1,x2,y)
    sourceMat  = source_inf.(x1,x2,y)
    x1,x2,y    = panel_frame(foil.col, foil.edge)
    edgeInf    = doublet_inf.(x1,x2,y)
    edgeMat    = zeros(size(doubletMat))
    edgeMat[:,1] = -edgeInf[:,1]
    edgeMat[:,end] = edgeInf[:,1]
    A = doubletMat + edgeMat
    σ = [flow.Uinf,0]'*foil.normals
    A/σ
end

A = make_infs(foil)
plot(A,marker =:circle)
A[end]+A[1]