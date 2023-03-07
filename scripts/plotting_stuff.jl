begin 
    n = 25
    T = LinRange(0, 1, n)
    
    anim = @animate for t in T        
        plot(foil.foil[1,:], 
        foil.foil[2,:] .+ foil.kine.(foil._foil[1,:],foil.k,foil.f,t),label="",aspect_ratio=:equal)        
    end
    gif(anim,"simple.gif", fps=12)
end
#FIND body velocities to be plugged into sigma 
# plot(foil.foil[1,:], foil.foil[2,:] .+ foil.kine.(foil.foil[1,:],foil.k,foil.f,t),label="",aspect_ratio=:equal)        
begin 
    old_cols = zeros(5,size(foil.col)...)
    foil, flow = init_params()
    for i=1:5       
        (foil)(flow)
        old_cols = circshift(old_cols,(1,0,0))
        old_cols[1,:,:] = foil.col
        # plot(foil.foil[1,:], foil.foil[2,:],
        #     label="", 
        #     aspect_ratio=:equal,
        #     ylims=(-0.35,0.35),
        #     xlims=(foil.foil[1,foil.N÷2] - 10*flow.Δt,flow.N*flow.Δt*flow.Uinf))
    end
    gif(anim,"simple.gif", fps=50)
end


begin
    vels = @animate for i = 1:flow.N
        plot(foil._foil[1,:],ForwardDiff.derivative.(kine, flow.Δt*i),color=:red,label="vel")
        plot!(foil.foil[1,:],foil.kine.(foil.foil[1,:],foil.k,foil.f,i*flow.Δt),
              label="pos",aspect_ratio=:equal,color=:green) 
        plot!(xlims=(-0.5,1.5), ylims=(-0.6, 0.6))
    end
    gif(vels,"vels.gif", fps=24)
end

k(t) = foil.kine.(foil._foil[1,:],foil.k,foil.f,t)
c(x) = foil.kine.(x,foil.k,foil.f,1.0)

ForwardDiff.derivative.(sin, 1:0.1:2)
ForwardDiff.derivative.(k, 1)
ForwardDiff.derivative(d, 1.0)
isa(d(1.0), Union{Real,AbstractArray})
d(1im)
