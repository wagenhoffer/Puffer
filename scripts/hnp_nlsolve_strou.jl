using NLsolve

T = Float32
strouhals = LinRange{T}(0.2, 0.4, 7)
td = LinRange{Int}(0, 10, 6)
θs = deg2rad.(td).|> T    
hs = LinRange{T}(0.0, 0.25, 6)


fna = Dict()
for θ0 in θs
    for h0 in hs
        for St in strouhals
            # Call the solver
            if θ0 == 0 && h0 == 0.0
                fna[(θ0, h0, St)] = (1.0f0, 0.0f0)
            else
                # @show θ0, h0, St
                function eq!(out, x)        
                    
                    f, a = x
                    t= 0:1/(100*f):1
                    θ = θ0*sin.(2*π*f*t .- π/2)
                    h = h0*sin.(2*π*f*t)
                    out[1] = St -  a * f 
                    out[2] = a - 2*maximum(h + sin.(θ)) 
                end
                
                sol = nlsolve(eq!, [1.0,1.0])

                # Extract the solution
                f_min, a_min = sol.zero
                fna[(θ0, h0, St)] = (f_min, a_min)
            end
        end
    end
end
fna