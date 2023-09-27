I have published a paper that covers a lot of the material implemented here [1] [2]. 




The first paper [1] specifically talks about the traveling wave form, anguilliform is already implemented (the other is a few minutes to set up), and how it produces thrust.
The studies use three big non-dimensional numbers that we care about variation of to make a surrogate of a "physical" swimmer. 
The Strouhal number  $St = \frac{fa_0}{U_|inf}$ , reduced frequency $f* = \frac{fc}{U_\inf}$ and ND wavenumber $k* = kc$. 
Here $f$ is frequency, $a_0$ is amplitude of motion at the trailing edge, $$ is chord or length of the swimmer, $k$ is the wavenumber of the wave traveling down the chord of the swimmer (think $\lambda$ if that helps) and $U_\inf$ is the freestream velocity (to be replaced with the self-propelled velocity in an upcoming pull). 

In the paper, the ranges of $0.25 \le f* \le 4$ and $0.025 \le St \le 0.4$ and $0.35 \le k* \le 2$. You only need to adjust the values of 
frequency and freestream to obtain the first two ranges and the third if an independent variable to adjust. The chord can be left as a constant of 1 meter (it plots nice) and the $a_0$ value is a physical measurement approximation [3]. 

Theese are the input parameters to start and the refinement of how many simulations needed to construct an accurate approximation/surrogate is not known

I have placed a copy of the time_increment! function here with comments to explain what is a function of the inputs
and what is a solution space. 

```
function time_increment!(flow::FlowParams, foil::Foil, wake::Wake)
    if flow.n != 1
        move_wake!(wake, flow)   
        release_vortex!(wake, foil)
    end    
    (foil)(flow) #does the kinematics of the foil and sets up the next time step to be solved 
                 # this includes new panel positions, and the normals and tangents of the foil (input space)
    
    A, rhs, edge_body = make_infs(foil)
    #A <- large dense influence matrix (input space)
    #rhs <- velocity from the body onto itself and from the vortex wake onto the panels (input space)
    #edgebody <- edge panel influence onto the panels (input space)
    setσ!(foil, flow)   
    # velocity influences from motion of foil and freestream (input)
    foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow) 
    # wake induced velocity (input)
    normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)' #linear alg
    foil.σs -= normal_wake_ind[:] #linear alg
    buff = edge_body * foil.μ_edge[1] #linear alg
    foil.μs = A \ (-rhs*foil.σs-buff)[:]
    # above we solve the system, this effectively tells us what vortices can be used to represent the foil
    # the μs are the first (and possibly most important) of the output space variables
    #the next two lines are taking the dynamic info we just obtained to make a static snapshot of the flow 
    set_edge_strength!(foil)
    cancel_buffer_Γ!(wake, foil)
    #the next two funcs find what velocity is induced on the vortex particles in the wake (to define how they move)
    body_to_wake!(wake, foil, flow)
    wake_self_vel!(wake, flow)    
    nothing
end
```


[1] Wagenhoffer, Nathan, Keith W. Moored, and Justin W. Jaworski. "Unsteady propulsion and the acoustic signature of undulatory swimmers in and out of ground effect." Physical Review Fluids 6.3 (2021): 033101. https://doi.org/10.1103/PhysRevFluids.6.033101

[2] Wagenhoffer, Nathan, Keith W. Moored, and Justin W. Jaworski. "On the noise generation and unsteady performance of combined heaving and pitching foils." Bioinspiration & Biomimetics 18.4 (2023): 046011. https://doi.org/10.1088/1748-3190/acd59d

[3] AP Maertens, Michael S Triantafyllou, Dick KP Yu. "Efficiency of fish propulsion" Bioinspiration & Biomimetics, 10.4 (2015) https://arxiv.org/pdf/1409.7263