
function set_ledge!(foil::Foil, flow::FlowParams)
    front = foil.N ÷ 2
    le_norm = sum(foil.normals[:, front:(front + 1)], dims = 2) / 2.0 * flow.Uinf * flow.Δt
    #initial the ledge
    front += 1
    foil.ledge = [foil.foil[:,front] foil.foil[:,front] .+ le_norm*0.4 foil.foil[:,front] .+ le_norm*1.4 ]
    # if flow.n == 1
    #     foil.ledge = [foil.foil[:, front] foil.foil[:, front] .+ le_norm * 0.4 foil.foil[:,
    #         front] .+
    #                                                                            le_norm * 1.4]
    # else
    #     foil.ledge = [foil.foil[:, front] foil.foil[:, front] .+ le_norm * 0.4 foil.ledge[:,
    #         2]]
    # end
end

function ledge_inf(foil::Foil)
    x1, x2, y = panel_frame(foil.col, foil.ledge)
    edgeInf = doublet_inf.(x1, x2, y)
    edgeMat = zeros(foil.N, foil.N)
    #TODO: make this the correct spot for sure
    mid = foil.N ÷ 2
    edgeMat[:, mid] = edgeInf[:, 1]
    edgeMat[:, mid + 1] = -edgeInf[:, 1]
    edgeMat, edgeInf[:, 2]
end

function get_μ!(foil::Foil, rhs, A, le_inf, buff, lesp)

    #now pseudo code, but if lesp, then add le_inf to A, else the simulation was fine
    if !lesp
        foil.μs = A \ (-rhs * foil.σs - buff)[:]
    else
        foil.μs = (A + le_inf) \ (-rhs * foil.σs - buff)[:]
        set_ledge_strength!(foil)
    end
    set_edge_strength!(foil)

    nothing
end
