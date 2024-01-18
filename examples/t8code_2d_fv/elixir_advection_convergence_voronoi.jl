#!/usr/bin/env julia
# Has to be at least Julia v1.9.0.

# Running in parallel:
#   ${JULIA_DEPOT_PATH}/.julia/bin/mpiexecjl --project=. -n 3 julia hybrid-t8code-mesh.jl
#
# More information: https://juliaparallel.org/MPI.jl/stable/usage/
using OrdinaryDiffEq
using Trixi

####################################################

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:x_neg => boundary_condition,
                           :x_pos => boundary_condition,
                           :y_neg => boundary_condition,
                           :y_pos => boundary_condition)

solver = FV(order = 1,
            surface_flux = flux_lax_friedrichs)

function boundaries(x)
    boundaries_ = Symbol[]
    if x[1] == -1.0
        push!(boundaries_, :x_neg)
    end
    if x[1] == 1.0
        push!(boundaries_, :x_pos)
    end
    if x[2] == -1.0
        push!(boundaries_, :y_neg)
    end
    if x[2] == 1.0
        push!(boundaries_, :y_pos)
    end
    return boundaries_
end

initial_refinement_level = 3
mesh_ = T8codeFVMesh{2}(Trixi.cmesh_new_periodic_tri, initial_refinement_level)

mesh = VoronoiMesh(mesh_, boundaries)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

ode = semidiscretize(semi, (0.0, 10.0));

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback)#, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, Euler(),# CarpenterKennedy2N54(williamson_condition=false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, saveat = 0.1, callback = callbacks)
summary_callback()

# using Plots; pyplot()
# @gif for i in eachindex(sol.u)
#     surface(semi.cache.coordinates[1, :], semi.cache.coordinates[2, :], sol.u[i],
#                     #=zaxis=[1.8, 2.2],=# xlabel="x", ylabel="y")
# end
# plt = display(surface(semi.cache.coordinates[1, :], semi.cache.coordinates[2, :], sol.u[end]))
# scatter(semi.cache.coordinates[1, :], semi.cache.coordinates[2, :])
