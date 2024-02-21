using OrdinaryDiffEq
using Trixi
using Smesh

####################################################

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2 = r > 0.5 ? 0.0 : 0.1882 * sin_phi
    p = r > 0.5 ? 1.0E-3 : 1.245

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_blast_wave

# Note: Only supported to use one boundary condition for all boundaries.
# To fix this: How do I distinguish which boundary I am at? TODO
boundary_condition = BoundaryConditionDirichlet(initial_condition)

solver = FV(surface_flux = flux_lax_friedrichs)

coordinates_min = [-1.0, -1.0]
coordinates_max = [1.0, 1.0]

initial_refinement_level = 5
n_points_x = 2^initial_refinement_level
n_points_y = 2^initial_refinement_level
data_points = mesh_basic(coordinates_min, coordinates_max, n_points_x, n_points_y)
mesh = TriangularMesh(data_points)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

ode = semidiscretize(semi, (0.0, 2.0));

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 20,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.05)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),#Euler(),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, saveat = 0.1, callback = callbacks);
summary_callback()
