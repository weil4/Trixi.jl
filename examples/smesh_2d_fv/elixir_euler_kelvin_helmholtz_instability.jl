using OrdinaryDiffEq
using Trixi
using Smesh

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    slope = 15
    amplitude = 0.02
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

solver = FV(surface_flux = flux_lax_friedrichs)

coordinates_min = [-1.0, -1.0]
coordinates_max = [1.0, 1.0]

initial_refinement_level = 5
n_points_x = 2^initial_refinement_level
n_points_y = 2^initial_refinement_level
data_points = mesh_basic(coordinates_min, coordinates_max, n_points_x, n_points_y)
mesh = TriangularMesh(data_points)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 3.7)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        # save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),#Euler(),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, saveat = 0.1, callback = callbacks);
summary_callback() # print the timer summary

# using Plots; pyplot()
# anim = @animate for i in eachindex(sol.u)
#     u_i = Trixi.wrap_array(sol.u[i], semi)
#     surface(semi.cache.midpoints[1, :], semi.cache.midpoints[2, :], u_i[1, :],
#                     #=zaxis=[1.8, 2.2],=# xlabel="x", ylabel="y", title="t=$(sol.t[i])")
# end
# gif(anim, "anim_fps15.gif", fps = 15)
# plt = display(surface(semi.cache.midpoints[1, :], semi.cache.midpoints[2, :], Trixi.wrap_array(sol.u[1], semi)[1, :]))
# plt = display(surface(semi.cache.midpoints[1, :], semi.cache.midpoints[2, :], Trixi.wrap_array(sol.u[end], semi)[1, :]))
# scatter(semi.cache.data_points[1, :], semi.cache.data_points[2, :], markercolor=:red, label="points")
# scatter!(semi.cache.midpoints[1, :], semi.cache.midpoints[2, :], markercolor=:blue, label="midpoints")
