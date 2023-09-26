
using OrdinaryDiffEq
using Trixi


###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
equations = IdealGlmMhdEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
volume_flux  = (flux_hindenlang_gassner, flux_nonconservative_powell)
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                # positivity_variables_cons=[1],
                                local_minmax_variables_cons=[1],
                                positivity_variables_nonlinear=(pressure,),
                                spec_entropy=true,
                                positivity_correction_factor=0.5,
                                bar_states=true)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg=volume_flux,
                                                volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Get the curved quad mesh from a mapping function
# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi, eta)
    y = 2.0 * eta + 1.0 / 6.0 * (cos(1.5 * pi * xi) * cos(0.5 * pi * eta))

    x = 2.0 * xi + 1.0 / 6.0 * (cos(0.5 * pi * xi) * cos(2 * pi * y))

    return SVector(x, y)
  end

cells_per_dimension = (16, 16)

mesh = StructuredMesh(cells_per_dimension, mapping, periodicity=true)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.9)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

cfl = 0.8
stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback(save_errors=false))

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks=stage_callbacks);
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

