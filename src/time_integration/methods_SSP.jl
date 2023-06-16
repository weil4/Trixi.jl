# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Abstract base type for time integration schemes of explicit strong stability-preserving (SSP)
# Runge-Kutta (RK) methods. They are high-order time discretizations that guarantee the TVD property.
abstract type SimpleAlgorithmSSP end

"""
    SimpleSSPRK33(; stage_callbacks=(APosterioriLimiter(), BoundsCheckCallback()))

The third-order SSP Runge-Kutta method of Shu and Osher.

## References

- Shu, Osher (1988)
  "Efficient Implementation of Essentially Non-oscillatory Shock-Capturing Schemes" (Eq. 2.18)
  [DOI: 10.1016/0021-9991(88)90177-5](https://doi.org/10.1016/0021-9991(88)90177-5)

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SimpleSSPRK33{StageCallbacks} <: SimpleAlgorithmSSP
    a::SVector{3, Float64}
    b::SVector{3, Float64}
    c::SVector{3, Float64}
    stage_callbacks::StageCallbacks

    function SimpleSSPRK33(;
                           stage_callbacks = (APosterioriLimiter(),
                                              BoundsCheckCallback()))
        a = SVector(0.0, 3 / 4, 1 / 3)
        b = SVector(1.0, 1 / 4, 2 / 3)
        c = SVector(0.0, 1.0, 1 / 2)

        # Butcher tableau
        #   c |       a
        #   0 |
        #   1 |   1
        # 1/2 | 1/4  1/4
        # --------------------
        #   b | 1/6  1/6  2/3

        new{typeof(stage_callbacks)}(a, b, c, stage_callbacks)
    end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SimpleIntegratorSSPOptions{Callback}
    callback::Callback # callbacks; used in Trixi
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function SimpleIntegratorSSPOptions(callback, tspan; maxiters = typemax(Int), kwargs...)
    SimpleIntegratorSSPOptions{typeof(callback)}(callback, false, Inf, maxiters,
                                                 [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct SimpleIntegratorSSP{RealT <: Real, uType, Params, Sol, F, Alg,
                                   SimpleIntegratorSSPOptions}
    u::uType
    du::uType
    r0::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg
    opts::SimpleIntegratorSSPOptions
    finalstep::Bool # added for convenience
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::SimpleIntegratorSSP, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

"""
    solve(ode, alg; dt, callbacks, kwargs...)

The following structures and methods provide the infrastructure for SSP Runge-Kutta methods
of type `SimpleAlgorithmSSP`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function solve(ode::ODEProblem, alg = SimpleSSPRK33()::SimpleAlgorithmSSP;
               dt, callback = nothing, kwargs...)
    u = copy(ode.u0)
    du = similar(u)
    r0 = similar(u)
    t = first(ode.tspan)
    iter = 0
    integrator = SimpleIntegratorSSP(u, du, r0, t, dt, zero(dt), iter, ode.p,
                                     (prob = ode,), ode.f, alg,
                                     SimpleIntegratorSSPOptions(callback, ode.tspan;
                                                                kwargs...), false)

    # resize container
    resize!(integrator.p, nelements(integrator.p.solver, integrator.p.cache))

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            error("unsupported")
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    elseif !isnothing(callback)
        error("unsupported")
    end

    for stage_callback in alg.stage_callbacks
        init_callback(stage_callback, integrator.p)
    end

    solve!(integrator)
end

function solve!(integrator::SimpleIntegratorSSP)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    # WARNING: Only works if the last callback got a variable `output_directory`.
    if callbacks.discrete_callbacks[end].condition isa SaveSolutionCallback
        output_directory = callbacks.discrete_callbacks[end].condition.output_directory
    else
        output_directory = "out"
    end

    integrator.finalstep = false
    while !integrator.finalstep
        if isnan(integrator.dt)
            error("time step size `dt` is NaN")
        end

        # if the next iteration would push the simulation beyond the end time, set dt accordingly
        if integrator.t + integrator.dt > t_end ||
           isapprox(integrator.t + integrator.dt, t_end)
            integrator.dt = t_end - integrator.t
            terminate!(integrator)
        end

        @. integrator.r0 = integrator.u
        for stage in eachindex(alg.c)
            t_stage = integrator.t + integrator.dt * alg.c[stage]
            # compute du
            integrator.f(integrator.du, integrator.u, integrator.p, t_stage)

            # perform forward Euler step
            @. integrator.u = integrator.u + integrator.dt * integrator.du

            for stage_callback in alg.stage_callbacks
                stage_callback(integrator.u, integrator, stage)
            end

            # perform convex combination
            @. integrator.u = alg.a[stage] * integrator.r0 + alg.b[stage] * integrator.u
        end

        integrator.iter += 1
        integrator.t += integrator.dt

        # handle callbacks
        if callbacks isa CallbackSet
            for cb in callbacks.discrete_callbacks
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
            end
        end

        # respect maximum number of iterations
        if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
            @warn "Interrupted. Larger maxiters is needed."
            terminate!(integrator)
        end
    end

    for stage_callback in alg.stage_callbacks
        finalize_callback(stage_callback, integrator.p)
    end

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u), prob)
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleIntegratorSSP) = integrator.du
get_tmp_cache(integrator::SimpleIntegratorSSP) = (integrator.r0,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleIntegratorSSP, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleIntegratorSSP, dt)
    integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::SimpleIntegratorSSP)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SimpleIntegratorSSP, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.r0, new_size)

    # Resize container
    resize!(integrator.p, new_size)
end

function Base.resize!(semi::AbstractSemidiscretization, new_size)
    resize!(semi, semi.solver.volume_integral, new_size)
end

Base.resize!(semi, volume_integral::AbstractVolumeIntegral, new_size) = nothing

function Base.resize!(semi, volume_integral::VolumeIntegralSubcellLimiting, new_size)
    # Resize container_antidiffusive_flux
    resize!(semi.cache.container_antidiffusive_flux, new_size)

    # Resize container_shock_capturing
    resize!(volume_integral.indicator.cache.container_shock_capturing, new_size)
    # Calc subcell normal directions before StepsizeCallback
    @unpack indicator = volume_integral
    if indicator isa IndicatorMCL ||
       (indicator isa IndicatorIDP && indicator.bar_states)
        resize!(indicator.cache.container_bar_states, new_size)
        calc_normal_directions!(indicator.cache.container_bar_states,
                                mesh_equations_solver_cache(semi)...)
    end
end

function calc_normal_directions!(container_bar_states, mesh::TreeMesh, equations, dg,
                                 cache)
    nothing
end

function calc_normal_directions!(container_bar_states, mesh::StructuredMesh, equations,
                                 dg, cache)
    @unpack weights, derivative_matrix = dg.basis
    @unpack contravariant_vectors = cache.elements

    @unpack normal_direction_xi, normal_direction_eta = container_bar_states
    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg)
            normal_direction = get_contravariant_vector(1, contravariant_vectors, 1, j,
                                                        element)
            for i in 2:nnodes(dg)
                for m in 1:nnodes(dg)
                    normal_direction += weights[i - 1] * derivative_matrix[i - 1, m] *
                                        get_contravariant_vector(1,
                                                                 contravariant_vectors,
                                                                 m, j, element)
                end
                for v in 1:(nvariables(equations) - 2)
                    normal_direction_xi[v, i - 1, j, element] = normal_direction[v]
                end
            end
        end
        for i in eachnode(dg)
            normal_direction = get_contravariant_vector(2, contravariant_vectors, i, 1,
                                                        element)
            for j in 2:nnodes(dg)
                for m in 1:nnodes(dg)
                    normal_direction += weights[j - 1] * derivative_matrix[j - 1, m] *
                                        get_contravariant_vector(2,
                                                                 contravariant_vectors,
                                                                 i, m, element)
                end
                for v in 1:(nvariables(equations) - 2)
                    normal_direction_eta[v, i, j - 1, element] = normal_direction[v]
                end
            end
        end
    end

    return nothing
end
end # @muladd
