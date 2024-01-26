# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct FV{SlopeLimiter, SurfaceFlux}
    order::Integer
    slope_limiter::SlopeLimiter
    surface_flux::SurfaceFlux

    function FV(; order = 1, slope_limiter = average_slope_limiter,
                surface_flux = flux_central)
        new{typeof(slope_limiter), typeof(surface_flux)}(order, slope_limiter,
                                                         surface_flux)
    end
end

function Base.show(io::IO, solver::FV)
    @nospecialize solver # reduce precompilation time

    print(io, "FV(")
    print(io, "order $(solver.order)")
    if solver.order > 1
        print(io, ", ", solver.slope_limiter)
    end
    print(io, ", ", solver.surface_flux)
    print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", solver::FV)
    @nospecialize solver # reduce precompilation time

    if get(io, :compact, false)
        show(io, solver)
    else
        summary_header(io, "FV{" * string(real(solver)) * "}")
        summary_line(io, "order", solver.order)
        if solver.order > 1
            summary_line(io, "slope limiter", solver.slope_limiter)
        end
        summary_line(io, "surface flux", solver.surface_flux)
        summary_footer(io)
    end
end

Base.summary(io::IO, solver::FV) = print(io, "FV(order=$(solver.order))")

@inline Base.real(solver::FV) = Float64 # TODO
@inline ndofs(mesh, solver::FV, cache) = nelementsglobal(mesh, solver, cache)

@inline function ndofsglobal(mesh, solver::FV, cache)
    ndofs(mesh, solver, cache)
end

@inline function get_node_vars(u, equations, solver::FV, element)
    SVector(ntuple(@inline(v->u[v, element]), Val(nvariables(equations))))
end

@inline function set_node_vars!(u, u_node, equations, solver::FV, element)
    for v in eachvariable(equations)
        u[v, element] = u_node[v]
    end
    return nothing
end

@inline function get_node_coords(x, equations, solver::FV, indices...)
    SVector(ntuple(@inline(idx->x[idx, indices...]), Val(ndims(equations))))
end

@inline function get_surface_node_vars(u, equations, solver::FV, indices...)
    # There is a cut-off at `n == 10` inside of the method
    # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
    # in Julia `v1.5`, leading to type instabilities if
    # more than ten variables are used. That's why we use
    # `Val(...)` below.
    u_ll = SVector(ntuple(@inline(v->u[1, v, indices...]), Val(nvariables(equations))))
    u_rr = SVector(ntuple(@inline(v->u[2, v, indices...]), Val(nvariables(equations))))
    return u_ll, u_rr
end

# General fallback
@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                            solver::FV, cache)
    wrap_array_native(u_ode, mesh, equations, solver, cache)
end

# Like `wrap_array`, but guarantees to return a plain `Array`, which can be better
# for interfacing with external C libraries (MPI, HDF5, visualization),
# writing solution files etc.
@inline function wrap_array_native(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                                   solver::FV, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                nvariables(equations) * nelements(mesh, solver, cache)
    end
    unsafe_wrap(Array{eltype(u_ode), 2}, pointer(u_ode),
                (nvariables(equations), nelements(mesh, solver, cache)))
end

@inline function wrap_array_native(u_ode::AbstractVector, mesh::VoronoiMesh, equations,
                                   solver::FV, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                nvariables(equations) * nnodes(mesh, cache)
    end
    unsafe_wrap(Array{eltype(u_ode), 2}, pointer(u_ode),
                (nvariables(equations), nnodes(mesh, cache)))
end

function average_slope_limiter(s1, s2)
    return 0.5 * s1 + 0.5 * s2
end

function minmod(s...)
    if all(s .> 0)
        return minimum(s)
    elseif all(s .< 0)
        return maximum(s)
    end
    return zero(eltype(s[1]))
end

function monotonized_central(s1, s2)
    return minmod(2 * s1, (s1 + s2) / 2, 2 * s2)
end

function SolutionAnalyzer(solver::FV; kwargs...)
end

function create_cache_analysis(analyzer, mesh,
                               equations, solver::FV, cache,
                               RealT, uEltype)
end

include("fv_t8code/fv_2d.jl")
include("fv_voronoi_old/fv_2d.jl")
end # @muladd
