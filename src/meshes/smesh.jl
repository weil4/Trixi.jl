# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct TriangularMesh{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}
    data_points::Array{RealT, 2}
    n_elements::Int
    # n_max_faces::Int
    # is_parallel::IsParallel
    current_filename::String
    unsaved_changes::Bool

    function TriangularMesh(data_points; current_filename = "", unsaved_changes = true)
        NDIMS = size(data_points, 1)
        @assert NDIMS == 2

        n_elements = 0 # Changed later

        # is_parallel = False()

        mesh = new{NDIMS, Cdouble}(data_points, n_elements, current_filename,
                                   unsaved_changes)

        return mesh
    end
end

@inline Base.ndims(::TriangularMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::TriangularMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline nelementsglobal(mesh::TriangularMesh, solver, cache) = nelements(mesh, solver,
                                                                         cache)
@inline nelements(mesh::TriangularMesh, solver, cache) = mesh.n_elements

@inline function eachelement(mesh::TriangularMesh, solver, cache)
    Base.OneTo(nelements(mesh, solver, cache))
end

function Base.show(io::IO, mesh::TriangularMesh)
    print(io, "TriangularMesh{", ndims(mesh), ", ", real(mesh), "}(")
    print(io, "# elements: ", mesh.n_elements)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::TriangularMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io,
                       "TriangularMesh{" * string(ndims(mesh)) * ", " *
                       string(real(mesh)) * "}")
        summary_line(io, "# elements", mesh.n_elements)
        summary_footer(io)
    end
end

mutable struct PolygonMesh{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}
    mesh_type::Symbol
    orthogonal_boundary_edges::Bool
    data_points::Array{RealT, 2}
    n_elements::Int
    # n_max_faces::Int
    # is_parallel::IsParallel
    current_filename::String
    unsaved_changes::Bool

    function PolygonMesh(data_points; mesh_type = :standard_voronoi,
                         orthogonal_boundary_edges = true,
                         current_filename = "", unsaved_changes = true)
        NDIMS = size(data_points, 1)
        @assert NDIMS == 2

        n_elements = 0 # Changed later

        # is_parallel = False()

        mesh = new{NDIMS, Cdouble}(mesh_type, orthogonal_boundary_edges, data_points,
                                   n_elements,
                                   current_filename, unsaved_changes)

        return mesh
    end
end

@inline Base.ndims(::PolygonMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::PolygonMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline nelementsglobal(mesh::PolygonMesh, solver, cache) = nelements(mesh, solver,
                                                                      cache)
@inline nelements(mesh::PolygonMesh, solver, cache) = mesh.n_elements

@inline function eachelement(mesh::PolygonMesh, solver, cache)
    Base.OneTo(nelements(mesh, solver, cache))
end

# @inline function nelementsglobal(mesh::PolygonMesh, solver, cache)
#     nelements(mesh, solver, cache)
# end

# @inline function nnodes(mesh::PolygonMesh, cache)
#     mesh.n_nodes
# end

function Base.show(io::IO, mesh::PolygonMesh)
    print(io, "PolygonMesh{", ndims(mesh), ", ", real(mesh), "}(")
    print(io, "# elements: ", mesh.n_elements)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::PolygonMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io,
                       "PolygonMesh{" * string(ndims(mesh)) * ", " *
                       string(real(mesh)) * "}")
        summary_line(io, "# elements", mesh.n_elements)
        summary_footer(io)
    end
end
end # muladd
