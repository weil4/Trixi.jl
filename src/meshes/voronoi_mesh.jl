# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct VoronoiMesh{NDIMS, RealT <: Real, Mesh, Func} <: AbstractMesh{NDIMS}
    mesh_::Mesh
    boundaries::Func
    number_voronoi_elements::Int
    # number_trees_global::Int
    # number_trees_local::Int
    # max_number_faces::Int
    # number_ghost_elements
    current_filename::String
    unsaved_changes::Bool

    function VoronoiMesh(mesh_, boundaries; current_filename = "",
                         unsaved_changes = true)
        NDIMS = ndims(mesh_)
        @assert NDIMS == 2

        mesh = new{NDIMS, Cdouble, typeof(mesh_), typeof(boundaries)}(mesh_, boundaries,
                                                                      0, # number of voronoi elements; is updated later
                                                                      current_filename,
                                                                      unsaved_changes)

        return mesh
    end
end

@inline Base.ndims(::VoronoiMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::VoronoiMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline nelementsglobal(mesh::VoronoiMesh, solver, cache) = nelements(mesh, solver, cache)
@inline function nelements(mesh::VoronoiMesh, solver, cache)
    size(cache.element_nodes, 2)
end

@inline function eachelement(mesh::VoronoiMesh, solver, cache)
    Base.OneTo(nelements(mesh, solver, cache))
end

@inline function nnodes(mesh::VoronoiMesh, cache)
    size(cache.coordinates, 2)
end

@inline function eachnode(mesh::VoronoiMesh, cache)
    Base.OneTo(nnodes(mesh, cache))
end

function Base.show(io::IO, mesh::VoronoiMesh)
    print(io, "VoronoiMesh{", ndims(mesh), ", ", real(mesh), "}(")
    print(io, "# voronoi elements: ", mesh.number_voronoi_elements)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::VoronoiMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io,
                       "VoronoiMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) *
                       "}")
        summary_line(io, "# voronoi elements", mesh.number_voronoi_elements)
        summary_footer(io)
    end
end
end # muladd
