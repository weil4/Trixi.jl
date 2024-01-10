# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct VoronoiMesh{NDIMS, RealT <: Real, Mesh, Func} <: AbstractMesh{NDIMS}
    mesh_::Mesh
    boundaries::Func
    # number_trees_global::Int
    # number_trees_local::Int
    # max_number_faces::Int
    # number_elements::Int
    # number_ghost_elements
    # tree shapes
    current_filename::String
    unsaved_changes::Bool

    function VoronoiMesh(mesh_, boundaries; current_filename = "",
                         unsaved_changes = true)
        NDIMS = ndims(mesh_)
        @assert NDIMS == 2

        mesh = new{NDIMS, Cdouble, typeof(mesh_), typeof(boundaries)}(mesh_, boundaries,
                                                                      current_filename,
                                                                      unsaved_changes)

        return mesh
    end
end

@inline Base.ndims(::VoronoiMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::VoronoiMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline function nelements(mesh::VoronoiMesh, solver, cache)
    size(cache.element_nodes, 2)
end

@inline function eachelement(mesh::VoronoiMesh, solver, cache)
    Base.OneTo(nelements(mesh, solver, cache))
end

@inline function nelementsglobal(mesh::VoronoiMesh, solver, cache)
    nelements(mesh, solver, cache)
end

@inline function nnodes(mesh::VoronoiMesh, cache)
    size(cache.coordinates, 2)
end

@inline function eachnode(mesh::VoronoiMesh, cache)
    Base.OneTo(nnodes(mesh, cache))
end

function Base.show(io::IO, mesh::VoronoiMesh)
    print(io, "VoronoiMesh{", ndims(mesh), ", ", real(mesh), "}(")
    # print(io, "#trees: ", mesh.number_trees_global)
    # print(io, ", #voronoi nodes: ", nnodes(mesh))
    print(")")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::VoronoiMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io,
                       "VoronoiMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) *
                       "}")
        # summary_line(io, "#trees", mesh.number_trees_global)
        # summary_line(io, "#elements", nelementsglobal(mesh))
        summary_footer(io)
    end
end

function create_cache(mesh::VoronoiMesh, equations,
                      solver, RealT, uEltype)
    coordinates, element_nodes = calc_nodes(mesh, RealT, uEltype)

    edges_nodes, edges_elements = calc_edges(mesh, coordinates, element_nodes)

    element_circumcenter = calc_circumcenters(coordinates, element_nodes)

    face_centers, face_sizes = calc_face_sizes(coordinates, element_nodes, edges_nodes,
                                               edges_elements, element_circumcenter)

    cache = (; coordinates, element_nodes, edges_nodes, edges_elements,
             element_circumcenter, face_centers, face_sizes)

    return cache
end

function calc_nodes(mesh, RealT, uEltype)
    (; forest) = mesh.mesh_
    # Check that the forest is a committed.
    @assert(t8_forest_is_committed(forest)==1)
    n_dims = ndims(mesh)
    (; max_number_faces, number_elements) = mesh.mesh_
    @assert max_number_faces == 3

    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)
    @assert num_ghost_elements == 0

    coordinates_ = RealT[]
    element_nodes = zeros(Int, max_number_faces, number_elements)

    # Loop over all local trees in the forest.
    element_index = 0
    node_index = 0

    node = zeros(RealT, n_dims)
    for itree in 0:(mesh.mesh_.number_trees_local - 1)
        tree_class = t8_forest_get_tree_class(forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

        # Get the number of elements of this tree.
        num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

        # Loop over all local elements in the tree.
        for ielement in 0:(num_elements_in_tree - 1)
            element_index += 1 # Note: Julia has 1-based indexing, while C/C++ starts with 0.

            element = t8_forest_get_element_in_tree(forest, itree, ielement)

            # Loop over all faces of an element.
            num_faces = t8_element_num_faces(eclass_scheme, element)

            for iface in 1:num_faces
                t8_forest_element_coordinate(forest, itree, element, iface - 1,
                                             @views(node))
                # Check if node already in coordinates
                existing_index = 0
                for i in 1:node_index
                    if node[1] == coordinates_[2 * i - 1] &&
                       node[2] == coordinates_[2 * i]
                        existing_index = i
                    end
                end
                if existing_index == 0 # Node doesn't exists
                    node_index += 1
                    append!(coordinates_, node...)
                    element_nodes[iface, element_index] = node_index
                else # Node already exists
                    element_nodes[iface, element_index] = existing_index
                end
            end
        end
    end

    coordinates = reshape(coordinates_, n_dims, div(length(coordinates_), n_dims))

    return coordinates, element_nodes
end

function calc_edges(mesh, coordinates, element_nodes)
    n_dims = ndims(mesh)
    n_nodes = size(coordinates, 2)

    edges_nodes_ = Int[]
    edges_elements_ = Int[]
    for node1 in 1:n_nodes
        for node2 in (node1 + 1):n_nodes
            sharing_elements = findall(col -> (node1 in element_nodes[:, col] &&
                                               node2 in element_nodes[:, col]),
                                       axes(element_nodes, 2))
            if length(sharing_elements) == 2 # Edge between nodes is in 2 elements
                # Add nodes to edge
                append!(edges_nodes_, node1, node2)
                # Add elements to edge
                append!(edges_elements_, sharing_elements...)
            elseif length(sharing_elements) == 1 # Edge between nodes is in 1 element -> boundary
                # Add nodes to edge
                append!(edges_nodes_, node1, node2)
                # Add elements to edge
                append!(edges_elements_, sharing_elements..., 0)
            end # Else: No edge between nodes
        end
    end

    edges_nodes = reshape(edges_nodes_, n_dims, div(length(edges_nodes_), n_dims))
    edges_elements = reshape(edges_elements_, 2, div(length(edges_elements_), 2))

    return edges_nodes, edges_elements
end

function calc_circumcenters(coordinates, element_nodes)
    circumcenters = Matrix{eltype(coordinates)}(undef, size(coordinates, 1),
                                                size(element_nodes, 2))

    for element in axes(element_nodes, 2)
        @views get_circumcenter_tri!(circumcenters[:, element],
                                     coordinates[:, element_nodes[1, element]],
                                     coordinates[:, element_nodes[2, element]],
                                     coordinates[:, element_nodes[3, element]])
    end

    return circumcenters
end

function get_circumcenter_tri!(circumcenter, a, b, c)
    # Use coordinates relative to point 'a' of the triangle.
    xba = b[1] - a[1]
    yba = b[2] - a[2]
    xca = c[1] - a[1]
    yca = c[2] - a[2]

    # Squares of lengths of the edges incident to 'a'.
    balength = xba * xba + yba * yba
    calength = xca * xca + yca * yca

    # Take your chances with floating-point roundoff
    denominator = 0.5 / (xba * yca - yba * xca)

    # Calculate offset (from 'a') of circumcenter.
    xcirca = (yca * balength - yba * calength) * denominator
    ycirca = (xba * calength - xca * balength) * denominator

    circumcenter[1] = xcirca + a[1]
    circumcenter[2] = ycirca + a[2]

    return nothing
end

function calc_face_sizes(coordinates, element_nodes, edges_nodes, edges_elements,
                         element_circumcenter)
    n_dims = size(coordinates, 1)
    face_centers = Matrix{eltype(coordinates)}(undef, n_dims, size(edges_nodes, 2))
    face_sizes = Vector{eltype(coordinates)}(undef, size(edges_nodes, 2))

    # I implemented two versions of this function.
    # One requires the precalculated circumcenters; the other doesn't

    # calculate voronoi face ends
    # circumcenter1 = zeros(eltype(coordinates), n_dims)
    # circumcenter2 = similar(circumcenter1)
    for edge in axes(edges_nodes, 2)
        circumcenter1 = view(element_circumcenter, :, edges_elements[1, edge])
        # @views get_circumcenter_tri!(circumcenter1,
        #                              coordinates[:, element_nodes[1, edges_elements[1, edge]]],
        #                              coordinates[:, element_nodes[2, edges_elements[1, edge]]],
        #                              coordinates[:, element_nodes[3, edges_elements[1, edge]]])

        element2 = edges_elements[2, edge]
        if element2 > 0 # No boundary: cellcenter -- cellcenter
            circumcenter2 = view(element_circumcenter, :, element2)
            # @views get_circumcenter_tri!(circumcenter2,
            #                              coordinates[:, element_nodes[1, edges_elements[2, edge]]],
            #                              coordinates[:, element_nodes[2, edges_elements[2, edge]]],
            #                              coordinates[:, element_nodes[3, edges_elements[2, edge]]])

            @views face_centers[:, edge] .= 0.5 * (circumcenter1 + circumcenter2)
            face_sizes[edge] = norm(circumcenter1 .- circumcenter2)
        else # boundary: cellcenter -- edgecenter
            node1 = edges_nodes[1, edge]
            node2 = edges_nodes[2, edge]
            @views edge_center = 0.5 * (coordinates[:, node1] + coordinates[:, node2])
            @views face_centers[:, edge] .= 0.5 * (circumcenter1 + edge_center)
            face_sizes[edge] = norm(circumcenter1 .- edge_center)
        end
    end

    return face_centers, face_sizes
end
end # muladd
