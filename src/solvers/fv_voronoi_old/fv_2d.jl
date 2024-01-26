# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::VoronoiMesh, equations,
                      solver, RealT, uEltype)
    coordinates, element_nodes = calc_nodes(mesh, RealT, uEltype)

    mesh.number_voronoi_elements = size(coordinates, 2)

    node_boundaries = calc_boundaries_nodes(mesh, coordinates, equations, solver)

    edges_nodes, edges_elements = calc_edges(mesh, coordinates, element_nodes)

    edge_boundaries = calc_boundaries_edges(mesh, coordinates, edges_nodes, edges_elements, equations, solver)

    nodes_edges = calc_nodes_edges(coordinates, edges_nodes)

    element_circumcenter = calc_circumcenters(coordinates, element_nodes)

    face_centers, face_sizes, face_boundary_size = calc_face_sizes(coordinates, element_nodes, edges_nodes,
                                                edges_elements, element_circumcenter, equations, solver)

    voronoi_cells_volume = calc_volume(coordinates, edges_nodes, element_nodes, element_circumcenter, face_centers, edges_elements, equations, solver)

    cache = (; coordinates, element_nodes, node_boundaries, edge_boundaries, voronoi_cells_volume, edges_nodes, edges_elements, nodes_edges,
             element_circumcenter, face_centers, face_sizes, face_boundary_size)

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

function calc_boundaries_nodes(mesh, coordinates, equations, solver)
    node_boundaries = Vector{Vector{Symbol}}(undef, size(coordinates, 2))
    for node in axes(coordinates, 2)
        x_node = get_node_coords(coordinates, equations, solver, node)
        node_boundaries[node] = mesh.boundaries(x_node)
    end

    return node_boundaries
end

function calc_edges(mesh, coordinates, element_nodes)
    n_dims = ndims(mesh)
    n_nodes = size(coordinates, 2)

    edges_nodes_ = Int[]
    edges_elements_ = Int[]
    for node1 in 1:n_nodes
        elements_node1 = findall(element_ -> node1 in view(element_nodes, :, element_), axes(element_nodes, 2))
        possible_neighbor_nodes = element_nodes[:, elements_node1]
        neighbor_nodes = setdiff(unique!(vec(possible_neighbor_nodes)), 1:node1)

        for node2 in neighbor_nodes
            # Add nodes to edge
            append!(edges_nodes_, node1, node2)

            # Decide wether edge is at boundary or inside domain
            sharing_elements = findall(element_ -> element_ in elements_node1 &&
                                                   node1 in view(element_nodes, :, element_) &&
                                                   node2 in view(element_nodes, :, element_),
                                       axes(element_nodes, 2))
            if length(sharing_elements) == 2 # Edge between nodes is in 2 elements
                # Add both adjacent elements
                append!(edges_elements_, sharing_elements...)
            else # length(sharing_elements) == 1 # Edge between nodes is in 1 element -> boundary
                # Add only adjacent element and 0 as placeholder
                append!(edges_elements_, sharing_elements..., 0)
            end
        end
    end

    edges_nodes = reshape(edges_nodes_, n_dims, div(length(edges_nodes_), n_dims))
    edges_elements = reshape(edges_elements_, 2, div(length(edges_elements_), 2))

    return edges_nodes, edges_elements
end

function calc_boundaries_edges(mesh, coordinates, edges_nodes, edges_elements, equations, solver)
    edge_boundaries = Vector{Symbol}(undef, size(edges_nodes, 2))
    for edge in axes(edges_nodes, 2)
        if edges_elements[2, edge] != 0
            edge_boundaries[edge] = :nothing
            continue
        end
        node1 = edges_nodes[1, edge]
        node2 = edges_nodes[2, edge]
        x_node1 = get_node_coords(coordinates, equations, solver, node1)
        x_node2 = get_node_coords(coordinates, equations, solver, node2)
        boundaries = mesh.boundaries(0.5 * (x_node1 + x_node2))
        # TODO
        @assert length(boundaries) == 1
        edge_boundaries[edge] = boundaries[1]
    end

    return edge_boundaries
end

function calc_nodes_edges(coordinates, edges_nodes)
    nodes_edges = Vector{Vector{Int}}(undef, size(coordinates, 2))
    for node in axes(coordinates, 2)
        nodes_edges[node] = findall(edge -> node in view(edges_nodes, :, edge), axes(edges_nodes, 2))
    end
    return nodes_edges
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
                         element_circumcenter, equations, solver)
    n_dims = size(coordinates, 1)
    face_centers = Matrix{eltype(coordinates)}(undef, n_dims, size(edges_nodes, 2))
    face_sizes = Vector{eltype(coordinates)}(undef, size(edges_nodes, 2))
    face_boundary_size = Vector{eltype(coordinates)}(undef, size(edges_nodes, 2))

    # I implemented two versions of this function.
    # One requires the precalculated circumcenters; the other doesn't

    # calculate voronoi face ends
    # circumcenter1 = zeros(eltype(coordinates), n_dims)
    # circumcenter2 = similar(circumcenter1)
    for edge in axes(edges_nodes, 2)
        element1 = edges_elements[1, edge]
        element2 = edges_elements[2, edge]
        circumcenter1 = get_node_coords(element_circumcenter, equations, solver, element1)
        # @views get_circumcenter_tri!(circumcenter1,
        #                              coordinates[:, element_nodes[1, element1]],
        #                              coordinates[:, element_nodes[2, element1]],
        #                              coordinates[:, element_nodes[3, element1]])

        if element2 > 0 # No boundary: cellcenter -- cellcenter
            circumcenter2 = get_node_coords(element_circumcenter, equations, solver, element2)
            # @views get_circumcenter_tri!(circumcenter2,
            #                              coordinates[:, element_nodes[1, element2]],
            #                              coordinates[:, element_nodes[2, element2]],
            #                              coordinates[:, element_nodes[3, element2]])

            @views face_centers[:, edge] .= NaN # Do I need this? 0.5 * (circumcenter1 + circumcenter2)
            face_sizes[edge] = norm(circumcenter1 .- circumcenter2)
            face_boundary_size[edge] = NaN
        else # boundary: cellcenter -- edgecenter
            # (edgecenter != geometric center of edge; it is the orthogonal projection to the face)
            x_node1 = get_node_coords(coordinates, equations, solver, edges_nodes[1, edge])
            x_node2 = get_node_coords(coordinates, equations, solver, edges_nodes[2, edge])
            x_node3 = get_node_coords(element_circumcenter, equations, solver, element1)

            h = x_node2 - x_node1
            h = h / norm(h)
            @views face_center = x_node1 + dot(x_node3 - x_node1, h) * h
            @views face_centers[:, edge] = face_center
            face_sizes[edge] = norm(circumcenter1 .- face_center)
            face_boundary_size[edge] = dot(x_node3 - x_node1, h)
        end
    end

    return face_centers, face_sizes, face_boundary_size
end

function calc_volume(coordinates, edges_nodes, element_nodes, element_circumcenter, face_centers, edges_elements, equations, solver)
    voronoi_cells_volume = Vector{eltype(coordinates)}(undef, size(coordinates, 2))

    for node in axes(coordinates, 2)
        x_node = get_node_coords(coordinates, equations, solver, node)
        edges = findall(edge -> node in view(edges_nodes, :, edge), axes(edges_nodes, 2))
        # TODO: some it doesn't work to just iterate over edges.
        # So I iterste over all edges and make sure that edge_ in edges
        boundary_edges = findall(edge_ -> edge_ in edges && edges_elements[2, edge_] == 0, axes(edges_nodes, 2))

        volume = zero(eltype(coordinates))
        # node is not at boundary
        if length(boundary_edges) == 0
            edge = edges[1]
            element1 = edges_elements[1, edge]
            for i in eachindex(edges)
                if element1 == edges_elements[1, edge]
                    element2 = edges_elements[2, edge]
                else element1 == edges_elements[2, edge]
                    element2 = edges_elements[1, edge]
                end
                circumcenter1 = get_node_coords(element_circumcenter, equations, solver, element1)
                circumcenter2 = get_node_coords(element_circumcenter, equations, solver, element2)

                volume += circumcenter1[1] * circumcenter2[2] - circumcenter2[1] * circumcenter1[2]

                # next edge and element
                edge = findfirst(edge_ -> edge_ in edges && element2 in view(edges_elements, :, edge_) && edge_ != edge, axes(edges_nodes, 2))
                # TODO
                # @assert length(edge) == 1
                # edge = edge[1]
                element1 = copy(element2)
            end
        else
            @assert length(boundary_edges) == 2
            inner_edges = setdiff(edges, boundary_edges)

            # first boundary edge
            edge = boundary_edges[1]
            element2 = edges_elements[1, edge]
            boundary_face_center = get_node_coords(face_centers, equations, solver, edge)
            volume += x_node[1] * boundary_face_center[2] - boundary_face_center[1] * x_node[2]

            # boundary interface
            circumcenter2 = get_node_coords(element_circumcenter, equations, solver, element2)
            volume += boundary_face_center[1] * circumcenter2[2] - circumcenter2[1] * boundary_face_center[2]

            # inner interface
            edge = findfirst(edge_ -> edge_ in edges && element2 in view(edges_elements, :, edge) && edge_ != edge, axes(edges_nodes, 2))
            element1 = copy(element2)
            for i in eachindex(inner_edges)
                if element1 == edges_elements[1, edge]
                    element2 = edges_elements[2, edge]
                else element1 == edges_elements[2, edge]
                    element2 = edges_elements[1, edge]
                end

                # TODO: use findfirst if everything is correct
                @assert length(element2) == 1
                element2 = element2[1]

                circumcenter1 = get_node_coords(element_circumcenter, equations, solver, element1)
                circumcenter2 = get_node_coords(element_circumcenter, equations, solver, element2)
                volume += circumcenter1[1] * circumcenter2[2] - circumcenter2[1] * circumcenter1[2]

                # next edge and element
                edge = findfirst(edge_ -> edge_ in edges && element2 in view(edges_elements, :, edge_) && edge_ != edge, axes(edges_nodes, 2))
                element1 = copy(element2)
            end

            # second boundary edge
            # inner interface
            boundary_face_center = get_node_coords(face_centers, equations, solver, edge)
            circumcenter1 = get_node_coords(element_circumcenter, equations, solver, element1)
            volume += circumcenter1[1] * boundary_face_center[2] - boundary_face_center[1] * circumcenter1[2]

            # boundary interface
            volume += boundary_face_center[1] * x_node[2] - x_node[1] * boundary_face_center[2]
        end
        # volume calculated with shoelace formula
        voronoi_cells_volume[node] = 0.5 * abs(volume)
    end

    return voronoi_cells_volume
end

function compute_coefficients!(u, func, t, mesh::VoronoiMesh, equations,
                               solver::FV, cache)
    for node in eachnode(mesh, cache)
        x_node = get_node_coords(cache.coordinates, equations, solver, node)
        u_node = func(x_node, t, equations)
        set_node_vars!(u, u_node, equations, solver, node)
    end
end

function allocate_coefficients(mesh::VoronoiMesh, equations, solver::FV, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(cache.coordinates),
          nvariables(equations) * nnodes(mesh, cache))
end

function rhs!(du, u, t, mesh::VoronoiMesh, equations, initial_condition,
              boundary_conditions, source_terms::Source, solver::FV,
              cache) where {Source}
    (; coordinates, element_nodes, nodes_edges, node_boundaries, edge_boundaries, voronoi_cells_volume, edges_nodes, edges_elements, element_circumcenter, face_centers, face_sizes) = cache

    du .= zero(eltype(du))

    for node in eachnode(mesh, cache)
        x_node = get_node_coords(coordinates, equations, solver, node)
        u_node = get_node_vars(u, equations, solver, node)
        edges = nodes_edges[node]

        volume = voronoi_cells_volume[node]

        @trixi_timeit timer() "inner interfaces" for edge in edges
            if edges_nodes[1, edge] == node
                neighbor_node = edges_nodes[2, edge]
            else
                neighbor_node = edges_nodes[1, edge]
            end

            x_neighbor_node = get_node_coords(coordinates, equations, solver,
                                              neighbor_node)
            u_neighbor = get_node_vars(u, equations, solver, neighbor_node)

            # TODO: Save the normal vector once
            normal = normalize(x_neighbor_node - x_node)
            @trixi_timeit timer() "surface flux" flux=solver.surface_flux(u_node,
                                                                          u_neighbor,
                                                                          normal,
                                                                          equations)
            for v in eachvariable(equations)
                du[v, node] -= (1 / volume) * face_sizes[edge] * flux[v]
            end
        end

        @trixi_timeit timer() "boundaries" for edge in edges
            element1 = edges_elements[1, edge]
            element2 = edges_elements[2, edge]
            if element2 != 0
                continue
            end
            edge_center = get_node_coords(face_centers, equations, solver, edge)

            boundary_size = norm(edge_center - x_node)

            normal = edge_center - get_node_coords(element_circumcenter, equations, solver, element1)
            normal = normal / norm(normal)

            boundary = edge_boundaries[edge]
            boundary_condition_function = boundary_conditions[boundary].boundary_value_function
            if length(node_boundaries[node]) == 1
                u_boundary = boundary_condition_function(x_node, t, equations)
            else # corner
                u_boundary = boundary_condition_function(0.5 * (x_node + edge_center), t, equations)
            end
            @trixi_timeit timer() "surface flux" flux=solver.surface_flux(u_node,
                                                                          u_boundary,
                                                                          normal,
                                                                          equations)

            for v in eachvariable(equations)
                du[v, node] -= (1 / volume) * boundary_size * flux[v]
            end
        end
    end

    return nothing
end
end # @muladd
