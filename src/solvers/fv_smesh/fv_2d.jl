# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::TriangularMesh, equations,
                      solver::FV, RealT, uEltype)
    (; data_points) = mesh

    # Triangulation
    (; triangulation_vertices) = mesh

    # Calculate neighbors in triangulation
    triangulation_neighbors = delaunay_compute_neighbors(data_points,
                                                         triangulation_vertices)

    volume, dx = calc_volume(data_points, triangulation_vertices, mesh, equations,
                             solver)

    midpoints = calc_midpoints(data_points, triangulation_vertices, mesh, equations,
                               solver)

    cache = (; triangulation_vertices, triangulation_neighbors, volume, dx, midpoints)

    return cache
end

function calc_volume(data_points, triangulation_vertices, mesh::TriangularMesh,
                     equations, solver::FV)
    (; n_elements) = mesh
    volume = Array{eltype(data_points)}(undef, n_elements)
    dx = similar(volume)

    for element in axes(triangulation_vertices, 2)
        node1 = triangulation_vertices[1, element]
        node2 = triangulation_vertices[2, element]
        node3 = triangulation_vertices[3, element]
        x_node1 = get_node_coords(data_points, equations, solver, node1)
        x_node2 = get_node_coords(data_points, equations, solver, node2)
        x_node3 = get_node_coords(data_points, equations, solver, node3)

        # Shoelace formula
        volume[element] = x_node1[1] * x_node2[2] -
                          x_node1[2] * x_node2[1]

        volume[element] += x_node2[1] * x_node3[2] -
                           x_node2[2] * x_node3[1]

        volume[element] += x_node3[1] * x_node1[2] -
                           x_node3[2] * x_node1[1]

        volume[element] *= 0.5

        # TODO: Calculation of dx useful?
        dx[element] = max(norm(x_node1 - x_node2), norm(x_node1 - x_node3),
                          norm(x_node2 - x_node3))
    end

    return volume, dx
end

function calc_midpoints(data_points, triangulation_vertices, mesh, equations, solver)
    midpoints = Matrix{eltype(data_points)}(undef, ndims(mesh),
                                            size(triangulation_vertices, 2))
    for element in axes(triangulation_vertices, 2)
        node1 = triangulation_vertices[1, element]
        node2 = triangulation_vertices[2, element]
        node3 = triangulation_vertices[3, element]
        midpoint = (get_node_coords(data_points, equations, solver, node1) .+
                    get_node_coords(data_points, equations, solver, node2) .+
                    get_node_coords(data_points, equations, solver, node3)) ./ 3.0
        midpoints[:, element] .= midpoint
    end

    return midpoints
end

function allocate_coefficients(mesh::TriangularMesh, equations, solver::FV, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(mesh.data_points),
          nvariables(equations) * nelements(mesh, solver, cache))
end

function compute_coefficients!(u, func, t, mesh::TriangularMesh, equations,
                               solver::FV, cache)
    (; midpoints) = cache
    for element in eachelement(mesh, solver, cache)
        midpoint = get_node_coords(midpoints, equations, solver, element)
        u_midpoint = func(midpoint, t, equations)
        set_node_vars!(u, u_midpoint, equations, solver, element)
    end
end

function rhs!(du, u, t, mesh::TriangularMesh, equations, initial_condition,
              boundary_conditions, source_terms::Source, solver::FV,
              cache) where {Source}
    (; data_points) = mesh
    (; triangulation_vertices, triangulation_neighbors, volume) = cache
    (; surface_flux) = solver

    du .= zero(eltype(du))

    for element in eachelement(mesh, solver, cache)
        u_node = get_node_vars(u, equations, solver, element)

        for node_index in 1:2
            node1 = triangulation_vertices[node_index, element]
            node2 = triangulation_vertices[node_index + 1, element]
            x_node1 = get_node_coords(data_points, equations, solver, node1)
            x_node2 = get_node_coords(data_points, equations, solver, node2)

            # TODO: Save at beginning!
            face = x_node2 - x_node1
            face_size = norm(face)

            # TODO: Save the normal vector once
            # Normal vector is face vector rotated clockwise by pi/2
            normal = SVector(face[2], -face[1]) / face_size

            neighbor = triangulation_neighbors[(node_index + 1) % 3 + 1, element]
            if neighbor != 0
                u_neighbor = get_node_vars(u, equations, solver, neighbor)
                @trixi_timeit timer() "surface flux" flux=surface_flux(u_node,
                                                                       u_neighbor,
                                                                       normal,
                                                                       equations)
            else # neighbor == 0
                x_midpoint_face = 0.5 * (x_node1 + x_node2)
                @trixi_timeit timer() "boundary flux" flux=boundary_conditions(u_node,
                                                                               normal,
                                                                               x_midpoint_face,
                                                                               t,
                                                                               surface_flux,
                                                                               equations)
            end

            for v in eachvariable(equations)
                du[v, element] -= (1 / volume[element]) * face_size * flux[v]
            end
        end

        # last face
        node1 = triangulation_vertices[3, element]
        node2 = triangulation_vertices[1, element]
        x_node1 = get_node_coords(data_points, equations, solver, node1)
        x_node2 = get_node_coords(data_points, equations, solver, node2)

        # TODO: Save at beginning!
        face = x_node2 - x_node1
        face_size = norm(face)

        # TODO: Save the normal vector once
        # Normal vector is face vector rotated clockwise by pi/2
        normal = SVector(face[2], -face[1]) ./ face_size

        neighbor = triangulation_neighbors[2, element]
        if neighbor != 0
            u_neighbor = get_node_vars(u, equations, solver, neighbor)
            @trixi_timeit timer() "surface flux" flux=surface_flux(u_node, u_neighbor,
                                                                   normal, equations)
        else # neighbor == 0
            x_midpoint_face = (x_node1 .+ x_node2) ./ 2.0
            @trixi_timeit timer() "boundary flux" flux=boundary_conditions(u_node,
                                                                           normal,
                                                                           x_midpoint_face,
                                                                           t,
                                                                           surface_flux,
                                                                           equations)
        end

        for v in eachvariable(equations)
            du[v, element] -= (1 / volume[element]) * face_size * flux[v]
        end
    end

    return nothing
end

function create_cache(mesh::PolygonMesh, equations,
                      solver::FV, RealT, uEltype)
    (; data_points) = mesh

    # Triangulation and Polygon mesh
    (; triangulation_vertices, voronoi_vertices_coordinates, voronoi_vertices, voronoi_vertices_interval) = mesh

    # Neighbors
    triangulation_neighbors = delaunay_compute_neighbors(data_points,
                                                         triangulation_vertices)

    voronoi_neighbors = voronoi_compute_neighbors(triangulation_vertices,
                                                  voronoi_vertices,
                                                  voronoi_vertices_interval,
                                                  triangulation_neighbors)

    volume, dx = calc_volume(voronoi_vertices_coordinates, voronoi_vertices,
                             voronoi_vertices_interval, mesh, equations, solver)

    cache = (; voronoi_vertices_coordinates, voronoi_vertices,
             voronoi_vertices_interval, voronoi_neighbors, volume, dx)

    return cache
end

function calc_volume(voronoi_vertices_coordinates, voronoi_vertices,
                     voronoi_vertices_interval, mesh::PolygonMesh, equations,
                     solver::FV)
    n_elements = nelements(mesh, equations, solver)
    volume = zeros(eltype(voronoi_vertices_coordinates), n_elements)
    dx = similar(volume)

    for element in eachindex(volume)
        node_index_start = voronoi_vertices_interval[1, element]
        node_index_end = voronoi_vertices_interval[2, element]

        # Shoelace formula
        for i in node_index_start:(node_index_end - 1)
            node1 = voronoi_vertices[i]
            node2 = voronoi_vertices[i + 1]
            volume[element] += voronoi_vertices_coordinates[1, node1] *
                               voronoi_vertices_coordinates[2, node2] -
                               voronoi_vertices_coordinates[2, node1] *
                               voronoi_vertices_coordinates[1, node2]
        end
        node_last = voronoi_vertices[node_index_end]
        node_first = voronoi_vertices[node_index_start]
        volume[element] += voronoi_vertices_coordinates[1, node_last] *
                           voronoi_vertices_coordinates[2, node_first] -
                           voronoi_vertices_coordinates[2, node_last] *
                           voronoi_vertices_coordinates[1, node_first]
        volume[element] *= 0.5

        dx[element] = 1 # TODO
    end

    return volume, dx
end

function allocate_coefficients(mesh::PolygonMesh, equations, solver::FV, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(mesh.data_points),
          nvariables(equations) * nelements(mesh, solver, cache))
end

function compute_coefficients!(u, func, t, mesh::PolygonMesh, equations,
                               solver::FV, cache)
    (; data_points) = mesh
    for element in eachelement(mesh, solver, cache)
        x_node = get_node_coords(data_points, equations, solver, element)
        u_node = func(x_node, t, equations)
        set_node_vars!(u, u_node, equations, solver, element)
    end
end

function rhs!(du, u, t, mesh::PolygonMesh, equations, initial_condition,
              boundary_conditions, source_terms::Source, solver::FV,
              cache) where {Source}
    (; voronoi_vertices_coordinates, voronoi_vertices, voronoi_vertices_interval, voronoi_neighbors, volume) = cache
    (; surface_flux) = solver

    du .= zero(eltype(du))

    for element in eachelement(mesh, solver, cache)
        u_node = get_node_vars(u, equations, solver, element)

        nodes_begin = voronoi_vertices_interval[1, element]
        nodes_end = voronoi_vertices_interval[2, element]

        for node_index in nodes_begin:(nodes_end - 1)
            node1 = voronoi_vertices[node_index]
            node2 = voronoi_vertices[node_index + 1]
            x_node1 = get_node_coords(voronoi_vertices_coordinates, equations, solver,
                                      node1)
            x_node2 = get_node_coords(voronoi_vertices_coordinates, equations, solver,
                                      node2)

            # TODO: Save at beginning!
            face = x_node2 - x_node1
            face_size = norm(face)

            # TODO: Save the normal vector once
            # Normal vector is face vector rotated clockwise by pi/2
            normal = SVector(face[2], -face[1]) ./ face_size

            neighbor = voronoi_neighbors[node_index]
            if neighbor != 0
                u_neighbor = get_node_vars(u, equations, solver, neighbor)
                @trixi_timeit timer() "surface flux" flux=surface_flux(u_node,
                                                                       u_neighbor,
                                                                       normal,
                                                                       equations)
            else # neighbor == 0
                x_midpoint_face = 0.5 * (x_node1 + x_node2)
                @trixi_timeit timer() "boundary flux" flux=boundary_conditions(u_node,
                                                                               normal,
                                                                               x_midpoint_face,
                                                                               t,
                                                                               surface_flux,
                                                                               equations)
            end

            for v in eachvariable(equations)
                du[v, element] -= (1 / volume[element]) * face_size * flux[v]
            end
        end

        # last face
        node1 = voronoi_vertices[nodes_end]
        node2 = voronoi_vertices[nodes_begin]
        x_node1 = get_node_coords(voronoi_vertices_coordinates, equations, solver,
                                  node1)
        x_node2 = get_node_coords(voronoi_vertices_coordinates, equations, solver,
                                  node2)

        # TODO: Save at beginning!
        face = x_node2 - x_node1
        face_size = norm(face)

        # TODO: Save the normal vector once
        # Normal vector is face vector rotated clockwise by pi/2
        normal = SVector(face[2], -face[1]) ./ face_size

        neighbor = voronoi_neighbors[nodes_end]
        if neighbor != 0
            u_neighbor = get_node_vars(u, equations, solver, neighbor)
            @trixi_timeit timer() "surface flux" flux=surface_flux(u_node, u_neighbor,
                                                                   normal, equations)
        else # neighbor == 0
            x_midpoint_face = 0.5 * (x_node1 + x_node2)
            @trixi_timeit timer() "boundary flux" flux=boundary_conditions(u_node,
                                                                           normal,
                                                                           x_midpoint_face,
                                                                           t,
                                                                           surface_flux,
                                                                           equations)
        end

        for v in eachvariable(equations)
            du[v, element] -= (1 / volume[element]) * face_size * flux[v]
        end
    end

    return nothing
end
end # @muladd
