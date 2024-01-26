# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::T8codeFVMesh, equations,
                      solver, RealT, uEltype)
    elements = init_elements(mesh, RealT, uEltype)

    interfaces = init_interfaces(mesh, equations, elements)

    u_ = init_solution!(mesh, equations)

    cache = (; elements, interfaces, u_)

    return cache
end

function compute_coefficients!(u, func, t, mesh::T8codeFVMesh, equations,
                               solver::FV, cache)
    for element in eachelement(mesh, solver, cache)
        x_node = SVector(cache.elements[element].midpoint) # Save t8code variables as SVector?
        u_node = func(x_node, t, equations)
        set_node_vars!(u, u_node, equations, solver, element)
    end
end

function allocate_coefficients(mesh::T8codeFVMesh, equations, solver::FV, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(cache.elements[1].volume),
          nvariables(equations) * nelements(mesh, solver, cache))
end

function rhs!(du, u, t, mesh::T8codeFVMesh, equations, initial_condition,
              boundary_conditions, source_terms::Source, solver::FV,
              cache) where {Source}
    @trixi_timeit timer() "update neighbor data" exchange_solution!(u, mesh, equations,
                                                                    solver, cache)
    @unpack elements, interfaces, u_ = cache

    du .= zero(eltype(du))

    @trixi_timeit timer() "reconstruction" reconstruction(u_, mesh, equations, solver,
                                                          cache)

    @trixi_timeit timer() "evaluation" evaluate_interface_values!(mesh, equations,
                                                                  solver, cache)

    @trixi_timeit timer() "update du" begin
        for interface in eachinterface(solver, cache)
            element = interfaces.neighbor_ids[1, interface]
            neighbor = interfaces.neighbor_ids[2, interface]
            face = interfaces.faces[1, interface]

            # TODO: Save normal and face_areas in interface
            normal = Trixi.get_variable_wrapped(elements[element].face_normals,
                                                equations, face)
            u_ll, u_rr = get_surface_node_vars(interfaces.u, equations, solver,
                                               interface)
            @trixi_timeit timer() "surface flux" flux=solver.surface_flux(u_ll, u_rr,
                                                                          normal,
                                                                          equations)
            @trixi_timeit timer() "for loop" for v in eachvariable(equations)
                flux_ = -elements[element].face_areas[face] * flux[v]
                du[v, element] += flux_
                if neighbor <= mesh.number_elements
                    du[v, neighbor] -= flux_
                end
            end
        end
        for element in eachelement(mesh, solver, cache)
            @unpack volume = cache.elements[element]
            for v in eachvariable(equations)
                du[v, element] = (1 / volume) * du[v, element]
            end
        end
    end # timer

    return nothing
end

function reconstruction(u_, mesh::Union{T8codeFVMesh, VoronoiMesh}, equations, solver, cache)
    if solver.order == 1
        return nothing
    elseif solver.order == 2
        linear_reconstruction(u_, mesh, equations, solver, cache)
    else
        error("order $(solver.order) not supported.")
    end

    return nothing
end

function linear_reconstruction(u_, mesh::T8codeFVMesh, equations, solver, cache)
    @unpack elements = cache

    slope = zeros(eltype(u_[1].u), nvariables(equations) * ndims(mesh))

    # Approximate slope
    for element in eachelement(mesh, solver, cache)
        @unpack u = u_[element]
        @unpack num_faces, face_connectivity, face_areas, face_normals, midpoint, face_midpoints, volume = cache.elements[element]

        # Reconstruction from Hou et al. 2015
        # u_faces = [zeros(length(u)) for i in 1:num_faces]
        # distances = zeros(num_faces)
        # for face in eachindex(u_faces)
        #     face_midpoint = Trixi.get_variable_wrapped(face_midpoints, equations, face)
        #     distance = norm(face_midpoint .- midpoint)
        #     u_faces[face] .+= u ./ distance
        #     distances[face] += 1 / distance
        #     for (face_, neighbor) in enumerate(elements[element].face_connectivity[1:num_faces])
        #         face_midpoint_neighbor_ = Trixi.get_variable_wrapped(face_midpoints, equations, face_)
        #         face_neighbor = elements[element].neighbor_faces[face_]
        #         face_midpoint_neighbor = Trixi.get_variable_wrapped(elements[neighbor].face_midpoints,
        #                                                             equations, face_neighbor)
        #         if face_midpoint_neighbor_ != face_midpoint_neighbor
        #             # Periodic boundary
        #             # - The face_midpoint must be synchronous at each side of the mesh.
        #             #   Is it possible to have shifted faces?
        #             # - Distance is implemented as the sum of the two distances to the face_midpoint.
        #             #   In general, this is not the actual distance.
        #             # distance = norm(face_midpoint .- face_midpoint_neighbor_) +
        #             #            norm(face_midpoint_neighbor_ .- elements[neighbor].midpoint)
        #             distance = abs(norm(elements[neighbor].midpoint .- face_midpoint) - 2)
        #         else
        #             distance = norm(elements[neighbor].midpoint .- face_midpoint)
        #         end
        #         u_faces[face] .+= u_[neighbor].u ./ distance
        #         distances[face] += 1 / distance
        #     end
        # end
        # u_faces ./= distances
        # This version of calculating the face values results in values with less difference (expected due to the average calculation)
        # Therefore, the slope is smaller (the slope from below is about 0.5*pi, which is good for initial_condition_convergence_test;
        # the new one much smaller). This leads to in different results and an caculated order of convergence of about 1 :/

        slope .= zero(eltype(slope))
        for face in 1:num_faces
            neighbor = face_connectivity[face]
            normal = Trixi.get_variable_wrapped(face_normals, equations, face)
            face_midpoint = Trixi.get_variable_wrapped(face_midpoints, equations, face)

            face_neighbor = elements[element].neighbor_faces[face]
            face_midpoint_neighbor = Trixi.get_variable_wrapped(elements[neighbor].face_midpoints,
                                                                equations,
                                                                face_neighbor)
            if face_midpoint != face_midpoint_neighbor
                # Periodic boundary
                # - The face_midpoint must be synchronous at each side of the mesh.
                #   Is it possible to have shifted faces?
                # - Distance is implemented as the sum of the two distances to the face_midpoint.
                #   In general, this is not the actual distance.
                distance = norm(face_midpoint .- midpoint) +
                           norm(face_midpoint_neighbor .- elements[neighbor].midpoint)
            else
                distance = norm(elements[neighbor].midpoint .- midpoint)
            end
            slope_ = (u_[neighbor].u .- u) ./ distance
            u_face = u .+ slope_ .* norm(face_midpoint .- midpoint)

            for v in eachvariable(equations)
                for d in eachindex(normal)
                    slope[(v - 1) * ndims(mesh) + d] += face_areas[face] * u_face[v] *
                                                        normal[d]
                end
            end
        end
        slope .*= 1 / volume
        s = Tuple(slope) # TODO: Allocations
        u_[element] = T8codeSolutionContainer(u, s) # TODO: Allocations
    end

    exchange_ghost_data(mesh, u_)

    return nothing
end

function evaluate_interface_values!(mesh::T8codeFVMesh, equations, solver, cache)
    (; elements, interfaces, u_) = cache

    for interface in eachinterface(solver, cache)
        element = interfaces.neighbor_ids[1, interface]
        neighbor = interfaces.neighbor_ids[2, interface]
        if solver.order == 1
            for v in eachvariable(equations)
                interfaces.u[1, v, interface] = u_[element].u[v]
                interfaces.u[2, v, interface] = u_[neighbor].u[v]
            end
        elseif solver.order == 2
            @unpack midpoint, face_midpoints = elements[element]
            face = interfaces.faces[1, interface]
            face_neighbor = interfaces.faces[2, interface]

            face_midpoint = Trixi.get_variable_wrapped(face_midpoints, equations, face)
            face_midpoints_neighbor = elements[neighbor].face_midpoints
            face_midpoint_neighbor = Trixi.get_variable_wrapped(face_midpoints_neighbor,
                                                                equations,
                                                                face_neighbor)

            for v in eachvariable(equations)
                s1 = Trixi.get_variable_wrapped(u_[element].slope, equations, v)
                s2 = Trixi.get_variable_wrapped(u_[neighbor].slope, equations, v)

                s1 = dot(s1,
                         (face_midpoint .- midpoint) ./ norm(face_midpoint .- midpoint))
                s2 = dot(s2,
                         (elements[neighbor].midpoint .- face_midpoint_neighbor) ./
                         norm(elements[neighbor].midpoint .- face_midpoint_neighbor))
                # Is it useful to compare such slopes in different directions? Alternatively, one could use the normal vector.
                # But this is again not useful, since u_face would use the slope in normal direction. I think it looks good the way it is.

                slope_v = solver.slope_limiter(s1, s2)
                interfaces.u[1, v, interface] = u_[element].u[v] +
                                                slope_v *
                                                norm(face_midpoint .- midpoint)
                interfaces.u[2, v, interface] = u_[neighbor].u[v] -
                                                slope_v *
                                                norm(elements[neighbor].midpoint .-
                                                     face_midpoint_neighbor)
            end
        else
            error("Order $(solver.order) is not supported.")
        end
    end

    return nothing
end

function get_element_variables!(element_variables, u,
                                mesh::Union{T8codeFVMesh, VoronoiMesh}, equations,
                                solver, cache)
    return nothing
end

function get_node_variables!(node_variables, mesh::Union{T8codeFVMesh, VoronoiMesh},
                             equations, solver, cache)
    return nothing
end

# Container data structures
include("containers.jl")
end # @muladd
