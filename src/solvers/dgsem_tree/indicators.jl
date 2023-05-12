# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


abstract type AbstractIndicator end

function create_cache(typ::Type{IndicatorType}, semi) where {IndicatorType<:AbstractIndicator}
  create_cache(typ, mesh_equations_solver_cache(semi)...)
end

function get_element_variables!(element_variables, indicator::AbstractIndicator, ::VolumeIntegralShockCapturingHG)
  element_variables[:indicator_shock_capturing] = indicator.cache.alpha
  return nothing
end

function get_element_variables!(element_variables, indicator::AbstractIndicator, ::VolumeIntegralShockCapturingSubcell)
  element_variables[:smooth_indicator_elementwise] = indicator.IndicatorHG.cache.alpha
  return nothing
end


"""
    IndicatorHennemannGassner(equations::AbstractEquations, basis;
                              alpha_max=0.5,
                              alpha_min=0.001,
                              alpha_smooth=true,
                              variable)
    IndicatorHennemannGassner(semi::AbstractSemidiscretization;
                              alpha_max=0.5,
                              alpha_min=0.001,
                              alpha_smooth=true,
                              variable)

Indicator used for shock-capturing (when passing the `equations` and the `basis`)
or adaptive mesh refinement (AMR, when passing the `semi`).

See also [`VolumeIntegralShockCapturingHG`](@ref).

## References

- Hennemann, Gassner (2020)
  "A provably entropy stable subcell shock capturing approach for high order split form DG"
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
struct IndicatorHennemannGassner{RealT<:Real, Variable, Cache} <: AbstractIndicator
  alpha_max::RealT
  alpha_min::RealT
  alpha_smooth::Bool
  variable::Variable
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorHennemannGassner(equations::AbstractEquations, basis;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   variable)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorHennemannGassner, equations, basis)
  IndicatorHennemannGassner{typeof(alpha_max), typeof(variable), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorHennemannGassner(semi::AbstractSemidiscretization;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   variable)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorHennemannGassner, semi)
  IndicatorHennemannGassner{typeof(alpha_max), typeof(variable), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, variable, cache)
end


function Base.show(io::IO, indicator::IndicatorHennemannGassner)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorHennemannGassner(")
  print(io, indicator.variable)
  print(io, ", alpha_max=", indicator.alpha_max)
  print(io, ", alpha_min=", indicator.alpha_min)
  print(io, ", alpha_smooth=", indicator.alpha_smooth)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorHennemannGassner)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
             "max. α" => indicator.alpha_max,
             "min. α" => indicator.alpha_min,
             "smooth α" => (indicator.alpha_smooth ? "yes" : "no"),
            ]
    summary_box(io, "IndicatorHennemannGassner", setup)
  end
end



"""
    IndicatorLöhner (equivalent to IndicatorLoehner)

    IndicatorLöhner(equations::AbstractEquations, basis;
                    f_wave=0.2, variable)
    IndicatorLöhner(semi::AbstractSemidiscretization;
                    f_wave=0.2, variable)

AMR indicator adapted from a FEM indicator by Löhner (1987), also used in the
FLASH code as standard AMR indicator.
The indicator estimates a weighted second derivative of a specified variable locally.

When constructed to be used for AMR, pass the `semi`. Pass the `equations`,
and `basis` if this indicator should be used for shock capturing.

## References

- Löhner (1987)
  "An adaptive finite element scheme for transient problems in CFD"
  [doi: 10.1016/0045-7825(87)90098-3](https://doi.org/10.1016/0045-7825(87)90098-3)
- http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node59.html#SECTION05163100000000000000
"""
struct IndicatorLöhner{RealT<:Real, Variable, Cache} <: AbstractIndicator
  f_wave::RealT # TODO: Taal documentation
  variable::Variable
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorLöhner(equations::AbstractEquations, basis;
                         f_wave=0.2, variable)
  cache = create_cache(IndicatorLöhner, equations, basis)
  IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorLöhner(semi::AbstractSemidiscretization;
                         f_wave=0.2, variable)
  cache = create_cache(IndicatorLöhner, semi)
  IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable, cache)
end


function Base.show(io::IO, indicator::IndicatorLöhner)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorLöhner(")
  print(io, "f_wave=", indicator.f_wave, ", variable=", indicator.variable, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorLöhner)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
             "f_wave" => indicator.f_wave,
            ]
    summary_box(io, "IndicatorLöhner", setup)
  end
end

const IndicatorLoehner = IndicatorLöhner

# dirty Löhner estimate, direction by direction, assuming constant nodes
@inline function local_löhner_estimate(um::Real, u0::Real, up::Real, löhner::IndicatorLöhner)
  num = abs(up - 2 * u0 + um)
  den = abs(up - u0) + abs(u0-um) + löhner.f_wave * (abs(up) + 2 * abs(u0) + abs(um))
  return num / den
end


"""
    IndicatorIDP

TODO: docstring

Blending indicator used for subcell shock-capturing [`VolumeIntegralShockCapturingSubcell`](@ref) proposed by
- Rueda-Ramírez, Pazner, Gassner (2022)
  "Subcell Limiting Strategies for Discontinuous Galerkin Spectral Element Methods"
- Pazner (2020)
  "Sparse invariant domain preserving discontinuous Galerkin methods with subcell convex limiting"
  [arXiv:2004.08503](https://doi.org/10.1016/j.cma.2021.113876)

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct IndicatorIDP{RealT<:Real, Cache, Indicator} <: AbstractIndicator
  IDPDensityTVD::Bool
  IDPPressureTVD::Bool
  IDPPositivity::Bool
  IDPSpecEntropy::Bool
  IDPMathEntropy::Bool
  BarStates::Bool
  cache::Cache
  positCorrFactor::RealT          # Correction factor for IDPPositivity
  IDPMaxIter::Int                 # Maximal number of iterations for Newton's method
  newton_tol::Tuple{RealT, RealT} # Relative and absolute tolerances for Newton's method
  IDPgamma::RealT                 # Constant for the subcell limiting of convex (nonlinear) constraints
                                  # (must be IDPgamma>=2*d, where d is the number of dimensions of the problem)
  IDPCheckBounds::Bool
  indicator_smooth::Bool          # activates smoothness indicator: IndicatorHennemannGassner
  thr_smooth::RealT               # threshold for smoothness indicator
  IndicatorHG::Indicator
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorIDP(equations::AbstractEquations, basis;
                      IDPDensityTVD=false,
                      IDPPressureTVD=false,
                      IDPPositivity=false,
                      IDPSpecEntropy=false,
                      IDPMathEntropy=false,
                      BarStates=true,
                      positCorrFactor=0.1, IDPMaxIter=10,
                      newton_tol=(1.0e-12, 1.0e-14), IDP_gamma=2*ndims(equations),
                      IDPCheckBounds=false,
                      indicator_smooth=false, thr_smooth=0.1, variable_smooth=density_pressure)

  if IDPMathEntropy && IDPSpecEntropy
    error("Only one of the two can be selected: IDPMathEntropy/IDPSpecEntropy")
  end

  length = 2 * (IDPDensityTVD + IDPPressureTVD) + IDPSpecEntropy + IDPMathEntropy +
              min(IDPPositivity, !IDPDensityTVD) + min(IDPPositivity, !IDPPressureTVD)

  cache = create_cache(IndicatorIDP, equations, basis, length, BarStates)

  if indicator_smooth
    IndicatorHG = IndicatorHennemannGassner(equations, basis, alpha_max=1.0, alpha_smooth=false,
                                            variable=variable_smooth)
  else
    IndicatorHG = nothing
  end
  IndicatorIDP{typeof(positCorrFactor), typeof(cache), typeof(IndicatorHG)}(IDPDensityTVD, IDPPressureTVD,
      IDPPositivity, IDPSpecEntropy, IDPMathEntropy, BarStates, cache, positCorrFactor, IDPMaxIter,
      newton_tol, IDP_gamma, IDPCheckBounds, indicator_smooth, thr_smooth, IndicatorHG)
end

function Base.show(io::IO, indicator::IndicatorIDP)
  @nospecialize indicator # reduce precompilation time
  @unpack IDPDensityTVD, IDPPressureTVD, IDPPositivity, IDPSpecEntropy, IDPMathEntropy = indicator

  print(io, "IndicatorIDP(")
  if !(IDPDensityTVD || IDPPressureTVD || IDPPositivity || IDPSpecEntropy || IDPMathEntropy)
    print(io, "No limiter selected => pure DG method")
  else
    print(io, "limiter=(")
    IDPDensityTVD  && print(io, "IDPDensityTVD, ")
    IDPPressureTVD && print(io, "IDPPressureTVD with positivity correlation factor of ",
                            indicator.positCorrFactor, ", ")
    IDPPositivity  && print(io, "IDPPositivity, ")
    IDPSpecEntropy && print(io, "IDPSpecEntropy, ")
    IDPMathEntropy && print(io, "IDPMathEntropy, ")
    print(io, "), ")
  end
  indicator.indicator_smooth && print(io, ", Smoothness indicator: ", indicator.IndicatorHG,
    " with threshold ", indicator.thr_smooth, "), ")
  print(io, "Local bounds with ")
  if indicator.BarStates
    print(io, "Bar States")
  else
    print(io, "FV solution")
  end
  print(io, ")")
end

function get_node_variables!(node_variables, indicator::IndicatorIDP, ::VolumeIntegralShockCapturingSubcell, equations)
  node_variables[:indicator_shock_capturing] = indicator.cache.ContainerShockCapturingIndicator.alpha
  # TODO BB: Im ersten Zeitschritt scheint alpha noch nicht befüllt zu sein.
  return nothing
end


"""
IndicatorMCL

TODO: docstring

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct IndicatorMCL{RealT<:Real, Cache, Indicator} <: AbstractIndicator
  cache::Cache
  DensityLimiter::Bool
  DensityAlphaForAll::Bool
  SequentialLimiter::Bool
  ConservativeLimiter::Bool
  PressurePositivityLimiterKuzmin::Bool       # synchronized pressure limiting à la Kuzmin
  PressurePositivityLimiterKuzminExact::Bool  # Only for PressurePositivityLimiterKuzmin=true: Use the exact calculation of alpha
  DensityPositivityLimiter::Bool
  DensityPositivityCorrelationFactor::RealT
  SemiDiscEntropyLimiter::Bool                # synchronized semidiscrete entropy fix
  IDPCheckBounds::Bool
  indicator_smooth::Bool                      # activates smoothness indicator: IndicatorHennemannGassner
  thr_smooth::RealT                           # threshold for smoothness indicator
  IndicatorHG::Indicator
  Plotting::Bool
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorMCL(equations::AbstractEquations, basis;
                      DensityLimiter=true,                  # Impose local maximum/minimum for cons(1) based on bar states
                      DensityAlphaForAll=false,             # Use the cons(1) blending coefficient for all quantities
                      SequentialLimiter=true,               # Impose local maximum/minimum for variables phi:=cons(i)/cons(1) i 2:nvariables based on bar states
                      ConservativeLimiter=false,            # Impose local maximum/minimum for conservative variables 2:nvariables based on bar states
                      PressurePositivityLimiterKuzmin=false,# Impose positivity for pressure â la Kuzmin
                      PressurePositivityLimiterKuzminExact=true,# Only for PressurePositivityLimiterKuzmin=true: Use the exact calculation of alpha
                      DensityPositivityLimiter=false,       # Impose positivity for cons(1)
                      DensityPositivityCorrelationFactor=0.0,# Correlation Factor for DensityPositivityLimiter in [0,1)
                      SemiDiscEntropyLimiter=false,
                      IDPCheckBounds=false,
                      indicator_smooth=false, thr_smooth=0.1, variable_smooth=density_pressure,
                      Plotting=true)
  if SequentialLimiter && ConservativeLimiter
    error("Only one of the two can be selected: SequentialLimiter/ConservativeLimiter")
  end
  cache = create_cache(IndicatorMCL, equations, basis, PressurePositivityLimiterKuzmin)
  if indicator_smooth
    IndicatorHG = IndicatorHennemannGassner(equations, basis, alpha_smooth=false,
                                            variable=variable_smooth)
  else
    IndicatorHG = nothing
  end
  IndicatorMCL{typeof(thr_smooth), typeof(cache), typeof(IndicatorHG)}(cache,
    DensityLimiter, DensityAlphaForAll, SequentialLimiter, ConservativeLimiter,
    PressurePositivityLimiterKuzmin, PressurePositivityLimiterKuzminExact,
    DensityPositivityLimiter, DensityPositivityCorrelationFactor, SemiDiscEntropyLimiter,
    IDPCheckBounds, indicator_smooth, thr_smooth, IndicatorHG, Plotting)
end

function Base.show(io::IO, indicator::IndicatorMCL)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorMCL(")
  indicator.DensityLimiter && print(io, "; dens")
  indicator.DensityAlphaForAll && print(io, "; dens alpha ∀")
  indicator.SequentialLimiter && print(io, "; seq")
  indicator.ConservativeLimiter && print(io, "; cons")
  if indicator.PressurePositivityLimiterKuzmin
    if indicator.PressurePositivityLimiterKuzminExact
      print(io, "; pres (Kuzmin ex)")
    else
      print(io, "; pres (Kuzmin)")
    end
  end
  indicator.DensityPositivityLimiter && print(io, "; dens pos")
  (indicator.DensityPositivityCorrelationFactor != 0.0) && print(io, " with correlation factor $(indicator.DensityPositivityCorrelationFactor)")
  indicator.SemiDiscEntropyLimiter && print(io, "; semid. entropy")
  indicator.indicator_smooth && print(io, "; Smoothness indicator: ", indicator.IndicatorHG,
    " with threshold ", indicator.thr_smooth)
  print(io, ")")
end

function get_node_variables!(node_variables, indicator::IndicatorMCL, ::VolumeIntegralShockCapturingSubcell, equations)
  if !indicator.Plotting
    return nothing
  end
  @unpack alpha = indicator.cache.ContainerShockCapturingIndicator
  variables = varnames(cons2cons, equations)
  for v in eachvariable(equations)
    s = Symbol("alpha_", variables[v])
    node_variables[s] = alpha[v, ntuple(_ -> :, size(alpha, 2) + 1)...]
  end

  if indicator.PressurePositivityLimiterKuzmin
    @unpack alpha_pressure = indicator.cache.ContainerShockCapturingIndicator
    node_variables[:alpha_pressure] = alpha_pressure
  end

  if indicator.SemiDiscEntropyLimiter
    @unpack alpha_entropy = indicator.cache.ContainerShockCapturingIndicator
    node_variables[:alpha_entropy] = alpha_entropy
  end

  @unpack alpha_eff = indicator.cache.ContainerShockCapturingIndicator
  for v in eachvariable(equations)
    s = Symbol("alpha_effective_", variables[v])
    node_variables[s] = alpha_eff[v, ntuple(_ -> :, size(alpha, 2) + 1)...]
  end

  @unpack alpha_mean = indicator.cache.ContainerShockCapturingIndicator
  for v in eachvariable(equations)
    s = Symbol("alpha_mean_", variables[v])
    node_variables[s] = copy(alpha_mean[v, ntuple(_ -> :, size(alpha, 2) + 1)...])
  end

  @unpack alpha_mean_pressure = indicator.cache.ContainerShockCapturingIndicator
  if indicator.PressurePositivityLimiterKuzmin
    @unpack alpha_mean_pressure = indicator.cache.ContainerShockCapturingIndicator
    node_variables[:alpha_mean_pressure] = alpha_mean_pressure
  end

  @unpack alpha_mean_entropy = indicator.cache.ContainerShockCapturingIndicator
  if indicator.SemiDiscEntropyLimiter
    @unpack alpha_mean_entropy = indicator.cache.ContainerShockCapturingIndicator
    node_variables[:alpha_mean_entropy] = alpha_mean_entropy
  end

  return nothing
end


"""
    IndicatorMax(equations::AbstractEquations, basis; variable)
    IndicatorMax(semi::AbstractSemidiscretization; variable)

A simple indicator returning the maximum of `variable` in an element.
When constructed to be used for AMR, pass the `semi`. Pass the `equations`,
and `basis` if this indicator should be used for shock capturing.
"""
struct IndicatorMax{Variable, Cache<:NamedTuple} <: AbstractIndicator
  variable::Variable
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorMax(equations::AbstractEquations, basis;
                      variable)
  cache = create_cache(IndicatorMax, equations, basis)
  IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorMax(semi::AbstractSemidiscretization;
                      variable)
  cache = create_cache(IndicatorMax, semi)
  return IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end


function Base.show(io::IO, indicator::IndicatorMax)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorMax(")
  print(io, "variable=", indicator.variable, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorMax)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
            ]
    summary_box(io, "IndicatorMax", setup)
  end
end

"""
    IndicatorNeuralNetwork

Artificial neural network based indicator used for shock-capturing or AMR.
Depending on the indicator_type, different input values and corresponding trained networks are used.

`indicator_type = NeuralNetworkPerssonPeraire()`
- Input: The energies in lower modes as well as nnodes(dg).

`indicator_type = NeuralNetworkRayHesthaven()`
- 1d Input: Cell average of the cell and its neighboring cells as well as the interface values.
- 2d Input: Linear modal values of the cell and its neighboring cells.

- Ray, Hesthaven (2018)
  "An artificial neural network as a troubled-cell indicator"
  [doi:10.1016/j.jcp.2018.04.029](https://doi.org/10.1016/j.jcp.2018.04.029)
- Ray, Hesthaven (2019)
  "Detecting troubled-cells on two-dimensional unstructured grids using a neural network"
  [doi:10.1016/j.jcp.2019.07.043](https://doi.org/10.1016/j.jcp.2019.07.043)

`indicator_type = CNN (Only in 2d)`
- Based on convolutional neural network.
- 2d Input: Interpolation of the nodal values of the `indicator.variable` to the 4x4 LGL nodes.

If `alpha_continuous == true` the continuous network output for troubled cells (`alpha > 0.5`) is considered.
If the cells are good (`alpha < 0.5`), `alpha` is set to `0`.
If `alpha_continuous == false`, the blending factor is set to `alpha = 0` for good cells and
`alpha = 1` for troubled cells.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

"""
struct IndicatorNeuralNetwork{IndicatorType, RealT<:Real, Variable, Chain, Cache} <: AbstractIndicator
  indicator_type::IndicatorType
  alpha_max::RealT
  alpha_min::RealT
  alpha_smooth::Bool
  alpha_continuous::Bool
  alpha_amr::Bool
  variable::Variable
  network::Chain
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorNeuralNetwork(equations::AbstractEquations, basis;
                                indicator_type,
                                alpha_max=0.5,
                                alpha_min=0.001,
                                alpha_smooth=true,
                                alpha_continuous=true,
                                alpha_amr=false,
                                variable,
                                network)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  IndicatorType = typeof(indicator_type)
  cache = create_cache(IndicatorNeuralNetwork{IndicatorType}, equations, basis)
  IndicatorNeuralNetwork{IndicatorType, typeof(alpha_max), typeof(variable), typeof(network), typeof(cache)}(
      indicator_type, alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable,
      network, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorNeuralNetwork(semi::AbstractSemidiscretization;
                                indicator_type,
                                alpha_max=0.5,
                                alpha_min=0.001,
                                alpha_smooth=true,
                                alpha_continuous=true,
                                alpha_amr=true,
                                variable,
                                network)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  IndicatorType = typeof(indicator_type)
  cache = create_cache(IndicatorNeuralNetwork{IndicatorType}, semi)
  IndicatorNeuralNetwork{IndicatorType, typeof(alpha_max), typeof(variable), typeof(network), typeof(cache)}(
      indicator_type, alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable,
      network, cache)
end


function Base.show(io::IO, indicator::IndicatorNeuralNetwork)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorNeuralNetwork(")
  print(io, indicator.indicator_type)
  print(io, ", alpha_max=", indicator.alpha_max)
  print(io, ", alpha_min=", indicator.alpha_min)
  print(io, ", alpha_smooth=", indicator.alpha_smooth)
  print(io, ", alpha_continuous=", indicator.alpha_continuous)
  print(io, indicator.variable)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorNeuralNetwork)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator type" => indicator.indicator_type,
             "max. α" => indicator.alpha_max,
             "min. α" => indicator.alpha_min,
             "smooth α" => (indicator.alpha_smooth ? "yes" : "no"),
             "continuous α" => (indicator.alpha_continuous ? "yes" : "no"),
             "indicator variable" => indicator.variable,
            ]
    summary_box(io, "IndicatorNeuralNetwork", setup)
  end
end


# Convert probability for troubled cell to indicator value for shockcapturing/AMR
@inline function probability_to_indicator(probability_troubled_cell, alpha_continuous, alpha_amr,
                                          alpha_min, alpha_max)
  # Initialize indicator to zero
  alpha_element = zero(probability_troubled_cell)

  if alpha_continuous && !alpha_amr
    # Set good cells to 0 and troubled cells to continuous value of the network prediction
    if probability_troubled_cell > 0.5
      alpha_element = probability_troubled_cell
    else
      alpha_element = zero(probability_troubled_cell)
    end

    # Take care of the case close to pure FV
    if alpha_element > 1 - alpha_min
      alpha_element = one(alpha_element)
    end

    # Scale the probability for a troubled cell (in [0,1]) to the maximum allowed alpha
    alpha_element *= alpha_max
  elseif !alpha_continuous && !alpha_amr
    # Set good cells to 0 and troubled cells to 1
    if probability_troubled_cell > 0.5
      alpha_element = alpha_max
    else
      alpha_element = zero(alpha_max)
    end
  elseif alpha_amr
    # The entire continuous output of the neural network is used for AMR
    alpha_element  = probability_troubled_cell

    # Scale the probability for a troubled cell (in [0,1]) to the maximum allowed alpha
    alpha_element *= alpha_max
  end

  return alpha_element
end


"""
    NeuralNetworkPerssonPeraire

Indicator type for creating an `IndicatorNeuralNetwork` indicator.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`IndicatorNeuralNetwork`](@ref)
"""
struct NeuralNetworkPerssonPeraire end

"""
    NeuralNetworkRayHesthaven

Indicator type for creating an `IndicatorNeuralNetwork` indicator.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`IndicatorNeuralNetwork`](@ref)
"""
struct NeuralNetworkRayHesthaven end

"""
    NeuralNetworkCNN

Indicator type for creating an `IndicatorNeuralNetwork` indicator.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`IndicatorNeuralNetwork`](@ref)
"""
struct NeuralNetworkCNN end

end # @muladd
