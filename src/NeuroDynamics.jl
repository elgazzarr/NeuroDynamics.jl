module NeuroDynamics

using Lux, ComponentArrays, LinearAlgebra, SciMLSensitivity, Plots, Zygote, Distributions, Interpolations, SpecialFunctions, DifferentialEquations, Random
import ChainRulesCore as CRC
import LuxCore: AbstractExplicitContainerLayer, AbstractExplicitLayer

abstract type LatentVariableModel <: AbstractExplicitContainerLayer{(:obs_encoder, :ctrl_encoder, :dynamics, :obs_decoder, :ctrl_decoder)} end
abstract type  DynamicalSystem <: AbstractExplicitLayer end
abstract type UDE <: AbstractExplicitContainerLayer{(:vector_field,)} end
abstract type SUDE <: AbstractExplicitContainerLayer{(:drift, :drift_aug, :diffusion)} end


# Core stuff
include("core/dynamics.jl")
export SDE, ODE
include("core/lvm.jl")
export LatentUDE, predict
include("core/encoders.jl")
export Encoder, Identity_Encoder, Recurrent_Encoder
include("core/decoders.jl")
export Decoder, Identity_Decoder, Linear_Decoder, MLP_Decoder


# Utils stuff
include("utils/plotting.jl")
export plot_samples, plot_ci, plot_preds, plot_phase_portrait_2d, plot_phase_portrait_3d, animate_sol, animate_timeseries, animate_oscillators
include("utils/misc.jl")
export sample_rp, interpolate!, basic_tgrad, dropmean
include("utils/losses.jl")
export kl_normal, poisson_loglikelihood, normal_loglikelihood, mse, frange_cycle_linear, bits_per_spike
include("systems/oscillators.jl")
export SlOscillators, HarmonicOscillators, BistableOscillators
include("systems/neural_populations.jl")
export ModernWilsonCowan

end
