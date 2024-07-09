module NeuroDynamics

using Lux, ComponentArrays, LinearAlgebra, SciMLSensitivity, Plots, Zygote, Distributions, Interpolations, SpecialFunctions, DifferentialEquations, Random, CairoMakie
import ChainRulesCore as CRC
using Parameters: @unpack, @with_kw
import LuxCore: AbstractExplicitContainerLayer, AbstractExplicitLayer

abstract type LatentVariableModel <: AbstractExplicitContainerLayer{(:obs_encoder, :ctrl_encoder, :dynamics, :obs_decoder, :ctrl_decoder)} end
abstract type  DynamicalSystem <: AbstractExplicitLayer end
abstract type UDE <: AbstractExplicitContainerLayer{(:vector_field,)} end
abstract type SUDE <: AbstractExplicitContainerLayer{(:drift, :drift_aug, :diffusion)} end


# Core stuff
include("core/dynamics.jl")
export SDE, ODE, sample_dynamics, phaseplot
include("core/lvm.jl")
export LatentUDE, predict
include("core/encoders.jl")
export Encoder, Identity_Encoder, Recurrent_Encoder
include("core/decoders.jl")
export Decoder, Identity_Decoder, Linear_Decoder, MLP_Decoder


# Utils
include("utils/plotting.jl")
export plot_samples, plot_ci, plot_preds, plot_phase_portrait_2d, plot_phase_portrait_3d, animate_sol, animate_timeseries, animate_oscillators, animate_spikes
include("utils/misc.jl")
export sample_rp, interpolate!, basic_tgrad, dropmean
include("utils/losses.jl")
export kl_normal, poisson_loglikelihood, normal_loglikelihood, mse, frange_cycle_linear, bits_per_spike

# Models
include("models/oscillators.jl")
export HarmonicOscillators, BistableOscillators
include("models/neural_populations.jl")
export WilsonCowan, ModernWilsonCowan
include("models/single_neuron.jl")
export FitzHughNagumo, HodgkinHuxley
end



