using NeuroDynamics
using Documenter

DocMeta.setdocmeta!(NeuroDynamics, :DocTestSetup, :(using NeuroDynamics); recursive=true)

makedocs(
    modules=[NeuroDynamics],
    authors="Ahmed El-Gazzar",
    sitename="NeuroDynamics.jl",
    format=Documenter.HTML(
        canonical="https://elgazzarr.github.io/NeuroDynamics.jl",
        edit_link="main",
        assets=["assets/logo.ico"]),

    pages=[
        "Home" => "index.md",
        "Tutorials" => Any["Setting up a differentiable model" => "tutorials/1-setting_up_model.md", "Constructing a LatentSDE" => "tutorials/2-building_latentUDE.md"],
        "Examples" => Any[
            "Modeling single neuron" => "examples/Modeling_HodkingHuxely.md", 
            "Infering neural dynamics in motor cortex" => "examples/Neural_mcmaze.md",
            "Infering neural and behavioral dynamics in delayed reach task" => "examples/Joint_mcmaze.md",
            "Infering neural and behavioral dynamics in Area2 during perturbed reach task" => "examples/Joint_area2bump.md"
        ]
    ]
)

deploydocs(
    repo="github.com/elgazzarr/NeuroDynamics.jl",
    devbranch="main"
)