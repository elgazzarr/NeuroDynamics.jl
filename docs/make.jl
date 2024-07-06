using NeuroDynamics
using Documenter

DocMeta.setdocmeta!(NeuroDynamics, :DocTestSetup, :(using NeuroDynamics); recursive=true)

makedocs(;
    modules=[NeuroDynamics],
    authors="Ahmed El-Gazzar",
    sitename="NeuroDynamics.jl",
    format=Documenter.HTML(;
        canonical="https://elgazzarr.github.io/NeuroDynamics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/elgazzarr/NeuroDynamics.jl",
    devbranch="main",
)

```@docs
MyFunction
```

