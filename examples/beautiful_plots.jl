using CairoMakie, Lux, Random

function plot3dphase()
    odeSol(x, y, z) = Point3f(-x, 2y, z) # x'(t) = -x, y'(t) = 2y
    fig = Figure(size = (1200, 800), fontsize = 12)
    ax = Axis3(fig[1, 1], xlabel = "x¹", ylabel = "x²", zlabel="x³", backgroundcolor = :black)
    streamplot!(ax, odeSol, -2 .. 4, -2 .. 2, -3 .. 3, colormap = :viridis,
        gridsize = (10, 10, 10), arrow_size = 0.07, linewidth = 1.5)
    autolimits!(ax)
    fig
end



function ModernWilsonCowan2(N, M)
    @compact(τ = rand32(N),
             J = glorot_uniform(N, N),
             B = glorot_uniform(N, M),
             b = ones32(N),
             name="WilsonCowan ($N states, $M inputs)") do xu
        x, u = xu      
        dx = (-x + tanh.(J*x + B * u + b))./τ
        @return dx
    end
end 

rng = Random.default_rng()
M = 2
model = ModernWilsonCowan2(2, M)
p, st = Lux.setup(rng, model)

freq = rand32(M); φ = 2π .* rand32(M);
u(t) = sin.(2π .* freq .* t .+ φ) * 5.0


function create_phaseplot(model, p, st, t)
    function sol(x)
        xu = (x, u(t))
        dx = model(xu, p, st)[1]
        return Point2f(dx...)
    end
    fig = Figure(size = (1200, 900), backgroundcolor = :transparent)
    ax = CairoMakie.Axis(fig[1, 1], xlabel = "x¹", ylabel = "x²", backgroundcolor = :transparent)
    streamplot!(ax, sol, -5 .. 5, -10 .. 10, colormap = Reverse(:viridis),
        gridsize = (40, 40), arrow_size = 15, linewidth = 3)
    fig
end

t = 0.0
fig = create_phaseplot(model, p, st, t)
save("/Users/ahmed.elgazzar/Code/MyPackages/NeuroDynamics/NeuroDynamics/figures/phaseplot_$(t).png", fig)