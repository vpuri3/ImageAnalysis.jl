#
using Plots

include("ImageAnalysis.jl")
using Main.ImageAnalysis

img = load("foot.pgm") .|> Float32

function dudt_ln(u::AbstractArray)
    apply_conv(u, LAPL_F32)
end

function dudt_nl(u::AbstractArray; λ=1f0)
    l2  = λ^2
    ∇u  = grad(u)
    ∇u2 = norm2(∇u)
    g   = gauss2.(∇u2, l2)

    rhs = Tuple(g .* uxi for uxi in ∇u)

    diver(rhs)
end

u = dudt_ln(img)
u = dudt_nl(img)

heatmap(u; clims=(0,1), c=:grays)

#filename = "chodu"
#save(filename * ".png", img)
#
