#

println("Activating environment")
import Pkg
Pkg.activate("../..")
Pkg.instantiate()

println("Importing ImageAnalysis.jl")
using ImageAnalysis

println("Importing external packages")
using Images: load, save
using Plots: plot, plot!, heatmap, savefig

println("Loading image data")
img = load("foot.pgm") .|> Float32

"""
dudt function for linear Laplace smoother

to be passed down to a time-stepper
"""
function dudt_ln(u::AbstractArray)
    apply_conv(u, LAPL_F32)
end

"""
 dudt function for nonlinear Gaussian smoother

to be passed down to a time-stepper
"""
function dudt_nl(u::AbstractArray; λ=1f0)
    l2  = λ^2
    ∇u  = grad(u)
    ∇u2 = norm2(∇u)
    g   = gauss2.(∇u2, l2)

    rhs = Tuple(g .* uxi for uxi in ∇u)

    diver(rhs)
end

# λ
l1 = 1f-1
l2 = 1f+1

println("Applying linear filter with dt=0.01 for 50 iterations, and 200 iterations")
ln1 = euler_fwd(img, dudt_ln; dt=0.01f0, niter=50)
ln2 = euler_fwd(img, dudt_ln; dt=0.01f0, niter=200)

println("Applying linear filter with dt=0.01 with λ=0.1, 10.0 for 200 iterations")
nl1 = euler_fwd(img, u -> dudt_nl(u; λ=l1); dt=0.01f0, niter=200)
nl2 = euler_fwd(img, u -> dudt_nl(u; λ=l2); dt=0.01f0, niter=200)

println("Producing plots")
p0 = heatmap(img[end:-1:begin, :]; clims=(0,1), c=:grays)
p1 = heatmap(ln1[end:-1:begin, :]; clims=(0,1), c=:grays)
p2 = heatmap(ln2[end:-1:begin, :]; clims=(0,1), c=:grays)
p3 = heatmap(nl1[end:-1:begin, :]; clims=(0,1), c=:grays)
p4 = heatmap(nl2[end:-1:begin, :]; clims=(0,1), c=:grays)

p0 = plot!(p0, title="Original")
p1 = plot!(p1, title="Laplacian Filter, dt=0.01, 50  iters")
p2 = plot!(p2, title="Laplacian Filter, dt=0.01, 200 iters")
p3 = plot!(p3, title="Nonlinear Filter, dt=0.01, 200 iters, λ=$l1")
p4 = plot!(p4, title="Nonlinear Filter, dt=0.01, 200 iters, λ=$l2")

filename = "foot_smooth"

println("Saving files $(filename)*.png")

savefig(p0, "foot")
savefig(p1, filename * "_ln1")
savefig(p2, filename * "_ln2")
savefig(p3, filename * "_nl1")
savefig(p4, filename * "_nl2")

println("Saving files $(filename)*.pgm")

save(filename * "_ln1" * ".pgm", ln1)
save(filename * "_ln2" * ".pgm", ln2)
save(filename * "_nl1" * ".pgm", nl1)
save(filename * "_nl2" * ".pgm", nl2)
#
