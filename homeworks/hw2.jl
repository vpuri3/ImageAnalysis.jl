#
using ImageAnalysis, Plots

img = load("../extra/foot.pgm") .|> Float32

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

# λ
l1 = 1f-1
l2 = 1f+1

ln  = euler_fwd(img, dudt_ln; dt=0.01f0, niter=200)
nl1 = euler_fwd(img, u -> dudt_nl(u; λ=l1); dt=0.01f0, niter=200)
nl2 = euler_fwd(img, u -> dudt_nl(u; λ=l2); dt=0.01f0, niter=200)

p0 = heatmap(img[end:-1:begin, :]; clims=(0,1), c=:grays)
p1 = heatmap(ln[end:-1:begin, :] ; clims=(0,1), c=:grays)
p2 = heatmap(nl1[end:-1:begin, :]; clims=(0,1), c=:grays)
p3 = heatmap(nl2[end:-1:begin, :]; clims=(0,1), c=:grays)

p0 = plot!(p0, title="Original")
p1 = plot!(p1, title="Laplacian Filter, dt=0.01, 200 iters")
p2 = plot!(p2, title="Nonlinear Filter, dt=0.01, 200 iters, λ=$l1")
p3 = plot!(p3, title="Nonlinear Filter, dt=0.01, 200 iters, λ=$l2")
##

filename = "foot_smooth"

save(filename * "_ln" * ".pgm", ln)
save(filename * "_nl1" * ".pgm", nl1)
save(filename * "_nl2" * ".pgm", nl2)

savefig(p0, "foot")
savefig(p1, filename * "_ln")
savefig(p2, filename * "_nl1")
savefig(p3, filename * "_nl2")
#
