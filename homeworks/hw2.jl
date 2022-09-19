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

ln  = euler_fwd(img, dudt_ln; dt=0.01f0, niter=200)
nl_lam01 = euler_fwd(img, u -> dudt_nl(u; λ=1.0f0 ); dt=0.01f0, niter=200)
nl_lam10 = euler_fwd(img, u -> dudt_nl(u; λ=10.0f0); dt=0.01f0, niter=200)

p0 = heatmap(img[end:-1:begin, :]; clims=(0,1), c=:grays)
p1 = heatmap(ln[end:-1:begin, :]; clims=(0,1), c=:grays)
p2 = heatmap(nl_lam01[end:-1:begin, :]; clims=(0,1), c=:grays)
p3 = heatmap(nl_lam10[end:-1:begin, :]; clims=(0,1), c=:grays)

p0 = plot!(p0, title="Original")
p1 = plot!(p1, title="Laplacian Filter, dt=0.01, 200 iters")
p2 = plot!(p2, title="Nonlinear Filter, dt=0.01, 200 iters, λ=1")
p3 = plot!(p3, title="Nonlinear Filter, dt=0.01, 200 iters, λ=10")
##

filename = "foot_smooth"

save(filename * "_ln" * ".pgm", ln)
save(filename * "_nl_lam01" * ".pgm", nl_lam01)
save(filename * "_nl_lam10" * ".pgm", nl_lam10)

savefig(p0, "foot")
savefig(p1, filename * "_ln")
savefig(p2, filename * "_nl_lam01")
savefig(p3, filename * "_nl_lam10")
#
