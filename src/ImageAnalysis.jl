module ImageAnalysis

using NNlib: conv

"""
Convention:
    * * * *
    * * * * | x
    * * * * | 
    * * * * v
    * * * * x
     ---> y
"""

#=================================================#
# CONVOLUTIONS
#=================================================#

const LAPL_F32 = [0f0  1f0 0f0
                  1f0 -4f0 1f0
                  0f0  1f0 0f0] ./ 1f0

const DX_F32 = [-1f0
                 0f0
                 1f0] ./ 2f0

const DY_F32 = [-1f0 0f0 1f0] ./ 2f0

function apply_conv(u::AbstractMatrix{T},
                    w::AbstractVecOrMat{T}) where{T}

    U = reshape(u, (size(u)..., 1, 1))

    W = if w isa AbstractVector
        reshape(w, (size(w)..., 1, 1, 1))
    else
        reshape(w, (size(w)..., 1, 1))
    end

    pad = size(W)[1:2] .÷ 2

    V = conv(U, W; pad=pad)

    dropdims(V; dims=(3,4))
end

#=================================================#
# VECTOR CALCULUS
#=================================================#

function grad(u::AbstractArray, ws=(DX_F32, DY_F32))
    Tuple(apply_conv(u, w) for w in ws)
end

function diver(us::NTuple{D, AbstractArray},
               ws::NTuple{D, AbstractArray}=(DX_F32, DY_F32)) where{D}
    tup = Tuple(apply_conv(u, w) for (u,w) in zip(us, ws))

    sum(tup)
end

function norm2(us::NTuple{D, AbstractArray}) where{D}
    u2s = Tuple(u .* u for u in us)
    sum(u2s)
end

#=================================================#
# GAUSSIAN SMOOTHER
#=================================================#

function gauss(x, λ)
    x2 = x * x
    l2 = λ * λ
    gauss2(x2, l2)
end

function gauss2(x2, l2)
    1 / (1+ x2/l2)
end

#=================================================#
# TIME STEPPERS
#=================================================#

function euler_fwd(u, dudt_func; dt=0.01f0, niter=100)
    for i=1:niter
        du = dudt_func(u)
        u += dt*du
    end

    u
end

#=================================================#

export
       LAPL_F32, DX_F32, DY_F32,

       apply_conv,

       grad, diver, norm2,

       gauss, gauss2,

       euler_fwd

#=================================================#

end # module
