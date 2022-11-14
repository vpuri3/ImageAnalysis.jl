#
using LinearAlgebra

filename = "mri_dcm.rawiv_subunit_00.rawiv"

io = open(filename)
x = reinterpret(Float64, read(io, sizeof(Float64)*10))
x .= ntoh.(x)
