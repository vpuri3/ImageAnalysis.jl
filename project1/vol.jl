#
using LinearAlgebra
using RawFile

filename = "mri_dcm.rawiv_subunit_00.rawiv"

io = open(filename)
x = reinterpret(Float32, read(io, sizeof(Float32)*32))
x .= ntoh.(x)
