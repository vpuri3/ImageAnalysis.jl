%

filename = 'mri_dcm.rawiv_subunit_00.rawiv'

data = readRawiv(filename)

data.spanXYZ
data.maxXYZ ./ (data.dimXYZ - 1)

img = data.image;

vol_voxel = prod(data.spanXYZ)
num_voxel = sum(img ~= 0, 'all')
vol = vol_voxel * num_voxel
