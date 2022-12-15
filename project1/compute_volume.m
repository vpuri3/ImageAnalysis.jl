%
filename = 'mri_dcm.rawiv_subunit_01.rawiv';

data = readRawiv(filename)
img  = data.image;

vol_voxel = prod(data.spanXYZ);
num_voxel = sum(img ~= 0, 'all');
vol = vol_voxel * num_voxel
