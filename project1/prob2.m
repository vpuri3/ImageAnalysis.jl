%

pfile = '2BG9_pot97129.rawiv';
pfile = readRawiv(pfile);

pot = pfile.image;
xm = linspace(pfile.minXYZ(1), pfile.maxXYZ(1), pfile.dimXYZ(1));
ym = linspace(pfile.minXYZ(2), pfile.maxXYZ(2), pfile.dimXYZ(2));
zm = linspace(pfile.minXYZ(3), pfile.maxXYZ(3), pfile.dimXYZ(3));

tfile = 'iso_tri.raw';
tfile = readTriRaw(tfile);

Nv = length(tfile.x);
xyz  = horzcat(tfile.x, tfile.y, tfile.z);
pval = zeros(Nv, 1);

for i=1:Nv
    pval(i) = compute_val(pot, xyz(i,:), xm, ym, zm);
end

%val = compute_val(pot, [9.1,12.2,14.4], xm, ym, zm);

%===================================================%
% HELPER FUNCTIONS
%===================================================%
function ixyz = voxel_idx(xyz, xm, ym, zm)
    % returns index to bottom-left corner of voxel

    x = xyz(1);
    y = xyz(2);
    z = xyz(3);

    dx = xm(2) - xm(1);
    dy = ym(2) - ym(1);
    dz = zm(2) - zm(1);

    ix = floor(x / dx) + 1;
    iy = floor(y / dy) + 1;
    iz = floor(z / dz) + 1;

    ixyz = [ix, iy, iz];
end

function uvw = local_coordinates(xyz, ixyz, xm, ym, zm) 
    % xyz: point coordinates
    % vi : voxel bottom left corner index

    ix = ixyz(1);
    iy = ixyz(2);
    iz = ixyz(3);

    [x0, x1] = deal(xm(ix), xm(ix + 1));
    [y0, y1] = deal(ym(ix), ym(ix + 1));
    [z0, z1] = deal(zm(ix), zm(ix + 1));

    x = xyz(1);
    y = xyz(2);
    z = xyz(3);

    u = (x - x0) / (x1 - x0);
    v = (y - y0) / (y1 - y0);
    w = (z - z0) / (z1 - z0);

    uvw = [u, v, w];
end

function wts = intp_wts(u, ixyz)
    % get interpolation weights from array, voxel idx
    %

    ix = ixyz(1);
    iy = ixyz(2);
    iz = ixyz(3);

    wts.u000 = u(ix  , iy  , iz  );

    wts.u100 = u(ix+1, iy  , iz  );
    wts.u010 = u(ix  , iy+1, iz  );
    wts.u001 = u(ix  , iy  , iz+1);

    wts.u110 = u(ix+1, iy+1, iz  );
    wts.u101 = u(ix+1, iy  , iz+1);
    wts.u011 = u(ix  , iy+1, iz+1);

    wts.u111 = u(ix+1, iy+1, iz+1);
end

function val = interpolate(wts, uvw)
    u = uvw(1);
    v = uvw(2);
    w = uvw(3);

    val  =       wts.u000 * (1 - u) * (1 - v) * (1 - w);
%
    val = val +  wts.u100 *      u  * (1 - v) * (1 - w);
    val = val +  wts.u010 * (1 - u) *      v  * (1 - w);
    val = val +  wts.u001 * (1 - u) * (1 - v) *      w ;
%
    val = val +  wts.u110 *      u  *      v  * (1 - w);
    val = val +  wts.u101 *      u  * (1 - v) *      w ;
    val = val +  wts.u011 * (1 - u) *      v  *      w ;
%
    val = val +  wts.u111 *      u  *      v  *      w ;

end

function val = compute_val(u, xyz, xm, ym, zm)
    % u - array to sample
    % xyz - point coordinates

    ixyz = voxel_idx(xyz, xm, ym, zm);
    uvw  = local_coordinates(xyz, ixyz, xm, ym, zm);
    wts  = intp_wts(u, ixyz);

    val = interpolate(wts, uvw);
end
