%
function raw = readTriRaw(rawName)

fid=fopen(rawName,'r');
vert = textscan(fid, '%f %f %f');

x = cell2mat(vert(1));
y = cell2mat(vert(2));
z = cell2mat(vert(3));

raw.nVerts = x(1);
raw.nElems = y(1);

x = x(2:end);
y = y(2:end);
z = z(2:end);

raw.x = x(1:raw.nVerts);
raw.y = y(1:raw.nVerts);
raw.z = z(1:raw.nVerts);

C1 = x(raw.nVerts+1:end);
C2 = y(raw.nVerts+1:end);
C3 = z(raw.nVerts+1:end);

raw.A = horzcat(C1, C2, C3);

end
