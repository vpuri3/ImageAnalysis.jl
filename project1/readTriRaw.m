%
function raw = readTriRaw(rawName)

fid=fopen(rawName,'r');
c = textscan(fid, '%f %f %f');

x = cell2mat(c(1));
y = cell2mat(c(2));
z = cell2mat(c(3));

raw.nVerts = x(1);
raw.nElems = y(1);

x = x(2:end);
y = y(2:end);
z = z(2:end);

raw.x = x(1:raw.nVerts);
raw.y = y(1:raw.nVerts);
raw.z = z(1:raw.nVerts);

end

