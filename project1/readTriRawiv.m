%
function rawiv = readRawiv(rawivName)

fid=fopen(rawivName,'r');
rawiv.numVerts      =	fread(fid,1,'uint32','b')';
rawiv.numCells      =	fread(fid,1,'uint32','b')';
rawiv.dimXYZ		=	fread(fid,3,'uint32','b')';
rawiv.originXYZ 	=	fread(fid,3,'float','b')';
rawiv.spanXYZ		=	fread(fid,3,'float','b')';
rawiv.image         =   fread(fid,prod(rawiv.dimXYZ),'*float','b');
rawiv.image         =   reshape(rawiv.image,rawiv.dimXYZ);
fclose(fid);

outName = [rawivName '_' num2str(rawiv.dimXYZ(1)) '_'...
						 num2str(rawiv.dimXYZ(2)) '_'...
						 num2str(rawiv.dimXYZ(3)) '.raw'];
fid=fopen(outName,'wb');
fwrite(fid, rawiv.image, 'float');
fclose(fid);

end

