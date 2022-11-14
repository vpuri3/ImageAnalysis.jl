function rawiv = readRawiv(rawivName)
%	Read rawiv data format into Matlab and saveas a raw file
%   http://ccvweb.csres.utexas.edu/docs/data-formats/rawiv.html
%
%   Useage:
%   rawiv = readRawiv(rawivName)
%
%   Example
%   rawiv = readRawiv('head.rawiv');
%
%   Author:     Sheng Yue
%   Email:      sheng.yue.84@gmail.com
%   Created:    18 March 2011
%   Version:    1.0
fid=fopen(rawivName,'r');
rawiv.minXYZ		=	fread(fid,3,'float','b')';
rawiv.maxXYZ		=	fread(fid,3,'float','b')';
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
fwrite(fid, rawiv.image, 'float')
fclose(fid);

end

