function dims = get_3d_dat_unmodified(filename)
nxs = [616, 284, 495];
nys = [485, 411, 615];
nzs = [719, 722, 752];
idx = 0;
result = strsplit(filename, '_');
if strcmp(result(2), '07')
    idx = 1;
elseif strcmp(result(2), '35')
    idx = 2;
elseif strcmp(result(2), '47')
    idx = 3;
end
x = nxs(idx);
y = nys(idx);
z = nzs(idx);
dims.x = x;
dims.y = y;
dims.z = z;
end
