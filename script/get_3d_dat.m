function dims = get_3d_dat(filename)
nxs = [616, 284, 494, ...
       616, 284, 494, ...
       616, 284, 494];
nys = [484, 410, 614, ...
       484, 410, 614, ...
       484, 410, 614];
% original
% train
% test
nzs = [718, 722, 752, ...
       318, 322, 352, ...
       400, 400, 400];
idx = 0;
result = strsplit(filename, '_');
if strcmp(result(2), '07')
    idx = 1;
elseif strcmp(result(2), '35')
    idx = 2;
elseif strcmp(result(2), '47')
    idx = 3;
end

multiple = 0;
if strcmp(result(end), 'train')
    multiple = 1;
elseif strcmp(result(end), 'test')
    multiple = 2;
else
    multiple = 0;
end;

x = nxs(3 * multiple + idx);
y = nys(3 * multiple + idx);
z = nzs(3 * multiple + idx);
dims.x = x;
dims.y = y;
dims.z = z;
end