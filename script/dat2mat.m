function dat2mat(path_dat, path_mat)
% 函数参数说明：输入的都是文件地址

% reset mat文件夹
[dir_mat, basename, ext] = fileparts(path_mat);
if exist(dir_mat, 'dir')
    disp('The dir_save already exists');
    disp('It will be deleted and recreated');
    rmdir(dir_mat, 's');
end
mkdir(dir_mat);
% 加载dat文件
fid = fopen(path_dat, 'r');
img = fread(fid, 'uint8=>uint8'); 
dims = get_3d_dat_unmodified(basename);
img = reshape(img, [dims.x, dims.y, dims.z]);
fclose(fid);
% 将dat数据，存储进mat文件
save(path_mat, 'img')
end