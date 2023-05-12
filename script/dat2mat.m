function dat2mat(path_dat, path_mat)
% ��������˵��������Ķ����ļ���ַ

% reset mat�ļ���
[dir_mat, basename, ext] = fileparts(path_mat);
if exist(dir_mat, 'dir')
    disp('The dir_save already exists');
    disp('It will be deleted and recreated');
    rmdir(dir_mat, 's');
end
mkdir(dir_mat);
% ����dat�ļ�
fid = fopen(path_dat, 'r');
img = fread(fid, 'uint8=>uint8'); 
dims = get_3d_dat_unmodified(basename);
img = reshape(img, [dims.x, dims.y, dims.z]);
fclose(fid);
% ��dat���ݣ��洢��mat�ļ�
save(path_mat, 'img')
end