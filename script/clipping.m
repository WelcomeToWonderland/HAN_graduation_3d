function clipping(path_original, path_clipping)
% ��������˵�����ļ�·��

% ���������ļ�
[dir, basename, ext] = fileparts(path_clipping);
if exist(dir, 'dir')
    disp('The dir already exists, and it will be deleted and recreated');
    rmdir(dir, 's');
end
mkdir(dir);
% �����ļ�
file = load(path_original);
img = file.img;
% clipping
shape = size(img);
disp(size(img));
if mod(shape(1), 2) ~= 0
    img(1, :, :) = [];
end
if mod(shape(2), 2) ~= 0
    img(:, 1, :) = [];
end
if mod(shape(3), 2) ~= 0
    img(:, :, 1) = [];
end
disp(size(img));
% �����ļ�
save(path_clipping, 'img');
end