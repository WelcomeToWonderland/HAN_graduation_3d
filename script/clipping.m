function clipping(path_original, path_clipping)
% 函数参数说明：文件路径

% 创建保存文件
[dir, basename, ext] = fileparts(path_clipping);
if exist(dir, 'dir')
    disp('The dir already exists, and it will be deleted and recreated');
    rmdir(dir, 's');
end
mkdir(dir);
% 加载文件
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
% 保存文件
save(path_clipping, 'img');
end