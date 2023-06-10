function show_3d(path, folder_save, basename, mat_field, is_untranslated, is_unclipping, reset)
% 参数说明
% 参数1~3：必须参数
% path：文件路径
% basename：还用于图片名构成
% 参数3~4：mat文件参数
% 参数5~6：dat文件参数
% 参数7：默认重置保存文件夹
if nargin<7
   reset = true; 
end

% 创建保存文件夹
dir_root = fullfile('..', 'script_results');
dir_save = fullfile(dir_root, folder_save);
print = sprintf('dir_save: %s', dir_save);
disp(print);
if exist(dir_save, 'dir')
    disp('The dir_save already exists');
    if reset
        disp('It will be deleted and recreated');
        rmdir(dir_save, 's');
    end
end
mkdir(dir_save);
% 根据文件类型，读取数据
result = strsplit(path, '.');
ext = result{end};
if strcmp(ext, 'DAT')
    fileID = fopen(path, 'r');  % 打开文件
    data = fread(fileID, Inf, 'uint8');  % 以 uint8 类型读取数据
    fclose(fileID);  % 关闭文件
    % 恢复三维数组形状
    if is_unclipping
        dims = get_3d_dat_unmodified(basename);
    else
        dims = get_3d_dat(basename);
    end
    data = reshape(data, [dims.x, dims.y, dims.z]);
    % dat数据还原，除0外，所有元素加1
    if ~is_untranslated
        data(data ~= 0) = data(data ~= 0) + 1;
    end
elseif strcmp(ext, 'mat')
    file = load(path);
    % 从结构体中，获取属性（数据）
    if strcmp(mat_field, 'f1')
        data = file.f1;
    elseif strcmp(mat_field, 'imgout')
        data = file.imgout;
    elseif strcmp(mat_field, 'img')
        data = file.img;
    end
    disp(class(data));
    % mat数据还原，加1000
    if ~is_untranslated
        if strcmp(mat_field, 'img')
            data(data ~= 0) = data(data ~= 0) + 1;
        else
            data = data + 1000;
        end
    end
else
    disp('It is not a DAT file or mat file');
end
% 展示保存切片数据
shape = size(data);
% i代表维度，j代表同一维度不同位置
for i = 1:3
    % 不同维度的长度
    length = shape(i);
    for j = 1:3
        pos = floor(length / 4) * j;
        if i == 1
            temp = data(pos, :, :);
            temp = reshape(temp, shape(2), shape(3));
        elseif i == 2
            temp = data(:, pos, :);
            temp = reshape(temp, shape(1), shape(3));
        elseif i == 3
            temp = data(:, :, pos);
            temp = reshape(temp, shape(1), shape(2));
        end
        savename = strcat(basename, sprintf('_%d_%d.png', uint8(i), uint8(j)));
        path_save = fullfile(dir_save, savename);
%         print = sprintf('path_save: %s', path_save);
%         disp(print);
%         disp(size(temp));
        imagesc(temp);
        saveas(gcf, path_save);
%         saveas(gca, path_save); 
    end
end

end