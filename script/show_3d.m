function show_3d(path, folder_save, basename, mat_field, is_untranslated, is_unclipping, reset)
% ����˵��
% ����1~3���������
% path���ļ�·��
% basename��������ͼƬ������
% ����3~4��mat�ļ�����
% ����5~6��dat�ļ�����
% ����7��Ĭ�����ñ����ļ���
if nargin<7
   reset = true; 
end

% ���������ļ���
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
% �����ļ����ͣ���ȡ����
result = strsplit(path, '.');
ext = result{end};
if strcmp(ext, 'DAT')
    fileID = fopen(path, 'r');  % ���ļ�
    data = fread(fileID, Inf, 'uint8');  % �� uint8 ���Ͷ�ȡ����
    fclose(fileID);  % �ر��ļ�
    % �ָ���ά������״
    if is_unclipping
        dims = get_3d_dat_unmodified(basename);
    else
        dims = get_3d_dat(basename);
    end
    data = reshape(data, [dims.x, dims.y, dims.z]);
    % dat���ݻ�ԭ����0�⣬����Ԫ�ؼ�1
    if ~is_untranslated
        data(data ~= 0) = data(data ~= 0) + 1;
    end
elseif strcmp(ext, 'mat')
    file = load(path);
    % �ӽṹ���У���ȡ���ԣ����ݣ�
    if strcmp(mat_field, 'f1')
        data = file.f1;
    elseif strcmp(mat_field, 'imgout')
        data = file.imgout;
    elseif strcmp(mat_field, 'img')
        data = file.img;
    end
    disp(class(data));
    % mat���ݻ�ԭ����1000
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
% չʾ������Ƭ����
shape = size(data);
% i����ά�ȣ�j����ͬһά�Ȳ�ͬλ��
for i = 1:3
    % ��ͬά�ȵĳ���
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