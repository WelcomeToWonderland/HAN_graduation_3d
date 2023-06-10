temp = 'D:\workspace\dataset\OABreast\clipping\Neg_07_Left\HR';
temp1 = 'D:\workspace\dataset\OABreast\original\Neg_07_Left';
temp2 = 'D:\workspace\dataset\OABreast\dat2mat';

% original
% is_original = true;
% is_unmodified = true;
% prefix_path = 'D:\workspace\dataset\OABreast\original';
% suffix_path = 'MergedPhantom.DAT';

% dat2mat
% is_untranslated = true;
% is_unclipping = true;
% prefix_path = 'D:\workspace\dataset\OABreast\dat2mat';
% suffix_path = 'HR';
% mat_field = 'img';

% clipping
% is_untranslated = true;
% is_unclipping = true;
% prefix_path = 'D:\workspace\dataset\OABreast\dat2mat\clipping';
% suffix_path = 'HR';
% mat_field = 'img';

% translated
% is_untranslated = false;
% is_unclipping = true;
% prefix_path = 'D:\workspace\dataset\OABreast\dat2mat\clipping\pixel_translation';
% suffix_path = 'SR\X2';
% mat_field = 'img';
% 
% reset = true;
% basenames = {'Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left', 'Neg_07_Left_train', 'Neg_07_Left_test'};
% for idx = 1:5
%     basename = basenames(idx);
%     basename = char(basename);
% %     folder_save = basename;
%     folder_save = strcat(basename, '_SR');
% %     path = fullfile(prefix_path, basename, suffix_path);
%     filename = strcat(basename, '.mat');
%     path = fullfile(prefix_path, basename, suffix_path, filename);
%     show_3d(path, folder_save, is_untranslated, is_unclipping, basename, reset, mat_field);
% end

% 单测
% mat_field = 'img';
path = 'D:\workspace\dataset\USCT\clipping\pixel_translation\3d\HR\20220517T112745.mat';
folder_save = 'HR_test_untranslated_true';
is_untranslated = true;
is_unclipping = false;
basename = '20220517T112745';
reset = true;
mat_field = 'f1';
show_3d(path, folder_save, basename, mat_field, is_untranslated, is_unclipping, reset);

% % 获取数据集的HR、LR和SR
% is_untranslated = false;
% basename = 'Neg_07_Left_test_remove_zero';
% mat_field = 'img';
% % 处理HR、LR以及通过bicubic获得的SR
% path_dataset = 'D:\workspace\dataset\OABreast\dat2mat\Neg_07_Left_test_remove_zero';
% resolutions = {'HR', 'LR\X2'};
% savefolder_suffixes = {'HR', 'LR'};
% savefolder_prefix = 'Neg_07_Left_test_correct_remove_zero';
% filename = 'Neg_07_Left_test.mat';
% for idx = 1:2
%     % 保存文件夹
%     savefolder_suffix = savefolder_suffixes(idx);
%     savefolder_suffix = char(savefolder_suffix);
%     folder_save = strcat(savefolder_prefix, '_', savefolder_suffix);
%     % 文件路径
%     resolution = resolutions(idx);
%     resolution = char(resolution);
%     path = fullfile(path_dataset, resolution, filename);
%     % 调用函数
%     show_3d(path, folder_save, basename, mat_field, is_untranslated);
% end
% % bicubic SR
% folder_save = 'Neg_07_Left_test_correct_remove_zero_bicubic_SR';
% path = 'D:\workspace\dataset\OABreast\dat2mat\Neg_07_Left_test_remove_zero\SR\X2\Neg_07_Left_test.mat';
% show_3d(path, folder_save, basename, mat_field, is_untranslated);
% % HAN SR 
% folder_save = 'Neg_07_Left_test_correct_remove_zero_HAN_SR';
% path = 'D:\workspace\dataset\Neg_07_Left_test_remove_zero_x2_SR.mat';
% show_3d(path, folder_save, basename, mat_field, is_untranslated);






