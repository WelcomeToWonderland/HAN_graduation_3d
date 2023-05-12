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

% % translated
% is_untranslated = false;
% is_unclipping = true;
% prefix_path = 'D:\workspace\dataset\OABreast\dat2mat\clipping\pixel_translation';
% suffix_path = 'LR\X2';
% mat_field = 'img';
% 
% reset = true;
% basenames = {'Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left', 'Neg_07_Left_train', 'Neg_07_Left_test'};
% for idx = 1:5
%     basename = basenames(idx);
%     basename = char(basename);
% %     folder_save = basename;
%     folder_save = strcat(basename, '_LR');
% %     path = fullfile(prefix_path, basename, suffix_path);
%     filename = strcat(basename, '.mat');
%     path = fullfile(prefix_path, basename, suffix_path, filename);
%     show_3d(path, folder_save, is_untranslated, is_unclipping, basename, reset, mat_field);
% end

% ����
mat_field = 'img';
path = 'D:\workspace\dataset\oabreast_2d_correct_new_start\results-Neg_07_Left_test\Neg_07_Left_test_x2_SR.mat';
folder_save = 'Neg_07_Left_test_HAN_new_start_SR';
is_untranslated = false;
is_unclipping = true;
basename = 'Neg_07_Left_test';
reset = true;
mat_field = 'f1';
show_3d(path, folder_save, is_untranslated, is_unclipping, basename, reset, mat_field);



