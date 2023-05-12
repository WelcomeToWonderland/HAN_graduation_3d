    prefix_dat = 'D:\workspace\dataset\OABreast\original';
    suffix_dat = 'MergedPhantom.DAT';
    prefix_mat = 'D:\workspace\dataset\OABreast\dat2mat';
    suffix_mat = 'HR';
    basenames = {'Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left', 'Neg_07_Left_train', 'Neg_07_Left_test'};
    for idx = 1:3
        basename = basenames(idx);
        basename = char(basename);
        filename = strcat(basename, '.mat');
        path_dat = fullfile(prefix_dat, basename, suffix_dat);
        path_mat = fullfile(prefix_mat, basename, suffix_mat, filename);
        dat2mat(path_dat, path_mat)
    end
