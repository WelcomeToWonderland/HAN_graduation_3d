    prefix_original = 'D:\workspace\dataset\OABreast\dat2mat';
    suffix_original = 'HR';
    prefix_clipping = 'D:\workspace\dataset\OABreast\dat2mat\clipping';
    suffix_clipping = 'HR';
    basenames = {'Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left', 'Neg_07_Left_train', 'Neg_07_Left_test'};
    for idx = 1:3
        basename = basenames(idx);
        basename = char(basename);
        filename = strcat(basename, '.mat');
        path_original = fullfile(prefix_original, basename, suffix_original, filename);
        path_cipping = fullfile(prefix_clipping, basename, suffix_clipping, filename);
        clipping(path_original, path_cipping)
    end
