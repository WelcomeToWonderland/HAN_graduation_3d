from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train
    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        """
        loader_train
        将多个dataset拼接，使用一个dataloader加载

        loader_test
        一个dataset，对应一个dataloader

        :param args:
        """
        print('Making Dataloader...')
        self.loader_train = None
        if not args.test_only:
            # 按照数据集名称，加载对应py文件，针对性建立dataset
            # 设立不同数据集对应py文件：不同数据集的文件夹结构不同
            datasets = []
            for d in args.data_train:
                """
                DIV2K-Q是DIV2K的子集
                """
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'

                # oabreast数据集
                if module_name in ['Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left',
                                   'Neg_07_Left_train', 'Neg_35_Left_train', 'Neg_47_Left_train',
                                   'Neg_07_Left_test', 'Neg_35_Left_test', 'Neg_47_Left_test']:
                    if args.is_3d :
                        m = import_module('data.oabreast_3d')
                    else:
                        m = import_module('data.oabreast')
                    datasets.append(getattr(m, 'OABreast')(args, name=d))
                # 其他数据集
                else:
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name=d))



            # 为dataset，建立dataloader
            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )


        self.loader_test = []
        for d in args.data_test:
            if d in ['Val20', 'Set20', 'Set5', 'Set14', 'B100', 'Urban100','Manga109']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            # oabreast修改
            elif d in ['Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left',
                       'Neg_07_Left_train', 'Neg_35_Left_train', 'Neg_47_Left_train',
                       'Neg_07_Left_test', 'Neg_35_Left_test', 'Neg_47_Left_test']:

                if args.is_3d:
                    m = import_module('data.oabreast_3d')
                else:
                    m = import_module('data.oabreast')
                testset = getattr(m, 'OABreast')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
