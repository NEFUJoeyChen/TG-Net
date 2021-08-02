import sys
import torch.utils.data as data
from os import listdir
from utils.tools import normalize
import os
from PIL import Image



class Dataset(data.Dataset):
    def __init__(self, transform=None):
        super(Dataset, self).__init__()
        # dataset_type:
        #     ['train', 'test']
        self.transform = transform
        self.samples = list()
        # self.data_path = data_path
        # self.dataset_type = dataset_type
        # self.image_shape = image_shape[:-1]  # [128, 128]
        # self.random_crop = random_crop
        # self.return_name = return_name
        f = open('./train.txt')
        lines = f.readlines()
        for line in lines:
            self.samples.append(line.strip())
        f.close()

    def __getitem__(self, index):
        item = self.samples[index]
        # img = cv2.imread(item.split(' _')[0])
        img = Image.open(item.split(' ')[0])
        if self.transform is not None:
            img = self.transform(img)
            img = normalize(img)
        label = int(item.split(' ')[-1])

        return img, label

    # def _find_samples_in_subfolders(self, dir):
    #     """
    #     Finds the class folders in a dataset.
    #     Args:
    #         dir (string): Root directory path.
    #     Returns:
    #         tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    #     Ensures:
    #         No class is a subdirectory of another.
    #     """
    #     if sys.version_info >= (3, 5):
    #         # Faster and available in Python 3.5 and above
    #         classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    #     else:
    #         classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    #     classes.sort()
    #     class_to_idx = {classes[i]: i for i in range(len(classes))}
    #     samples = []
    #     for target in sorted(class_to_idx.keys()):
    #         d = os.path.join(dir, target)
    #         if not os.path.isdir(d):
    #             continue
    #         for root, _, fnames in sorted(os.walk(d)):
    #             for fname in sorted(fnames):
    #                 if is_image_file(fname):
    #                     path = os.path.join(root, fname)
    #                     # item = (path, class_to_idx[target])
    #                     # samples.append(item)
    #                     samples.append(path)
    #     return samples

    def __len__(self):
        return len(self.samples)


