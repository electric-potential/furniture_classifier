import glob
from matplotlib import pyplot as plt
import os
from torch.utils.data import Dataset
import pandas as pd
import csv
from torchvision import transforms
from torchvision.io import read_image
from torch.nn.functional import pad

def filename_to_annotation(fname):
    """
    Helper function for generate_annotations_csv.
    """
    if fname.lower().startswith("bed"):
        return 0
    elif fname.lower().startswith("chair"):
        return 1
    elif fname.lower().startswith("sofa"):
        return 2
    else:
        raise Exception("All filenames must have one annotation.")

def generate_annotations_csv(path, train_fname, test_fname, test_size=0.1, dirs=["Bed/*", "Chair/*", "Sofa/*"], annotation_func=filename_to_annotation):
    """
    First step in dataset creation pipeline.
    If you have raw data, start here. Generates
    training and test set csv files used in custom
    dataset class later on.
    """
    train = []
    test = []
    for furniture in dirs:
        p = os.path.join(path, furniture)
        size = len(glob.glob(p))
        for count, file in enumerate(glob.glob(p)):
            fname = file.split("/")[-1]
            idx = annotation_func(fname)
            data = {"filename": fname, "annotation": idx}
            if count%int(size*test_size) == 0:
                test.append(data)
            else:
                train.append(data)
    fields = ["filename", "annotation"]
    with open(train_fname, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for data in train:
            writer.writerow(data)
    with open(test_fname, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for data in test:
            writer.writerow(data)
    return

class FurnitureTransform(object):
    def __init__(self, out_size, norm=True, crop=True, crop_pixels=25):
        self.out_size = out_size
        self.norm = norm
        self.crop = crop
        self.crop_pixels = crop_pixels

    def normalize(self, img):
        """
        Applies a standard image channel normalization, making
        it easier to traverse the loss landscape.
        """
        return (img/255)-0.5

    def padding(self, img):
        """
        Pad the side of the image that is smaller with whitespace.
        This is required to resize the image and maintain a proper
        aspect ratio.
        """
        if img.shape[1] != img.shape[2]:
            m = max(img.shape[1], img.shape[2])
            if m == img.shape[1]:
                img = pad(img, ((img.shape[1]-img.shape[2])//2,(img.shape[1]-img.shape[2])//2 + (img.shape[1]-img.shape[2])%2,0,0,0,0), value=0.5 if self.norm else 255)
            else:
                img = pad(img, (0,0,(img.shape[2]-img.shape[1])//2,(img.shape[2]-img.shape[1])//2 + (img.shape[2]-img.shape[1])%2,0,0), value=0.5 if self.norm else 255)
        return img

    def resize(self, img):
        """
        Resizes the image maintaining square aspect ratio.
        """
        if self.crop:
            resize = transforms.Resize((self.out_size+self.crop_pixels, self.out_size+self.crop_pixels))
        else:
            resize = transforms.Resize((self.out_size, self.out_size))
        return resize.forward(img)

    def cropped(self, img, corner):
        """
        Crops the image into 4 different corners, effectively
        multiplying the size of the dataset by 4.
        """
        if corner%4 == 0:
            return img[:,:self.out_size,:self.out_size]
        elif corner%4 == 1:
            return img[:,self.crop_pixels:,:self.out_size]
        elif corner%4 == 2:
            return img[:,self.crop_pixels:,self.crop_pixels:]
        else:
            return img[:,:self.out_size,self.crop_pixels:]

    def __call__(self, img, corner=0):
        if self.norm:
            img = self.normalize(img)
        img = self.padding(img)
        img = self.resize(img)
        if self.crop:
            img = self.cropped(img, corner)
        return img
        

class FurnitureDataset(Dataset):
    """
    Standard furniture dataset to be used in training.
    A transform can be passed in to perform data augmentation
    and pre-processing. Depending on the model, the size of
    each image might need to be augmented (this is handled
    automatically in the FurnitureTransform class).
    """
    def __init__(self, labels_file, data_dir, trans=FurnitureTransform(256)):
        self.labels = pd.read_csv(labels_file)
        self.data_dir = data_dir
        self.trans = trans

    def __len__(self):
        return len(self.labels) if self.trans is None or not self.trans.crop else len(self.labels)*4 # cropping corners augmentation

    def show(self, idx):
        """
        displays a member of the dataset.
        WARNING: This method is platform dependant, it
        will most likely not work in a jupyter notebook.
        """
        out = self[idx][0].permute(1,2,0)
        out += 0.5 if self.trans is not None and self.trans.norm else 0
        plt.imshow(out)
        plt.show()

    def __getitem__(self, idx):
        if self.trans is None or not self.trans.crop:
            data_path = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
            image = read_image(data_path)
            if self.trans is not None:
                image = self.trans(image)
            label = self.labels.iloc[idx, 1]
        else:
            data_path = os.path.join(self.data_dir, self.labels.iloc[idx//4, 0])
            image = read_image(data_path)
            image = self.trans(image, idx%4)
            label = self.labels.iloc[idx//4, 1]
        return image, label

