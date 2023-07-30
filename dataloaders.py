import numpy as np
from skimage.util.shape import view_as_windows
from itertools import product

import cv2

import os
from torch.utils.data import Dataset
from albumentations import (
    HorizontalFlip, VerticalFlip, 
    ShiftScaleRotate, OpticalDistortion, GridDistortion, ElasticTransform, 
    RandomBrightnessContrast, 
    IAASharpen, IAAEmboss, OneOf, Compose   
)
# https://albumentations.ai/docs/getting_started/setting_probabilities/


class ThebeGenerator(Dataset):
    """
    Dataset class for Thebe data.

    Attributes:
        root (str): Data path
        split (str): Choose set 'train' or 'val'
        red (int): Factor of reduction of training images
        n_imgs (int): Number of training images if 'red' was not defined
        aug (str): Augmentation options

    Methods:
        __getitem__(index): Get the filename by index
        __len__(): Get the total number of samples in the dataset
        augmentation: Make the data augmentation
    """

    def __init__(self, root, split, red=1, n_imgs=32, aug=False):
        
        np.random.seed(42)
        
        self.split = split
        self.rootPath = root
        self.list_IDs = os.listdir(os.path.join(self.rootPath, self.split, 'seismic'))
        self.aug = aug
        self.domain = root.split('_')[1]
        
        if red > 0 :
            n_imgs = len(self.list_IDs)//red
            
        np.random.shuffle(self.list_IDs)
        self.list_IDs = self.list_IDs[:n_imgs]
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Get the filename by index

        Parameters:
            index (string): filename

        Returns:
            X (numpy.array): image input (1, ps, ps)
            y (numpy.array): target input (ps, ps)
        """
        ID = self.list_IDs[index]
        
        X = np.load(os.path.join(self.rootPath, self.split, 'seismic', ID))

        if self.domain == 'tgt': # since there is no labels, it loads the same input
            y = np.load(os.path.join(self.rootPath, self.split, 'seismic', ID ))
        else:
            y = np.load(os.path.join(self.rootPath, self.split, 'faults', ID )).astype(np.float32)
        
        if self.aug:
            X, y = self.augmentation(X,y)
        
        X = np.expand_dims(X, axis=0)

        return X, y
    
    def augmentation(self, X, y):
        
        if self.aug == "elastic":
            aug = ElasticTransform(p=0.5)
            
        if self.aug == "flipv":
            aug = VerticalFlip(p=0.5)
            
        if self.aug == "type0":
            aug = OneOf([
                Compose([VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]),
            ], p=0.9)

        if self.aug == "type1":
            aug = OneOf([
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
                IAASharpen(p=1),
                IAAEmboss(p=1),
                RandomBrightnessContrast(p=1),
                VerticalFlip(p=1),
                Compose([VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]),
                ElasticTransform(p=1, alpha=400, sigma=400 * 0.05, alpha_affine=400 * 0.03),
                GridDistortion(p=1),
                OpticalDistortion(p=1)
            ], p=0.9)

        augmented = aug(image=X, mask=y)
        return augmented['image'], augmented['mask']


class FaultSeg2DGenerator(Dataset):
    """
    Dataset class for FaultSeg data.

    Attributes:
        root (str): Data path
        split (str): Choose set 'train' or 'val'
        red (int): Factor of reduction of training images
        n_imgs (int): Number of training images if 'red' was not defined
        aug (str): Augmentation options

    Methods:
        __getitem__(index): Get a data sample and its label by index
        __len__(): Get the total number of samples in the dataset
        augmentation: Make the data augmentation
    """
    
    def __init__(self, root, split, red=1, n_imgs=32, aug=False):

        np.random.seed(42)
        
        self.split = split
        self.rootPath = root
        self.list_IDs = os.listdir(os.path.join(self.rootPath, self.split, 'seis'))
        self.aug = aug
        self.dim = (128,128,128)
        
        self.list_X = []
        self.list_y = []
        for ID in self.list_IDs:
            X = np.fromfile(os.path.join(self.rootPath, self.split, 'seis', ID), dtype=np.single)
            y = np.fromfile(os.path.join(self.rootPath, self.split, 'fault', ID), dtype=np.single)
            X = np.reshape(X, self.dim)
            y = np.reshape(y, self.dim).astype(np.float32)
            
            X = (X-np.mean(X))/np.std(X)
            
            for i in range(128):
                self.list_X.append(X[i].T)
                self.list_y.append(y[i].T)
            # for x in range(128):
            #     self.list_X.append(X[:,x].T)
            #     self.list_y.append(y[:,x].T)
                
        self.list_IDs = [x for x in range(len(self.list_X))]
        
        if red > 0 :
            n_imgs = len(self.list_IDs)//red
            
        np.random.shuffle(self.list_IDs)
        self.list_IDs = self.list_IDs[:n_imgs]     
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Get the images by index

        Parameters:
            index (int): random position

        Returns:
            X (numpy.array): image input (1, ps, ps)
            y (numpy.array): target input (ps, ps)
        """
        ID = self.list_IDs[index]
        X = self.list_X[ID]
        y = self.list_y[ID]
        
        if self.aug:
            X, y = self.augmentation(X,y)
        
        X = np.expand_dims(X, axis=0)

        return X, y
    
    def augmentation(self, X, y):
        
        if self.aug == "type0":
            aug = OneOf([
                Compose([VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]),
            ], p=0.9)
        
        if self.aug == "type1":
            aug = OneOf([
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
                IAASharpen(p=1),
                IAAEmboss(p=1),
                RandomBrightnessContrast(p=1),
                VerticalFlip(p=1),
                Compose([VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]),
                ElasticTransform(p=1, alpha=400, sigma=400 * 0.05, alpha_affine=400 * 0.03),
                GridDistortion(p=1),
                OpticalDistortion(p=1)
            ], p=0.9)
            
        augmented = aug(image=X, mask=y)
        return augmented['image'], augmented['mask']


class F3Generator(Dataset):
    """
    Dataset class for Thebe data.

    Attributes:
        root (str): Data path
        red (int): Factor of reduction of training images
        n_imgs (int): Number of training images if 'red' was not defined
        aug (str): Augmentation options

    Methods:
        __getitem__(index): Get the filename by index
        __len__(): Get the total number of samples in the dataset
        augmentation: Make the data augmentation
    """

    def __init__(self, root, red=1, n_imgs=32, aug=False):
        
        np.random.seed(42)
        
        self.rootPath = root
        self.list_IDs = os.listdir(os.path.join(self.rootPath, 'seismic'))
        self.aug = aug
        
        if red > 0 :
            n_imgs = len(self.list_IDs)//red
            
        np.random.shuffle(self.list_IDs)
        self.list_IDs = self.list_IDs[:n_imgs]
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Get the filename by index

        Parameters:
            index (string): filename

        Returns:
            X (numpy.array): image input (1, ps, ps)
        """

        ID = self.list_IDs[index]
        
        X = np.load(os.path.join(self.rootPath, 'seismic', ID))
        
        if self.aug:
            X = self.augmentation(X)
        
        X = np.expand_dims(X, axis=0)

        return X
    
    def augmentation(self, X):
        
        if self.aug == "type0":
            aug = OneOf([
                Compose([VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]),
            ], p=0.9)
        
        if self.aug == "type1":
            aug = OneOf([
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
                IAASharpen(p=1),
                IAAEmboss(p=1),
                RandomBrightnessContrast(p=1),
                VerticalFlip(p=1),
                Compose([VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]),
                ElasticTransform(p=1, alpha=400, sigma=400 * 0.05, alpha_affine=400 * 0.03),
                GridDistortion(p=1),
                OpticalDistortion(p=1)
            ], p=0.9)

        augmented = aug(image=X)
        return augmented['image']


class PatchGenerator:
    """
    Class that handle patches from an image or section.

    Attributes:
        dims (tuple): Dimensions (height,  width)
        ps (int): Patch size
        stride (int): Stride size
        phase (str): 'train' or 'test'

    Methods:
        get_pad_values_test(dim): Function to get the pad values for testing.
        get_patches(img): Function to generate patches from section or image.
        reconstruct_image(patches): Function to reconstruct image or section from patches.
    """

    def __init__(self, dims, ps, stride, phase='test'):
        self.ps = ps
        self.stride = stride
        self.h, self.w = dims
        self.overlap = ps - stride

        if phase == 'test':
            self.nx, self.left_pad, self.right_pad = self.get_pad_values_test(self.w)
            self.ny, self.top_pad, self.bottom_pad = self.get_pad_values_test(self.h)
        elif phase == 'train':
            self.nx, self.left_pad, self.right_pad = self.get_pad_values_train(self.w)
            self.ny, self.top_pad, self.bottom_pad = self.get_pad_values_train(self.h)
        
        # self.values = self.nx, self.left_pad, self.right_pad, self.ny, self.top_pad, self.bottom_pad
        print("nx, left_pad, right_pad, ny, top_pad, bottom_pad")
        print(self.nx, self.left_pad, self.right_pad, self.ny, self.top_pad, self.bottom_pad)

    def get_pad_values_train(self, dim):
        """
        Function to get the pad values for training.

        Parameters
            dim (int): dimension (height or width)
        """
        
        if dim > self.ps:
            n_patches = int(np.ceil((dim - self.overlap)/self.stride))
        else:
            n_patches = 1

        new_dim = self.stride*(n_patches) + self.overlap
        pad1 = (new_dim-dim)//2
        pad2 = new_dim-dim-pad1
        
        return n_patches, pad1, pad2

    
    def get_pad_values_test(self, dim):
        """
        Function to get the pad values for testing.

        Parameters
            dim (int): dimension (height or width)
        """

        if dim > self.ps:
            n_patches = int(np.ceil((dim) / self.stride))
        else:
            n_patches = 1

        new_dim = self.stride * (n_patches) + 2 * self.overlap
        pad1 = (new_dim - dim) // 2
        pad2 = new_dim - dim - pad1

        return n_patches + 1, pad1, pad2

    def get_patches(self, img):
        """
        Function to generate patches from section or image.

        Parameters
        img (numpy.array): 2D image or section
        """

        arr = np.pad(img, ((self.top_pad, self.bottom_pad), (self.left_pad, self.right_pad)), "reflect")
        patches = view_as_windows(arr, (self.ps, self.ps), step=self.stride)
        patches = patches.reshape((self.nx * self.ny, self.ps, self.ps))

        return patches
    
    def reconstruct_image(self, patches):
        """
        Function to reconstruct image or section from patches.

        Parameters
        ----------
        patches (numpy.array): Shape of array (ny, nx, ps, ps, 1)
        """

        image = np.zeros((self.h + self.top_pad + self.bottom_pad,
                          self.w + self.left_pad + self.right_pad, 1),
                          dtype=patches.dtype)
        
        for i, j in product(range(self.ny), range(self.nx)):
            patch = patches[i,j]
            image[(i*self.stride):(i*self.stride + self.ps), 
                  (j*self.stride):(j*self.stride + self.ps)] += patch

        recover = image / 4  # each pixel is repeated four times
        return recover[self.top_pad:self.top_pad+self.h, self.left_pad:self.left_pad+self.w]


class DefaultGeneratorTest(Dataset):
    """
    Class to load real datasets for testing.

    Attributes:
        patches (numpy.array): Dimensions (height, width)

    Methods:
        __getitem__(index): Get the patch by index
        __len__(): Get the total number of samples in the dataset
    """

    def __init__(self, patches):
        self.images = patches

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image
    

class ResizeGeneratorTest(Dataset):

    def __init__(self, patches):
        self.images = patches

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        image = cv2.resize(image, (128,128))
        image = np.expand_dims(image, 0)
        
        return image