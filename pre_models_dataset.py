import torchvision
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, MNIST
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image



transforms_train = {"imagenet100": transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 transforms.CenterCrop(224),
                                                 transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ]),
              "imagenet1k": transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 transforms.CenterCrop(224),
                                                 #transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ]),
              "imagenet_m": transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                transforms.CenterCrop(224),
                                                #transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                ]),
             "imagenet50": transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                           (0.229, 0.224, 0.225)),
                                                transforms.CenterCrop(224),
                                                #transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                ])
              }



transforms_test = {"imagenet100": transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 transforms.CenterCrop(224),
                                                 transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ]),
              "imagenet1k": transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 transforms.CenterCrop(224),
                                                 #transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ]),
              "imagenet_m": transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                transforms.CenterCrop(224),
                                                #transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                ]),
              "imagenet50": transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                transforms.CenterCrop(224),
                                                #transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                ]),
              "mnist": transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,)),
                                           transforms.Resize((32, 32), interpolation=transforms.functional.InterpolationMode.BILINEAR),]),
              "dtd": transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                         transforms.Resize((224,224), interpolation=transforms.functional.InterpolationMode.BILINEAR)]),
              }


class outlier_dataset(Dataset):

    def __init__(self, ori_dataset):

        self.ori_dataset = ori_dataset

    def __getitem__(self, idx):

        img, _ = self.ori_dataset[idx]
        return img, 1000

    def __len__(self):
        return len(self.ori_dataset)



def ImageNet100(data_path="../datasets/imagenet100", train=True, opt=None):

    if train:
        transform = transforms_train["imagenet100"]
    else:
        transform = transforms_test["imagenet100"]
        data_path = "../datasets/imagenet100" + "_test"
    imagenet100 = torchvision.datasets.ImageFolder(data_path, transform=transform)

    return imagenet100


def ImageNet50(data_path="../datasets/imagenet50", train=True, outliers=False, opt=None):

    if train:
        transform = transforms_train["imagenet50"]
    else:
        transform = transforms_test["imagenet50"]
        data_path = "../datasets/imagenet100" + "_test"
    imagenet50 = torchvision.datasets.ImageFolder(data_path, transform=transform)

    if outliers:
        imagenet50_outlier = outlier_dataset(imagenet50)
        return imagenet50_outlier

    return imagenet50



def ImageNet1k(data_path="../datasets/imagenet100", train=True, opt=None):

    if train:
        transform = transforms_train["imagenet1k"]
    else:
        transform = transforms_test["imagenet1k"]
        data_path = "../datasets/imagenet100" + "_test"
    imagenet1k = torchvision.datasets.ImageFolder(data_path, transform=transform)

    return imagenet1k


def ImageNet_M(data_path="../datasets/imagenet-M-train", train=True, opt=None):

    imagenet_m_class_list_base = ["n01728572", "n01728920",
                             "n01817953", "n01818515",
                             "n01978287", "n01978455",
                             "n02514041", "n01443537",
                             "n02066245", "n01484850",
                             "n02110063", "n02109047",
                             "n01664065", "n01665541",
                             "n02480495", "n02481823",
                             "n02123045", "n01622779"]

    imagenet_m_class_list_novel = ["n01729322", "n01729977",
                                   "n01819313", "n01820546",
                                   "n01980166", "n01981276",
                                   "n02607072", "n02643566",
                                   "n01491361", "n01494475",
                                   "n02089867", "n02102177",
                                   "n01667114", "n01667778",
                                   "n02480855", "n02486410",
                                   "n02124075", "n02123394"]

    imagenet_m_class_list_dog = ["n02091134", "n02092002",
                                 "n02110341", "n02089078",
                                 "n02086910", "n02093256",
                                 "n02113712", "n02105162",
                                 "n02091467", "n02106550",
                                 "n02104365", "n02086079",
                                 "n02090721", "n02108915",
                                 "n02107683", "n02085936",
                                 "n02088094", "n02085782"]

    if train:
        transform = transforms_train["imagenet_m"]
    else:
        transform = transforms_test["imagenet_m"]
    imagenet_m = torchvision.datasets.ImageFolder(data_path, transform=transform)

    return imagenet_m



class iCIFAR100(CIFAR100):

    def __init__(self, root,
                 train=True,
                 target_transform=None,
                 download=False,
                 label_dict=None,
                 outliers=False,
                 opt=None):
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        target_transform=target_transform,
                                        download=download)
        self.label_dict = label_dict

        if outliers:
            class_indices = [14, 18, 21, 24, 28, 31, 38, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 66, 67, 68, 70, 71, 72,
                             73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                             97, 98, 99]
            label_dict = {}
            for i, idx in enumerate(class_indices):
                label_dict[str(idx)] = 1000
        else:
            class_indices = [4, 30, 55, 1, 32, 54, 62, 9, 10, 16, 0, 51, 53, 22, 39, 5, 20, 25, 6, 7, 3, 42, 43, 12, 17, 37,
                           23, 33, 15, 19, 34, 63, 64, 26, 45, 2, 11, 35, 27, 29, 36, 50, 65, 47, 52, 8, 13, 48, 41, 69]
            label_dict = {}
            for i, idx in enumerate(class_indices):
                label_dict[str(idx)] = i

        self.transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
                                 ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
                                 ])

        #if "vit" in opt.model:
        #    self.transform_train.transforms.insert(0, transforms.Resize(224))
        #    self.transform_test.transforms.insert(0, transforms.Resize(224))

        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in class_indices:
                    train_data.append(self.data[i])
                    label = self.targets[i]
                    label = label_dict[str(label)]
                    train_labels.append(label)

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in class_indices:
                    test_data.append(self.data[i])
                    label = self.targets[i]
                    label = label_dict[str(label)]
                    test_labels.append(label)

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def get_part_data(self, xidxs):

        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels


"""
{"snake": ["n01728572", "n01728920", "n01729322", "n01729977", "n01734418", "n01735189", "n01737021", "n01739381", "n01740131", "n01742172"],
"Parrot": ["n01817953", "n01818515", "n01819313", "n01820546"],
"crab": ["n01978287", "n01978455", "n01980166", "n01981276", "n01986214"],
#"bird": ["n01828970", "n01833805", "n02007558"],       # "n01796340",  "n01829413", "n01843383",
"fish": ["n02514041", "n01443537", "n02607072", "n02643566", "n02640242"],  # "n01498041",
"lizard": ["n01688243", "n01689811", "n01692333", "n01693334", "n01629819"],
"shark": ["n02066245", "n01484850", "n01491361", "n01494475"],
"dog": ["n02110063", "n02109047", "n02089867", "n02102177", "n02091134", "n02092002",
"n02110341", "n02089078", "n02086910", "n02093256", "n02113712", "n02105162", "n02091467", "n02106550",
       "n02104365", "n02086079", "n02090721", "n02108915", "n02107683", "n02085936", "n02088094", "n02085782",
       "n02090622", "n02113624", "n02107142", "n02107574", "n02086240", "n02102973", "n02112018", "n02106030", "n02099601",
"n02106166", "n02088364", "n02100236", "n02099849", "n02110958", "n02099429", "n02094258", "n02099267", "n02112350", "n02109961",
       "n02101388", "n02113799", "n02101006", "n02093428", "n02105855", "n02111500", "n02099712", "n02111889", "n02107312","n02091032", "n02102318",
"n02100877", "n02102480", "n02086646", "n02116738", "n02091244", "n02089973", "n02106662"], # "n02117135",
"turtle": ["n01664065", "n01665541", "n01667114", "n01667778", "n01669191"],
"ape": ["n02480495", "n02481823", "n02480855", "n02486410"],
"cat": [ "n02123045", "n01622779", "n02124075", "n02123394", "n02123159", "n02123597"]}  #"n02123045","n02497673",
"""

def imagenet50_medium_outliers(data_path="../datasets/imagenet_medium_outliers"):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    transforms.CenterCrop(224),])
    imagenet50_medium_outliers = torchvision.datasets.ImageFolder(data_path, transform=transform, target_transform=lambda y : 1000)

    return imagenet50_medium_outliers


def cifar_medium_outliers(data_path="../datasets/cifar_medium_outliers"):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    transforms.Resize((32, 32), interpolation=transforms.functional.InterpolationMode.BILINEAR), ])
    cifar_medium_outliers = torchvision.datasets.ImageFolder(data_path, transform=transform, target_transform=lambda y : 1000)

    return cifar_medium_outliers


def DTD(data_path="../datasets/DTD"):

    transform = transforms_test["dtd"]
    dtd = torchvision.datasets.ImageFolder(data_path, transform=transform, target_transform=lambda y : 1000)

    return dtd



class mnist(MNIST):

    def __init__(self, root="../datasets",
                 classes=range(10),
                 train=True,
                 download=True,):
        super(mnist, self).__init__(root, train=train,
                                    download=download)

        self.transform = transforms_test["mnist"]
        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(0, len(self.data)):  # subsample
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(1000)

            self.traindata = torch.stack(train_data).numpy()
            self.trainlabels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(0, len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(1000)

            print(len(test_data))
            self.testdata = torch.stack(test_data).numpy()
            self.testlabels = test_labels


    def __getitem__(self, index):
        if self.train:
            img, target = self.traindata[index], self.trainlabels[index]
        else:
            img, target = self.testdata[index], self.testlabels[index]

        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.traindata)
        else:
            return len(self.testdata)



if __name__ == "__main__":

    dtd = DTD()
    print(len(dtd))