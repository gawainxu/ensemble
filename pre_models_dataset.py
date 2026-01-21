import torchvision
import torchvision.transforms as transforms

transforms = {"imagenet100": transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 transforms.CenterCrop(224),
                                                 transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ]),
              "imagenet1k": transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 transforms.CenterCrop(224),
                                                 transforms.Resize(256,
                                                 interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ])}


def ImageNet100(data_path):

    transform = transforms["imagenet100"]
    imagenet100 = torchvision.datasets.ImageFolder(data_path, transform=transform)

    return imagenet100


def ImageNet1k(data_path):
    transform = transforms["imagenet1k"]
    imagenet1k = torchvision.datasets.ImageFolder(data_path, transform=transform)

    return imagenet1k