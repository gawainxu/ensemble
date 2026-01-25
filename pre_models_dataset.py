import torchvision
import torchvision.transforms as transforms

transforms_train = {"imagenet100": transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 transforms.CenterCrop(224),
                                                 transforms.Resize(256, interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ]),
              "imagenet1k": transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 transforms.CenterCrop(224),
                                                 transforms.Resize(256,
                                                 interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ]),
              "imagenet_m": transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                transforms.CenterCrop(224),
                                                transforms.Resize(256,
                                                                  interpolation=transforms.functional.InterpolationMode.BILINEAR),
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
                                                 transforms.Resize(256,
                                                 interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                 ]),
              "imagenet_m": transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                transforms.CenterCrop(224),
                                                transforms.Resize(256,
                                                                  interpolation=transforms.functional.InterpolationMode.BILINEAR),
                                                ])
              }



def ImageNet100(data_path="../datasets/ImageNet100", train=True):

    if train:
        transform = transforms_train["imagenet100"]
    else:
        transform = transforms_test["imagenet100"]
    imagenet100 = torchvision.datasets.ImageFolder(data_path, transform=transform)

    return imagenet100


def ImageNet1k(data_path="../datasets/ImageNet100", train=True):

    if train:
        transform = transforms_train["imagenet1k"]
    else:
        transform = transforms_test["imagenet1k"]
    imagenet1k = torchvision.datasets.ImageFolder(data_path, transform=transform)

    return imagenet1k


def ImageNet_M(data_path="../datasets/ImageNet-M-train", train=True):

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

    if train:
        transform = transforms_train["imagenet_m"]
    else:
        transform = transforms_test["imagenet_m"]
    imagenet_m = torchvision.datasets.ImageFolder(data_path, transform=transform)

    return imagenet_m




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


