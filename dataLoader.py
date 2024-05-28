
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Algoritmo de classificação de imagens utilizado para desenvolver o modelo
class DataLoader():

    """
    data_dir -> the data root path
    train_dir -> the directory for the images used to train the model
    test_dir -> the directory for the images used to test the model
    val_dir -> the directory for the images used to validate the model
    classes -> labels to classify images
                - all images on NORMAL folders are classified as NORMAL
                - all images on PNEUMONIA folders are classified as PNEUMONIA

    """
    def __init__(self, train_dir = './chest_xray/train', 
                 test_dir = './chest_xray/test', 
                 val_dir = './chest_xray/val'):
        
        self.__test_dir = test_dir
        self.__train_dir = train_dir
        self.__val_dir = val_dir


    def load_data(self):

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        """
        Loads the datasets with ImageFolder
        ImageFolder loads the data considering the folder structure, i.e. that each subfolder corresponds to a class

        Example; considering that 'root' is the root path for data, and there are 2 classes 'dog' and 'cat':
            root/dog/xxx.png
            root/dog/xxy.png
            root/dog/[...]/xxz.png

            root/cat/123.png
            root/cat/nsdf3.png
            root/cat/[...]/asd932_.png
        """
        trainset = datasets.ImageFolder(root=self.__train_dir, transform=transform)
        valset = datasets.ImageFolder(root=self.__val_dir, transform=transform)
        testset = datasets.ImageFolder(root=self.__test_dir, transform=transform)


        """
        Create DataLoaders for each dataset, to load the data in batches
        in training, shuffle = True, because we want to shuffle the data in order to avoid overfitting
        and get better results
        but in testing and validation, shuffle = False, because we don't have to necessarily shuffle the data
        
        """
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
        valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

        return trainloader, valloader, testloader
    


