
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Load the used dataset to develop the deep learning model
class DataLoad:

    """
    train_dir -> the directory for the images used to train the model
    test_dir -> the directory for the images used to test the model
    val_dir -> the directory for the images used to validate the model
    classes -> labels to classify images
                - all images on NORMAL folders are classified as NORMAL
                - all images on PNEUMONIA folders are classified as PNEUMONIA

    """
    def __init__(self):
        self.__test_dir = './chest_xray/train'
        self.__train_dir = './chest_xray/test'
        self.__val_dir = './chest_xray/val'


    def load_data(self):

        """
        Define the transform used for image pre-processing
        The transform uses a sequence of transforms simultaneously applied to each image before being fed to the
        deep learning model

        transforms.Compose() -> combines several transforms into a single specific sequence. Each image will have
                    the following transformations:
        
            - transforms.ToTensor() -> transforms the image in the range [0, 255] to
                a PyTorch tensor (torch.FloatTensor) of shape (C x H x W) in the range [0.0, 1.0]

            - transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
                -> normalizes each channel of the input and the image pixels values.
                -> on this case, all three channels are normalized with mean of 0.5 and standard deviation of 0.5
                    that rescales the pixels values to be on the range [-1, 1]
        
        """
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


        transform -> the pre-processing that will be applied to the images (transform.Compose)
            
        """
        trainset = datasets.ImageFolder(root=self.__train_dir, transform=transform)
        valset = datasets.ImageFolder(root=self.__val_dir, transform=transform)
        testset = datasets.ImageFolder(root=self.__test_dir, transform=transform)


        """
        Create DataLoaders for each dataset, to load the data in batches
        in training, shuffle = True, because we want to shuffle the data in order to avoid overfitting
        and get better results
        but in testing and validation, shuffle = False, because we don't have to necessarily shuffle the data

        num_workers is the number of subprocesses to use for data loading.
        
        """
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
        valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

        return trainloader, testloader, valloader
    

"""
Auxiliar code to test this class through output analysis


dataload = DataLoad()

train, test, val = dataload.load_data()

print("Val attributes:")
for key, value in val.__dict__.items():
    print(f"{key}: {value}")


"""