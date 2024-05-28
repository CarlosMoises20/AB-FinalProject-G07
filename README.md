# AB-FinalProject-G07

This repository contains the code developed to implement our final project on Algorithms in Bioinformatics.

The project consists on detecting pneumonia through deep learning models.


chest_xray -> folder with images used to train, test and validate the implemented model

The folder contains 3 subfolders: 
    test: contains the images used to test the model
    train: contains the images used to train the model
    val: contains the images used to validate the model

Those subfolders contain, each one, two sub-subfolders:

    NORMAL: images which indicate that the person has no problems

    PNEUNOMIA: images which indicate that the person has pneumonia. In this case, the person can have a virus or bacteria. To determine that, this folder has images that contain, on its name, whether the word 'virus', indicating that the person has virus, or the name 'bacteria', indicating that the person has a bacteria.


dataLoader.py -> file that stores a class that loads the data to be used by the model developed in model.py

model.py -> the model to be developed

main.py -> file that must be executed to test the implementation

Run the following command to test the implementation:
```python
python ./main.py
```