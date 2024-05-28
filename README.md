# AB-FinalProject-G07

This repository contains the code developed to implement our final project on Algorithms in Bioinformatics.

The project consists on detecting pneumonia through deep learning models.



chest_xray -> folder with images used to train, test and validate the implemented model

The folder contains 3 subfolders: 
    test: contains the images used to test the model
    train: contains the images used to train the model
    val: contains the images used to validate the model

Those subfolders contain, each one, two sub-subfolders:
    NORMAL: images which indicate the person has no problems
    PNEUNOMIA: images which indicate the person has pneumonia. In this case, the person can have a virus or bacteria. To determine that, this folder has images that contain, on its name, whether the word 'virus', indicating that the person has virus, or the name 'bacteria', indicating that the person has a bacteria.


main.py -> file that must be executed


model.py -> the model to be developed

pneumoniaDetector.py -> file that stores a class that implements a image classification algorithm