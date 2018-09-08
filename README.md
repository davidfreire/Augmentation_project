# Description

This small project compares the performance of the Keras classical augmentation algorithm and some configurations of the albumentation library. 

Performance is tested in two different jupyter notebooks:

- samples/FileDataGenerator_test --> compares the execution time to augment all the dataset.
- samples/Neural_Network_test --> compares the performance of both approaches on the same dataset. Albumentation library results are quite promising (check the loss function graph and the execution time).


For this purpose, I've developed my own data generator, that read images, not from a directory but from a list of images and their labels. The FileDataGen class (at FileDataGenerator.py) considers both, Keras classical augmentation algorithm and the albumentation approach. 

New albumentation configurations can be implemented modifying the augmentation_clss (at augment.py).


The albumentation library has been developed by Buslaev Alexander, Alexander Parinov, Vladimir Iglovikov. 

- Doc at https://albumentations.readthedocs.io
- Github at https://github.com/albu/albumentations 

