# Description

This small project compares the performance of the Keras classical augmentation algorithms and some configurations of the albumentations library. 

Performance is tested in two different jupyter notebooks:

- samples/FileDataGenerator_test --> compares the execution time to augment all the dataset.
- samples/Neural_Network_test --> compares the performance of both approaches on the same dataset. Albumentations library results are quite promising (check the loss function graph and the execution time).


For this purpose, I've developed my own data generator, that read images, not from a directory but from a list of images and their corresponding labels. The FileDataGen class (at FileDataGenerator.py) considers both, Keras classical augmentation algorithm and the albumentations library. 

New albumentations configurations can be implemented by modifying the augmentation_clss (at augment.py).


The albumentations library has been developed by Buslaev Alexander, Alexander Parinov, Vladimir Iglovikov. 
- Doc at https://albumentations.readthedocs.io
- Github at https://github.com/albu/albumentations 

<br /><br /><br />
Reference:<br /><br />
@misc{Freire18_aug,<br />
  author = {Freire-Obre\'on, D.},<br />
  title = {Augmentation Project},<br />
  year = {2018},<br />
  publisher = {GitHub},<br />
  journal = {GitHub repository},<br />
  howpublished = {\url{https://github.com/davidfreire/Augmentation_project}},<br />
  doi= {10.5281/zenodo.1438467}<br />
}
