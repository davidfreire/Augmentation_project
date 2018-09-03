import numpy as np
import keras
from PIL import Image as pil_image
from augmentations import augmentation_clss #pip install albumentations
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
        
        

class ImgListDataGen(keras.utils.Sequence):
    
    # Initialization
    def __init__(self, img_files, labels, batch_size=32, target_size=None, 
                 n_classes=10, class_mode='categorical', shuffle=True, rescale=1., aug_mode=None):
        self.target_size = target_size
        self.batch_size = batch_size
        self.labels = labels
        self.img_files = img_files
        self.n_classes = n_classes
        if class_mode not in {'categorical', 'binary', 'sparse', 'input', None}:
            raise ValueError('Invalid class_mode:', class_mode, '; expected one of "categorical", '
                             '"binary", "sparse", "input" or None.')
        self.class_mode = class_mode  
        self.shuffle = shuffle
        self.rescale=rescale
        self.augmentation= aug_mode
        self.on_epoch_end()

    # Denotes the number of batches per epoch. Times the generator is going to be called (how many __getitem__ index parameter).
    def __len__(self):
        return int(np.floor(len(self.img_files) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_imgs_temp = [self.img_files[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.data_generation(list_imgs_temp, list_labels_temp)
        
        return X, y

    # Updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # Generates data containing batch_size samples
    def data_generation(self, list_imgs_temp, list_labels_temp):

        # Initialization
        X = []
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, img_file in enumerate(list_imgs_temp):
            # Store sample
            img = np.array(self.load_img(img_file, target_size=self.target_size))#skimage.io.imread(img_file
            
            
            if self.augmentation:
                aug=augmentation_clss(self.augmentation)
                augmented_img=aug.augment_img(img)
                while(np.array_equal(augmented_img,np.zeros(augmented_img.shape)) == True):  #Avoid black images
                    augmented_img=aug.augment_img(img)
                img = augmented_img
                
              
            if self.rescale:
                img = img.astype(np.float32) * self.rescale
                 
            X.append(img)

            # Store label
            if (self.labels is not None):
                y[i] = list_labels_temp[i]
                
        
        X = np.array(X)
        if self.class_mode == 'input':
            y = X.copy()
        elif self.class_mode == 'sparse':
            y = np.array(y)
        elif self.class_mode == 'binary':
            y = np.array(y).astype(np.float32)
        elif self.class_mode == 'categorical':
            y = np.array(keras.utils.to_categorical(y, num_classes=self.n_classes))
        
        return X, y
    
    #load_img code extracted from keras_preprocessing/image.py
    def load_img(self, path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest'):
        """Loads an image into PIL format.
        # Arguments
            path: Path to image file.
            color_mode: One of "grayscale", "rbg", "rgba". Default: "rgb".
                The desired image format.
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
        # Returns
            A PIL Image instance.
        # Raises
            ImportError: if PIL is not available.
            ValueError: if interpolation method is not supported.
        """
        if grayscale is True:
            warnings.warn('grayscale is deprecated. Please use '
                          'color_mode = "grayscale"')
            color_mode = 'grayscale'
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(path)
        if color_mode == 'grayscale':
            if img.mode != 'L':
                img = img.convert('L')
        elif color_mode == 'rgba':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        elif color_mode == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img  
    
    