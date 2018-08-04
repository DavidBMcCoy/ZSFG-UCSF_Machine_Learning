"""
utilities for random augmentations on volumetric images
"""

import datetime

import numpy as np

from skimage import exposure
from scipy import ndimage


def no_aug_pass_image(image): 
    return image
    
def random_shift_image(image, isseg=False):
    """
    wow! this function takes ~2.5s on my machine, for one image.
    """
    order = 0 if isseg == True else 5
    image = np.squeeze(image)
    random_pixel = np.random.uniform(low=-20, high=20, size=(2,))
    image = ndimage.interpolation.shift(
        image,
        (int(random_pixel[0]), int(random_pixel[1]), 0),
        order=order,
        mode='nearest')
    image = np.expand_dims(image, axis = 4)
    return image


def random_zoom(image, isseg=False, **kwargs):
    """
    zoom in or out on the first two dimensions of the image volume. note that if we zoom out, the
    extra space in the output image will be filled with zeros.
    """
    order = 0 if isseg == True else 3
    image = np.squeeze(image)
    height, width, depth= image.shape
    # the logic that was here before gave zoom_tuple type (array, array, int...). numpy doesn't
    # implement a __round__ method for singleton arrays (anymore), which broke some of the logic in
    # ndimage.zoom. thus, zoom_tuple's type is now (float, float, int...)
    random_factor = np.random.uniform(low=0.8, high=1.2, size=1)[0]
    zoom_tuple = (random_factor, random_factor) + (1, ) * (image.ndim - 2)

    if random_factor < 1.0:
        zh = int(np.round(height * random_factor))
        zw = int(np.round(width * random_factor))
        top = (height - zh) // 2
        left = (width - zw) // 2

        out = np.zeros_like(image)
        #print(zoom_tuple)
        out[top:top+zh, left:left+zw] = ndimage.zoom(image, zoom_tuple,**kwargs)

    # Zooming in
    elif random_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(height / random_factor))
        zw = int(np.round(width / random_factor))
        top = (height - zh) // 2
        left = (width - zw) // 2

        #print(zoom_tuple)
        out = ndimage.zoom(image[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - height) // 2)
        trim_left = ((out.shape[1] - width) // 2)
        out = out[trim_top:trim_top+height, trim_left:trim_left+width]
    else:
        out = image
    return np.expand_dims(out, axis = 4)


def random_rotation(image, isseg=False):
    """
    apply a random rotation to the first two dimensions of the image
    """
    order = 0 if isseg == True else 5
    random_theta = np.random.uniform(low=-90, high=90, size=(1,))

    return ndimage.rotate(image, float(random_theta), reshape=False, order=order, mode='nearest')


def random_intensify(image):
    """
    scale the image intensity by a random factor
    """
    random_intensify_factor = np.random.uniform(low=0.8, high=1.6, size=1)[0]
    return image*random_intensify_factor


def random_axis_flip(image):
    """
    reflect the image on one of the first two dimensions
    """
    random_axis = np.random.choice([0, 1], size=1)
    if random_axis[0] == 0:
        image = np.fliplr(image)
    if random_axis[0] == 1:
        image = np.flipud(image)

    return image


def contrast_stretching(image):
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))

    return image


def histogram_equalization(image):
    image = exposure.equalize_hist(image)

    return image


# def adaptive_equalization(image):
#     image = exposure.equalize_adapthist(image, clip_limit=0.03)
#
#     return image

def affine_transform(image):
    """
    apply a random affine transformation to one of the first two dimensions of the image
    this one takes 5s on my machine.
    """
    random_axis = np.random.choice([0, 1], size=1)
    if random_axis[0] == 0:
        random_theta = np.random.uniform(low=0, high=0.2, size=(1,))
        random_x_shear = np.matrix([[1, random_theta, 0, 0], [0, 1, 0, 0],[0,0,1,0],[0,0,0,0]])
        image = ndimage.interpolation.affine_transform(image, matrix = random_x_shear)
    if random_axis[0] == 1:
        random_theta = np.random.uniform(low=0, high=0.2, size=(1,))
        random_y_shear = np.matrix([[1, 0, 0, 0], [random_theta, 1, 0, 0],[0,0,1,0],[0,0,0,0]])
        image = ndimage.interpolation.affine_transform(image, matrix = random_y_shear)

    return image


def noisy(image, noise_type=None):
    """
    add noise to the image
    NOTE: haven't validated numerical correctness, but it runs to completion
    """
    if noise_type is None:
        noises = ["gauss", "s&p", "poisson", "speckle"]
        selection = np.random.randint(len(noises))
        noise_type = noises[selection]

    if noise_type == "gauss":
        row, col, dept, ch= image.shape
        mean = 0
        var = 0.001
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, dept, ch))
        out = image + gauss

    elif noise_type == "s&p":
        # BIG NOTE: i'm assuming this noise type is supposed to randomly (max out | zero out)
        # pixels in the image. in this context, the max value is 255 so that's what i'm using
        # instead of 1.
        salt_value = 255
        pepper_value = 0
        row,col,dept,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = int(np.ceil(amount * image.size * s_vs_p))
        salt_idx = np.random.permutation(out.size)[:num_salt]
        np.put(out, salt_idx, salt_value)

        # Pepper mode
        num_pepper = int(np.ceil(amount * image.size * (1. - s_vs_p)))
        pepper_idx = np.random.permutation(out.size)[:num_pepper]
        np.put(out, pepper_idx, pepper_value)

    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        out = np.random.poisson(image * vals) / float(vals)

    elif noise_type == "speckle":
        row,col,dept,ch = image.shape
        gauss = np.random.randn(row,col,dept,ch)
        gauss = gauss.reshape(row,col,dept,ch)
        out = image + image * gauss

    else:
        raise ValueError("noise type {} is not supported".format(noise_type))

    return out


def distort_elastic(image, smooth=10.0, scale=100.0, seed=0, distort_z=True):
    """
    Elastic distortion of images.

    Channel axis in RGB images will not be distorted but grayscale or
    RGB images are both valid inputs. RGB and grayscale images will be
    distorted identically for the same seed.

    Simard, et. al, "Best Practices for Convolutional Neural Networks
    applied to Visual Document Analysis",
    in Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    :param ndarray image: Image of shape [h,w] or [h,w,c]
    :param float smooth: Smoothes the distortion.
    :param float scale: Scales the distortion.
    :param int seed: Seed for random number generator. Ensures that for the
      same seed images are distorted identically.
    :return: Distorted image with same shape as input image.
    :rtype: ndarray

    NOTE: modified for volumetric images. distort_z controls whether image is distorted along Z
    axis or not.
    runs in about 2.5s.
    I don't exactly grasp what it's supposed to be doing, so i can't guarantee it's correct.
    """

    #raise ArithmeticError("this function works but may not be correct. caveat utilitor!")

    if distort_z:
        # image will be distorted on Z axis
        # create random, smoothed displacement field
        rnd = np.random.RandomState(int(seed))
        h, w, d, c = image.shape
        dxy = rnd.rand(3, h, w, d, c) * 2 - 1
        dxy = ndimage.gaussian_filter(dxy, smooth, mode="constant")
        dxy = dxy / np.linalg.norm(dxy) * scale
        dxyz = dxy[0], dxy[1], dxy[2], np.zeros_like(dxy[0])

        # create transformation coordinates and deform image
        ranges = [np.arange(d) for d in image.shape]
        grid = np.meshgrid(*ranges, indexing='ij')
        idx = [np.reshape(v + dv, (-1, 1)) for v, dv in zip(grid, dxyz)]
        distorted = ndimage.map_coordinates(image, idx, order=1, mode='reflect')

    else:
        # image won't be distorted on Z axis
        # create random, smoothed displacement field
        rnd = np.random.RandomState(int(seed))
        h, w, d, c = image.shape
        dxy = rnd.rand(2, h, w, d, c) * 2 - 1
        dxy = ndimage.gaussian_filter(dxy, smooth, mode="constant")
        dxy = dxy / np.linalg.norm(dxy) * scale
        dxyz = dxy[0], dxy[1], np.zeros_like(dxy[0]), np.zeros_like(dxy[0])

        # create transformation coordinates and deform image
        ranges = [np.arange(d) for d in image.shape]
        grid = np.meshgrid(*ranges, indexing='ij')
        print((grid[0] + dxyz[0]).shape)
        idx = [np.reshape(v + dv, (-1, 1)) for v, dv in zip(grid, dxyz)]
        print(np.asarray(idx).shape)
        distorted = ndimage.map_coordinates(image, idx, order=1, mode='reflect')

    return distorted.reshape(image.shape)


def random_batch_augmentation(batch,
                              allowed_transformations,
                              max_transformations,
                              seed=None):
    """
    perform up to `max_transformations` number of transformations on each image in the batch, out of
    the set of transformations indexed by allowed_transformations.
    """
    shape = batch.shape
    batch_aug = np.empty(shape, dtype=batch.dtype)

    transformations = [
        no_aug_pass_image,
        random_shift_image,
        random_rotation,
        random_intensify,
        random_axis_flip,
        contrast_stretching,
        histogram_equalization,
        affine_transform,
        random_zoom,
        noisy,
        distort_elastic]

    # iterate over images in the batch
    for index, image in enumerate(batch):

        # seed the generator?
        if seed is not None:
            np.random.seed(seed)

        # how many transformations can we do?
        if max_transformations > 1:
            num_transformations = np.random.randint(max_transformations)
        else:
            num_transformations = 1

        # which ones shall we do?
        which_transformations = np.random.choice(
            allowed_transformations,
            num_transformations,
            replace=False)

        # execute selected transformations sequentially
        for this_transformation_index in which_transformations:
            image = transformations[this_transformation_index](image)

        # fire this augmented image into the batch accumulator
        batch_aug[index,...] = image

    return batch_aug
