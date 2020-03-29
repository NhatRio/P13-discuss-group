import os
from pathlib import Path
import cv2 as cv
import numpy as np
import tifffile
import scipy.ndimage

INPUT_IMAGE = r"D:\Viet\craftmanship\tensorflow\demos\images\lenna.png"
CONV1_WEIGHT = r"D:\Viet\craftmanship\tensorflow\demos\images\conv1.npy"
CONV2_WEIGHT = r"D:\Viet\craftmanship\tensorflow\demos\images\conv2.npy"

def read_image_opencv(file_name):
    image = cv.imread(file_name)
    image = image[..., ::-1]  # conversion BGR OpenCV to RGB
    return image

def load_npy(file_name):
    # weight is saved in (output_channels x input_channels x height x width) format. Need to transpose to
    # (height x width x input_channels x output_channels) format
    weight = np.load(file_name)
    return np.transpose(weight, (2, 3, 1, 0))

def compute_feature_map(image, weight, output_path, suffix="", do_upscale=False):
    assert image.dtype == np.float32, "Image must be convert to float32 before computing convolution"
    assert image.ndim == 3, "image must be a 3D array of size (height x width x channel), e.g 512x512x3"
    assert weight.ndim == 4, "weight must be a 4D array with format (height x width x input_channels x output_channels), e.g. 3x3x3x16"
    assert image.shape[2] == weight.shape[2], "Number of channels of image must be equal to the number of input channels of the weight"
    # weight contains several filters (e.g 16). For each filter, compute the convolutional result with image
    img_h, img_w, img_c = image.shape
    output_channels = weight.shape[-1]
    conv_image = np.zeros((img_h, img_w, output_channels), dtype=np.float32)
    # for saved some intermediate convolved image
    base_name = Path(output_path).stem
    dir_name = Path(output_path).parent
    for i in range(output_channels):
        print(f"{suffix} - Compute output feature map on {i}/{output_channels} channel")
        conv_image[:, :, i] = scipy.ndimage.convolve(image, weight[..., i], mode="constant", cval=0.0)[..., (img_c-1)//2]
        if i % 5 == 0:
            """Convert and save image as uint16 TIFF."""
            tmp = np.uint16(conv_image[:, :, i] * (2 ** 16 - 1))
            # upscale the intermediate image if needed for visualization purpose.
            if do_upscale:
                tmp = scipy.ndimage.zoom(tmp,zoom=2, mode="nearest")
            cv.imwrite(os.path.join(dir_name, base_name + suffix + "_" + str(i) + ".tiff"), tmp)
    return conv_image


def read_image_and_apply_convs():
    # read image
    image = read_image_opencv(INPUT_IMAGE)
    # normalize between [0, 1]
    image = image.astype(np.float32) / 255
    print("imput imaeg", image.shape)

    # load weight of first convolution
    w1 = load_npy(CONV1_WEIGHT)
    print("first conv", w1.shape)
    # apply first convolution
    conv1_image = compute_feature_map(image, w1, INPUT_IMAGE, suffix="_conv1")
    print("first results", conv1_image.shape)

    # downsample image (pooling) and apply second conv
    conv2_image = conv1_image[0::2, 0::2, :]
    print("Pooling", conv2_image.shape)

    # load weight of second convolution
    w2 = load_npy(CONV2_WEIGHT)
    print("second conv", w2.shape)
    # apply second conv
    conv2_image = compute_feature_map(conv2_image, w2, INPUT_IMAGE, suffix="_conv2", do_upscale=True)
    print("after second conv", conv2_image.shape)

    return conv1_image, conv2_image


if __name__ == "__main__":
    # pass image through the first conv
    conv1, conv2 = read_image_and_apply_convs()
    print(conv1.shape)
    print(conv2.shape)
