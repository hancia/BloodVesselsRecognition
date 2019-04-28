import cv2
from keras_preprocessing.image import img_to_array, array_to_img
from matplotlib import pyplot as plt
from skimage import filters
import numpy as np
from keras import *
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, core
from keras.engine.saving import model_from_json


# from google.colab import drive
# drive.mount('/content/drive/')


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    pred_images = np.empty((pred.shape[0], pred.shape[1]))
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i, pix] = pred[i, pix, 1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] > pred[i, pix, 0]:
                    pred_images[i, pix] = 1
                else:
                    pred_images[i, pix] = 0
    else:
        pass
    pred_images = np.reshape(pred_images, (pred_images.shape[0], 1, patch_height, patch_width))
    return pred_images


def get_picture(i):
    if i < 10:
        i = '0{}'.format(i)
    filename = '/content/drive/My Drive/iwm2/pics/{}_h.jpg'.format(i)
    bwimage = preprocess_image(filename)
    return bwimage


def read_image(path):
    print('Reading image ' + path)
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return im, imgray


def adjust_gamma(imgs, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = cv2.LUT(np.array(imgs, dtype=np.uint8), table)
    # apply gamma correction using the lookup table
    # new_imgs = np.empty(imgs.shape)
    # for i in range(imgs.shape[0]):
    #     new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs


def dataset_normalized(img):
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img - img_mean) / img_std
    img_normalized = ((img_normalized - np.min(img_normalized)) / (
                np.max(img_normalized) - np.min(img_normalized))) * 255
    return img_normalized


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgg = clahe.apply(img)
    # plt.imshow(imgg, cmap=plt.get_cmap("gray"))
    # plt.show()
    return img


def standardize(img):
    new_img = np.empty((img.shape[0], img.shape[1]), dtype=float)
    for im in range(0, img.shape[0]):
        for i in range(0, img.shape[1]):
            new_img[im][i] = float(img[im][i] / 255.0)
    return new_img


def preprocess_image(filename):
    im, imgray = read_image(filename)
    # cv2.equalizeHist(imgray, imgray)
    imgray = apply_clahe(imgray)
    imgray = dataset_normalized(imgray)
    imgray = adjust_gamma(imgray)
    imgray = standardize(imgray)
    # cv2.equalizeHist(imgray, imgray)
    #
    # cv2.GaussianBlur(imgray, (3, 3), 3, imgray)
    # bwimage = imgray
    # bwimage = filters.sobel(imgray)
    # for pixel in range(0, bwimage.shape[0]):
    #     for i in range(0, bwimage.shape[1]):
    #         if bwimage[pixel][i] < 0.03:
    #             bwimage[pixel][i] = 0
    #         else:
    #             bwimage[pixel][i] = 1
    # plt.imshow(imgray, cmap=plt.get_cmap("gray"))
    # plt.show()
    return imgray


def extract_patches(img):
    p = []
    for x in range(0, img.shape[0], 48):
        for y in range(0, img.shape[1], 48):
            patch = img[x:x + 48, y:y + 48]
            # plt.imshow(patch)
            # plt.show()
            if patch.shape != (48, 48):
                new_patch = np.ones((48, 48))
                for x in range(0, patch.shape[0]):
                    for y in range(0, patch.shape[1]):
                        new_patch[x][y] = patch[x][y]
                patch = new_patch
            p.append(img_to_array(patch, data_format="channels_first"))
    return np.array(p)


def get_images(img):
    images = []
    patches = extract_patches(img)
    images.extend(patches)
    return np.array(images)


def get_partial_image(img, finalpic):
    patches_imgs_test = get_images(img)
    predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)

    pred_patches = pred_to_imgs(predictions, 48, 48, "original")
    print(pred_patches.shape)
    x = 0
    y = 0
    for pic in pred_patches:
        for p in pic:
            temp = finalpic[x:x + 48, y:y + 48]
            finalpic[x:x + 48, y:y + 48] = p[0:0 + temp.shape[0], 0:0 + temp.shape[1]]
            y += 48
            if y + 48 > img.shape[1]:
                y = 0
                x += 48
    return finalpic


json_file = open('/content/drive/My Drive/iwm2/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/content/drive/My Drive/iwm2/_last_weights.h5")
print("Loaded model from disk")

img = get_picture(10)
finalpic = np.empty((img.shape[0], img.shape[1]))

finalpic = get_partial_image(img, finalpic)

print("Printing final picture")
plt.imshow(finalpic, cmap=plt.get_cmap("gray"))
plt.show()

finalpic *= 255
cv2.imwrite("/content/drive/My Drive/iwm2/results/finalimage.jpg", finalpic)