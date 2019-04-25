import cv2
from keras.engine.saving import model_from_json
from keras_preprocessing.image import img_to_array, array_to_img
from matplotlib import pyplot as plt
from skimage import filters
import numpy as np


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


def read_image(path):
    print('Reading image ' + path)
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return im, imgray


def preprocess_image(filename):
    im, imgray = read_image(filename)

    cv2.equalizeHist(imgray, imgray)

    cv2.GaussianBlur(imgray, (3, 3), 3, imgray)
    bwimage = filters.sobel(imgray)
    # bwimage = imgray
    for pixel in range(0, bwimage.shape[0]):
        for i in range(0, bwimage.shape[1]):
            if bwimage[pixel][i] < 0.03:
                bwimage[pixel][i] = 0
            else:
                bwimage[pixel][i] = 1
    # plt.imshow(bwimage, cmap=plt.get_cmap("gray"))
    # plt.show()
    return bwimage


def extract_patches(img, i, j):
    p = []
    for x in range(0, 480, 48):
        for y in range(0, 480,  48):
            patch = img[x+i:x+i+ 48, y+j :y+j+ 48]
            # plt.imshow(patch)
            # plt.show()
            if patch.shape != (48,48):
                new_patch = np.empty((48,48))
                for x in range(0,patch.shape[0]):
                    for y in range(0, patch.shape[1]):
                        new_patch[x][y] = patch[x][y]
                patch = new_patch
            p.append(img_to_array(patch, data_format="channels_first"))
    return np.array(p)


def get_picture(i):
    if i < 10:
        i = '0{}'.format(i)
    filename = 'pics/{}_h.jpg'.format(i)
    bwimage = preprocess_image(filename)
    return bwimage


def get_images(img, i, j):
    images = []
    patches = extract_patches(img, i, j)
    images.extend(patches)
    return np.array(images)


def recompone(data,N_h,N_w):
    N_pacth_per_img = N_w*N_h
    N_full_imgs = int(data.shape[0]/N_pacth_per_img)
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w*N_h
    full_recomp = np.empty((N_full_imgs,data.shape[1],N_h*patch_h,N_w*patch_w))
    k = 0
    s = 0
    while (s<data.shape[0]):
        single_recon = np.empty((data.shape[1],N_h*patch_h,N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]=data[s]
                s+=1
        full_recomp[k]=single_recon
        k+=1
    return full_recomp


def get_partial_image(img, finalpic, i, j):
    patches_imgs_test = get_images(img, i, j)
    predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)

    pred_patches = pred_to_imgs(predictions, 48, 48, "original")

    pred_imgs = recompone(pred_patches, 1, 1)

    x = i
    y = j
    for pic in pred_imgs:
        for p in pic:
            temp = finalpic[x:x + 48, y:y + 48]
            finalpic[x:x + 48, y:y + 48] = p[0:0+temp.shape[0], 0:0+temp.shape[1]]
        y += 48
        if y + 48 > j+480:
            y = j
            x += 48
    return finalpic


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("_last_weights.h5")
print("Loaded model from disk")

img = get_picture(1)
finalpic = np.empty((img.shape[0], img.shape[1]))

for i in range(0, finalpic.shape[0]-48, 480):
    for j in range(0, finalpic.shape[1]-48, 480):
        finalpic = get_partial_image(img, finalpic, i, j)

plt.imshow(finalpic)
plt.show()
