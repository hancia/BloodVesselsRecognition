import random
import cv2
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, core, Flatten, ZeroPadding2D
from keras_preprocessing.image import img_to_array
from skimage import filters
from keras import *
import numpy as np
from matplotlib import pyplot as plt
# from google.colab import drive
# drive.mount('/content/drive/')

def get_unet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(1, patch_height, patch_width))
    print(inputs.shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    print(conv1.shape)
    conv1 = Dropout(0.2)(conv1)
    print(conv1.shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    print(conv1.shape)
    pool1 = MaxPooling2D((2, 2))(conv1)
    print(pool1.shape)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    print(pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    print(conv3.shape)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    print(up1.shape)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    print(conv4.shape)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    print(up2.shape)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    print(conv5.shape)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
    print(conv6.shape)
    conv6 = core.Reshape((2, 48 * 48))(conv6)
    conv6 = core.Permute((2, 1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)
    print("conv7 shape")
    print(conv7.shape)

    model = Model(inputs=inputs, outputs=conv7)
    model.load_weights("/content/drive/My Drive/iwm2/_last_weights.h5")

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model(images, masks):
    # =========== Construct and save the model arcitecture =====
    n_ch = 1
    patch_height = 48
    patch_width = 48
    model = get_unet(n_ch, patch_height, patch_width)

    # ============  Training ==================================
    checkpointer = ModelCheckpoint(filepath='./_best_weights.h5', verbose=1, monitor='val_loss', mode='auto',
                                   save_best_only=True)  # save at each epoch if the validation decreased

    model_json = model.to_json()
    with open("/content/drive/My Drive/iwm2/model.json", "w") as json_file:
        json_file.write(model_json)
    model.fit(images, masks, nb_epoch=50, batch_size=32, verbose=2, shuffle=True, validation_split=0.1)

    # ========== Save and test the last model ===================
    model.save_weights('/content/drive/My Drive/iwm2/_last_weights.h5', overwrite=True)


def read_image(path):
    print('Reading image ' + path)
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return im, imgray


def read_mask(path):
    print('Reading image ' + path)
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    for i in range(0, imgray.shape[0]):
        for j in range(0, imgray.shape[1]):
            if imgray[i][j] == 255:
                imgray[i][j] = 1
            else:
                imgray[i][j] = 0
    return im, imgray


def adjust_gamma(imgs, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
    # apply gamma correction using the lookup table
    # new_imgs = np.empty(imgs.shape)
    # for i in range(imgs.shape[0]):
    #     new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs


def dataset_normalized(img):
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img-img_mean)/img_std
    img_normalized = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized)-np.min(img_normalized)))*255
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
            new_img[im][i] = float(img[im][i]/255.0)
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

#     plt.imshow(imgray, cmap=plt.get_cmap("gray"))
#     plt.show()
    return imgray


def extract_patches(img, bwmask):
    p = []
    m = []
    for i in range(10000):
        x = random.randint(1,img.shape[0]-48)
        y = random.randint(1,img.shape[1]-48)
        patch = img[x:x + 48, y:y + 48]
        mask1 = bwmask[x:x + 48, y: y + 48]
#         if np.sum(np.array(mask1)) <= 0.004:
#             # plt.imshow(mask1)
#             # plt.show()
#             continue
        if patch.shape[0] != 48 or patch.shape[1] != 48:
            continue
        # plt.imshow(patch)
        # plt.show()
        p.append(img_to_array(patch, data_format="channels_first"))
        m.append(img_to_array(mask1, data_format="channels_first"))
    return np.array(p), np.array(m)


def masks_Unet(masks):
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, 2))
    for i in range(masks.shape[0]):
        for j in range(im_h * im_w):
            if masks[i, j] == 0:
                new_masks[i, j, 0] = 1
                new_masks[i, j, 1] = 0
            else:
                new_masks[i, j, 0] = 0
                new_masks[i, j, 1] = 1
    return new_masks


if __name__ == '__main__':
    images = []
    masks = []
    for i in range(1, 9):
        if i < 10:
            i = '0{}'.format(i)
        filename = '/content/drive/My Drive/iwm2/pics/{}_h.jpg'.format(i)
        maskname = '/content/drive/My Drive/iwm2/pics/{}_h.tif'.format(i)
        bwimage = preprocess_image(filename)
        # im, m = read_image(filename)
        m, mask = read_mask(maskname)
        patches, maskspatches = extract_patches(bwimage, mask)
        masks.extend(maskspatches)
        images.extend(patches)
    # print(images)
    # print(masks)
    images = np.array(images)
    masks = np.array(masks)
    print(masks.shape)
    masks = masks_Unet(masks)
    print(images.shape)
    print(masks.shape)
    build_model(images, masks)
