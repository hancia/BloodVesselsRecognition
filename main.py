import random
import cv2
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, core, Flatten, ZeroPadding2D
from keras_preprocessing.image import img_to_array
from keras import *
import numpy as np


# from google.colab import drive
# drive.mount('/content/drive/')


def get_unet(n):
    inputs = Input(shape=(1, n, n))
    print(inputs.shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    print(conv1.shape)
    conv1 = Dropout(0.2)(conv1)
    print(conv1.shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    print(conv1.shape)
    pool1 = MaxPooling2D((2, 2))(conv1)
    print(pool1.shape)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    print(pool2.shape)

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

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    print(up2.shape)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    print(conv5.shape)

    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
    print(conv6.shape)
    conv6 = core.Reshape((2, 48 * 48))(conv6)
    conv6 = core.Permute((2, 1))(conv6)

    conv7 = core.Activation('softmax')(conv6)
    print("conv7 shape")
    print(conv7.shape)

    model = Model(inputs=inputs, outputs=conv7)

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_model(images, masks):
    model = get_unet(48)

    model_json = model.to_json()
    with open("/content/drive/My Drive/iwm2/model.json", "w") as json_file:
        json_file.write(model_json)
    model.fit(images, masks, nb_epoch=50, batch_size=32, verbose=2, shuffle=True, validation_split=0.1)
    model.save_weights('/content/drive/My Drive/iwm2/weights.h5', overwrite=True)


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


def equalize_histogram(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


def normalize(img):
    new_img = np.empty((img.shape[0], img.shape[1]), dtype=float)
    for im in range(0, img.shape[0]):
        for i in range(0, img.shape[1]):
            new_img[im][i] = float(img[im][i] / 255.0)
    return new_img


def preprocess_image(filename):
    im, imgray = read_image(filename)
    imgray = equalize_histogram(imgray)
    imgray = normalize(imgray)
    return imgray


def extract_patches(img, bwmask):
    p = []
    m = []
    for i in range(10000):
        x = random.randint(1, img.shape[0] - 48)
        y = random.randint(1, img.shape[1] - 48)
        patch = img[x:x + 48, y:y + 48]
        mask1 = bwmask[x:x + 48, y: y + 48]
        msk = np.empty((2304, 2))
        if patch.shape[0] != 48 or patch.shape[1] != 48:
            continue
        pixel = 0
        for row in mask1:
            for pix in row:
                if pix == 0:
                    msk[pixel][0] = 1
                    msk[pixel][1] = 0
                else:
                    msk[pixel][1] = 1
                    msk[pixel][0] = 0
                pixel += 1
        p.append(img_to_array(patch, data_format="channels_first"))
        m.append(msk)
    return np.array(p), np.array(m)


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
    # masks = masks_Unet(masks)
    print(images.shape)
    print(masks.shape)
    build_model(images, masks)
