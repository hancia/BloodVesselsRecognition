import cv2
from matplotlib import pyplot as plt
from skimage import filters
import random


def extract_patches(img):
    patches = []
    for i in range(9500):
        x = random.randint(0,img.shape[0])
        y = random.randint(0,img.shape[1])
        patch = img[x:x+48, y:y+48]
        patches.append(patch)
    return patches


if __name__ == '__main__':
    filename = 'pics/15_h.jpg'
    im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    plt.imshow(im)
    plt.show()

    cv2.equalizeHist(imgray,imgray)
    plt.imshow(imgray)
    plt.show()

    cv2.GaussianBlur(imgray,  (3, 3), 3, imgray)
    plt.imshow(imgray)
    plt.show()
    bwimage = filters.sobel(imgray)
    for pixel in range (0,bwimage.shape[0]):
        for i in range (0, bwimage.shape[1]):
            if bwimage[pixel][i] < 0.03:
                bwimage[pixel][i] = 255
            else:
                bwimage[pixel][i] = 0
    plt.imshow(bwimage, cmap=plt.get_cmap("gray"))
    plt.show()
    patches = extract_patches(bwimage)