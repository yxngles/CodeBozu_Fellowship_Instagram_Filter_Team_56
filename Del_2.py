import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

bozu = cv.imread("Media/Bozu.png")
red_bozu = cv.imread("Del 1/Red_Bozu.jpg")
gray_bozu = cv.imread("Del 1/Gray_Bozu.jpg")

bozu_float = np.array(bozu, dtype="float32")
rows, cols = red_bozu.shape[:2]

v_flip = np.zeros([rows, cols, 3], dtype="uint8")
h_flip = np.zeros([rows, cols, 3], dtype="uint8")


def vertical_flip(img):
    for i in range(rows):
        v_flip[i, :] = img[rows - i - 1]
    return v_flip


def horizontal_flip(img):
    for i in range(rows):
        h_flip[i] = img[i][::-1]
    return h_flip


def clip(broken_image):
    broken_image[broken_image < 0] = 0
    broken_image[broken_image > 255] = 255


def contrast(image, alpha):
    contrast_array = np.ones((rows, cols, 3), dtype="float32") * alpha
    contrasted = cv.multiply(image, contrast_array)
    clip(contrasted)
    contrasted = np.array(contrasted, dtype="uint8")
    return contrasted


def add_brightness(image, beta):
    brightness_array = np.ones((rows, cols, 3), dtype="float32") * beta
    brightened = cv.add(image, brightness_array)
    clip(brightened)
    brightened = np.array(brightened, dtype="uint8")
    return brightened


def apply_threshold(image, threshold):
    temp = np.zeros((rows, cols, 3), dtype="uint8")
    temp[:, :] = image[:, :]
    temp[temp < threshold] = 0
    temp[temp > threshold] = 255
    return temp


cv.imwrite("Del 2/Vertical_Red_Bozu.png", vertical_flip(red_bozu))
cv.imwrite("Del 2/Horizontal_Red_Bozu.png", horizontal_flip(red_bozu))
cv.imwrite("Del 2/Contrasted_Bozu.jpg", contrast(bozu_float, 1.5))
cv.imwrite("Del 2/Bright_Bozu.jpg", add_brightness(bozu_float, 150))
cv.imwrite("Del 2/Bozu_in_the_dark.jpg", apply_threshold(gray_bozu, 160))
cv.imwrite("Del 2/Silhouette_Bozu.jpg", apply_threshold(gray_bozu, 100))

