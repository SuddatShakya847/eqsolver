import os
from os import listdir
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def show_processed_images(segmented_chars, processed_images):
    if not processed_images or not segmented_chars:
        return

    # segmented images
    num = len(segmented_chars)
    rows = int(np.ceil(num / 5))
    plt.figure(figsize=(10, 2 * rows))
    for i, char_data in enumerate(segmented_chars):
        plt.subplot(rows, 5, i + 1)
        plt.imshow(char_data['image'], cmap='gray')
        plt.axis('off')

    plt.tight_layout()

    # processed images
    num = len(processed_images)
    rows = int(np.ceil(num / 2))
    plt.figure(figsize=(12, 4 * rows))
    for i, img in enumerate(processed_images):
        plt.subplot(rows, 2, i + 1)
        plt.imshow(img['image'], cmap='gray')
        plt.title(img['title'])
        plt.axis('off')

    # Increase space between images
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.tight_layout()
    plt.show()


def image_padding(image):
    height, width = image.shape
    aspect_ratio = width / height

    if aspect_ratio == 1.0:
        return image

    elif aspect_ratio > 1.0:
        # padding on the top and bottom
        diff = width - height
        top = diff // 2
        bottom = diff // 2
        return cv.copyMakeBorder(
            image, top, bottom, 0, 0, cv.BORDER_CONSTANT)

    elif aspect_ratio < 1.0:
        diff = height - width
        left = diff // 2
        right = diff // 2
        # padding on the left and right
        return cv.copyMakeBorder(
            image, 0, 0, left, right, cv.BORDER_CONSTANT)


# process image to be passed into model when predicting
def process_image(image, isDebug: bool):
    img_original = image.copy()
    grayscaled = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binarized = cv.threshold(
        grayscaled, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # remove noise
    kernel = np.ones((2, 2), np.uint8)
    binarized = cv.morphologyEx(binarized, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(
        binarized, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours by x-coordinate (left to right)
    sorted_contours = sorted(
        contours, key=lambda c: cv.boundingRect(c)[0])

    # Filter out very small contours (noise)
    char_contours = [
        c for c in sorted_contours if cv.contourArea(c) > 100]

    segmented_chars = []
    img_rect = img_original.copy()
    for contour in char_contours:
        # Get bounding box
        x, y, w, h = cv.boundingRect(contour)
        img_rect = cv.rectangle(
            img_rect, (x, y), (x + w, y + h), (0, 255, 0), 8)

        # Extract character ROI with some padding
        padding = 2
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(binarized.shape[1], x + w + padding)
        y2 = min(binarized.shape[0], y + h + padding)

        # Extract character
        char_img = binarized[y1:y2, x1:x2]

        # add padding to make it square
        char_img = image_padding(char_img)
        # resize for the model
        char_img = cv.resize(char_img, (28, 28), interpolation=cv.INTER_AREA)

        # Save for later recognition
        segmented_chars.append({
            'image': char_img,
            'position': (x, y, w, h),
        })

    processed_images = []
    if isDebug:
        contours_img = cv.drawContours(image, contours, -1, (255, 0, 255), 3)
        processed_images = [
            {"title": "Grayscaled", "image": grayscaled},
            {"title": "Binarized", "image": binarized},
            {"title": "Contours", "image": contours_img},
            {"title": "Segments", "image": img_rect}
        ]

    processed = img_rect
    return (processed, segmented_chars, processed_images)


# These are utils to process the dataset ####

def count_dataset(path: str = "dataset/"):
    map: dict = dict()
    for files in listdir(path):
        symbol, _ = files.split("-")
        if symbol in map:
            map[symbol] += 1
            continue

        map[symbol] = 1

    print(map)


def process_dataset(dir_path: str, processed_dir: str):
    for file in listdir(dir_path):
        if os.path.isdir(file):
            continue

        [filename, extension] = file.split(".")
        if extension != "jpg" and extension != "png":
            continue

        # add / at the end of path if its not there
        dir_path = dir_path if dir_path[-1] == '/' or dir_path[-1] == '\\' \
            else dir_path + '/'
        processed_dir = processed_dir if processed_dir[-1] == '/' \
            or processed_dir[-1] == '\\' \
            else processed_dir + '/'

        img_path = dir_path + file

        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.bitwise_not(image)
        _, image = cv.threshold(
            image, 128, 255, cv.THRESH_BINARY + cv.THRESH_BINARY)
        image = image.reshape(28, 28, 1)

        split = file.split("-")

        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        folder = processed_dir + split[0] + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        cv.imwrite(folder + filename + ".jpg", image[:, :, 0])


# organizes images into folders according to their label
def organize_dir(dir_path: str = "processed/"):
    for file in listdir(dir_path):
        img_path = dir_path + file
        split = file.split("-")

        folder = dir_path + split[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        os.rename(img_path, folder + '/' + file)


def main():
    # organize_dir()
    process_dataset("dataset/", "processed/")


if __name__ == "__main__":
    main()
