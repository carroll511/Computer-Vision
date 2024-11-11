import cv2
import numpy as np
import CV_Assignment_1.image_module as im
import matplotlib.pyplot as plt
import os
import time

# 1-1. Image Filtering by Cross-Correlation
def image_padding_1d_h(img, size): # size -> pad size
    pad_left = img[:, 0].reshape(-1, 1)
    pad_right = img[:, -1].reshape(-1, 1)

    for i in range(size):
        img = np.hstack((pad_left, img))
    for i in range(size):
        img = np.hstack((img, pad_right))
    
    return img  

def image_padding_1d_v(img, size): # size -> pad size
    pad_up = img[0, :].reshape(1, -1)
    pad_down = img[-1, :].reshape(1, -1)

    for i in range(size):
        img = np.vstack((pad_up, img))
    for i in range(size):
        img = np.vstack((img, pad_down))

    return img

def image_padding_2d(img, size_h, size_w): # size -> pad size
    pad_left = img[:, 0].reshape(-1, 1)
    pad_right = img[:, -1].reshape(-1, 1)

    for i in range(size_w):
        img = np.hstack((pad_left, img, pad_right))

    pad_up = img[0, :].reshape(1, -1)
    pad_down = img[-1, :].reshape(1, -1)

    for i in range(size_h):
        img = np.vstack((pad_up, img, pad_down))

    return img
    
def cross_correlation_1d(img, kernel):
    # Get the dimensions of the image and kernel
    filtered_img = np.zeros(img.shape)
    img_h, img_w = img.shape

    kernel_h, kernel_w = kernel.shape

    if kernel.shape[0] == 1: # 행벡터
        axis = 0
        padded_img = image_padding_1d_h(img, kernel_w // 2)
    
    else: # 열벡터
        axis = 1
        padded_img = image_padding_1d_v(img, kernel_h // 2)

    for i in range(img_h):
        for j in range(img_w):
            if axis == 0: # 행벡터
                filtered_img[i, j] = np.sum(padded_img[i, j:j + kernel_w] * kernel)
            else: # 열벡터
                filtered_img[i, j] = np.sum(padded_img[i:i + kernel_h, j] * kernel.flatten())

    # filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
    return filtered_img

def cross_correlation_2d(img, kernel):
    # Get the dimensions of the image and kernel
    filtered_img = np.zeros(img.shape)
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape

    padded_img = image_padding_2d(img, kernel_h // 2, kernel_w // 2)

    for i in range(img_h):
        for j in range(img_w):
            filtered_img[i, j] = np.sum(padded_img[i:i + kernel_h, j:j + kernel_w] * kernel)

    # filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
    return filtered_img

# 1-2. The Gaussian Filter
def get_gaussian_filter_1d(size, sigma):
    filter = np.zeros((1, size))
    for i in range(size):
        x = i - size // 2
        filter[0, i] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))
    filter /= np.sum(filter)
    return filter

def get_gaussian_filter_2d(size, sigma):
    filter = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = i - size // 2
            y = j - size // 2
            filter[i, j] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    filter /= np.sum(filter)
    return filter

def visualize_nine_gaussian_filters(img, input_image_file_name):
    kernel = [5, 11, 17]
    sigma = [1, 6, 11]

    fig = plt.figure(figsize=(10, 10))

    for i in range(len(kernel)):
        for j in range(len(sigma)):
            filter = get_gaussian_filter_2d(kernel[i], sigma[j])
            filtered_img = cross_correlation_2d(img, filter)
            filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
            plt.subplot(3, 3, i * 3 + j + 1)
            plt.imshow(filtered_img, cmap='gray')
            plt.axis('off')

            text = f"{kernel[i]}x{kernel[i]}, sigma={sigma[j]}"
            plt.text(0.5, 1.00, text, fontsize=10, ha='center', va='top', transform=plt.gca().transAxes)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    output_dir = './result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/part_1_gaussian_filtered_{input_image_file_name}.png', bbox_inches='tight', pad_inches = 0)

    plt.show()
    return img

def compare_results(img, kernel_1d, kernel_2d):
    start_1d = time.time()
    result_1d = cross_correlation_1d(img, kernel_1d)
    result_1d = cross_correlation_1d(result_1d, kernel_1d.reshape((-1, 1)))
    end_1d = time.time()

    start_2d = time.time()
    result_2d = cross_correlation_2d(img, kernel_2d)
    end_2d = time.time()

    print(f"1D: {end_1d - start_1d:.4f} sec")
    print(f"2D: {end_2d - start_2d:.4f} sec")

    difference_map = result_1d - result_2d

    for i in range(difference_map.shape[0]):
        for j in range(difference_map.shape[1]):
            if difference_map[i, j] < 0:
                difference_map[i, j] = 0

    total_difference = np.sum(np.absolute(difference_map))
    print(f"Total sum of absolute differences: {total_difference}")

    result_1d = np.clip(result_1d, 0, 255).astype(np.uint8)
    result_2d = np.clip(result_2d, 0, 255).astype(np.uint8)
    difference_map = np.clip(difference_map, 0, 255).astype(np.uint8)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(result_1d, cmap='gray')
    plt.title("1D Filter Result")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(result_2d, cmap='gray')
    plt.title("2D Filter Result")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(difference_map, cmap='gray')
    plt.title("Difference Map")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    lenna = im.load_lenna()
    shapes = im.load_shapes()
    
    # 1-2-c)
    print(get_gaussian_filter_1d(5, 1))
    print(get_gaussian_filter_2d(5, 1))

    # 1-2-d)
    nine_lenna = visualize_nine_gaussian_filters(lenna, 'lenna')
    nine_shapes = visualize_nine_gaussian_filters(shapes, 'shapes')

    # 1-2-e)
    kernel_1d = get_gaussian_filter_1d(17, 6)
    kernel_2d = get_gaussian_filter_2d(17, 6)
    compare_results(lenna, kernel_1d, kernel_2d)
    compare_results(shapes, kernel_1d, kernel_2d)

    # filtered_img_using_1d = cv2.sepFilter2D(lenna, -1, kernel_1d, kernel_1d)
    # filtered_img_using_2d = cv2.filter2D(lenna,-1, 6)
    # Difference_Map = np.abs(filtered_img_using_1d-filtered_img_using_2d)
    # abs_diff_sum = np.sum(Difference_Map)
    # plt.imshow(Difference_Map, cmap='gray')
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()
    # print(abs_diff_sum)


    # kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])  # Example 1D kernel
    # print(kernel.shape)
    # # print(kernel.shape[0])

    # # 1D cross-correlation (custom implementation)
    # result_custom = cross_correlation_2d(lenna, kernel)

    # # OpenCV implementation using filter2D
    # result_cv2 = cv2.filter2D(lenna, -1, kernel)

    # # Display both results for comparison
    # cv2.imshow("Custom 1D Cross-Correlation", result_custom)
    # cv2.imshow("OpenCV filter2D Result", result_cv2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(result_custom, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(result_cv2, cmap='gray')
    # plt.axis('off')

    # plt.show()

if __name__ == "__main__":
    main()

