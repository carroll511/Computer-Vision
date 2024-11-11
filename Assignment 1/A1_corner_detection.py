import CV_Assignment_1.A1_image_filtering as imf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import CV_Assignment_1.image_module as im
import CV_Assignment_1.A1_edge_detection as ed

def sobel_filter(img):
    s_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    s_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    df_dx = imf.cross_correlation_2d(img, s_x)
    df_dy = imf.cross_correlation_2d(img, s_y)

    return df_dx, df_dy

def compute_corner_response(img):
    output_img = np.zeros(img.shape)

    # Apply sobel filters
    i_x, i_y = sobel_filter(img)
    i_x2 = i_x ** 2
    i_y2 = i_y ** 2
    i_xy = i_x * i_y

    # compute second moment matrix M
    # You can utilize an uniform window function
    # i.e. w(x, y) = 1 if (x, y) is lying in the window, otherwise w(x, y) = 0
    # Use 5x5 window to compute the matrix M
    window_size = 5
    width = window_size // 2
    
    for i in range(width, output_img.shape[0] - width):
        for j in range(width, output_img.shape[1] - width):
            cov_i_x2 = np.sum(i_x2[i - width: i + width + 1, j - width: j + width + 1])
            cov_i_y2 = np.sum(i_y2[i - width: i + width + 1, j - width: j + width + 1])
            cov_i_xy = np.sum(i_xy[i - width: i + width + 1, j - width: j + width + 1])

            # Use response function with k = 0.04
            det = cov_i_x2 * cov_i_y2 - (cov_i_xy ** 2)
            trace = cov_i_x2 + cov_i_y2
            response = det - 0.04 * (trace ** 2)
            output_img[i, j] = response

    # 다 계산하고, update all the negative responses to 0
    output_img[output_img < 0] = 0

    # normalize them to a range of [0, 1]
    max_val = np.amax(output_img)
    min_val = np.amin(output_img)
    r = (output_img - min_val) / (max_val - min_val)
    return r

def apply_corner_raw(img, img_name):
    start_time = time.time()
    corner_response = compute_corner_response(img)
    end_time = time.time()
    print(f"Computational time for compute_corner_response with {img_name}: {end_time - start_time:.4f} sec")

    # corner_response = np.clip(corner_response, 0, 255).astype(np.uint8)
    plt.imshow(corner_response, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    output_dir = './result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f'{output_dir}/part_3_corner_raw_{img_name}.png', bbox_inches='tight', pad_inches = 0)
    plt.show()

    return corner_response

def change_color_green(R, img, img_name):
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] > 0.1:
                img[i, j] = [0, 255, 0]
    
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    output_dir = './result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/part_3_corner_bin_{img_name}.png', bbox_inches='tight', pad_inches = 0)
    plt.show()

def non_maximum_suppression_win(R, winSize):
    # Suppresses corner response at (x, y) if it is not a maximum value within window
    # centered at (x, y)
    # Although the response is a local maxima, it is suprressed if it not greater than 0.1
    # winSize = 11
    width = winSize // 2
    suppressed_R = np.copy(R)
    for i in range(width, R.shape[0] - width):
        for j in range(width, R.shape[1] - width):
            if np.amax(R[i - width: i + width + 1, j - width: j + width + 1]) > R[i, j]:
                suppressed_R[i, j] = 0

    suppressed_R[suppressed_R < 0.1] = 0
            

    return suppressed_R

def apply_non_maximum_suppression_win(R, img, img_name, winSize):
    start_time = time.time()
    suppressed_R = non_maximum_suppression_win(R, winSize)
    end_time = time.time()

    print(f"Computational time for non_maximum_suppression_win with {img_name}: {end_time - start_time:.4f} sec")

    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    
    for i in range(suppressed_R.shape[0]):
        for j in range(suppressed_R.shape[1]):
            if suppressed_R[i, j] != 0:
                # print(f"Non-suppressed point at ({i}, {j}) with value {suppressed_R[i, j]}")
                circle = plt.Circle((j, i), radius=2, color=(0, 1, 0), fill=False)
                plt.gca().add_patch(circle)
    plt.tight_layout()
    output_dir = './result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/part_3_corner_sup_{img_name}.png', bbox_inches='tight', pad_inches = 0)
    plt.show()

def main():
    lenna = im.load_lenna()
    shapes = im.load_shapes()

    # 3-1 Gaussian filter (7, 1.5)
    filter = imf.get_gaussian_filter_2d(7, 1.5)
    lenna_filtered = imf.cross_correlation_2d(lenna, filter)
    shapes_filtered = imf.cross_correlation_2d(shapes, filter)

    # 3-2 corner response values
    lenna_R = apply_corner_raw(lenna_filtered, 'lenna')
    shapes_R = apply_corner_raw(shapes_filtered, 'shapes')

    # 3-3 Thresholding and NMS
    # a) greater than 0.1 green
    # b) display and store
    change_color_green(lenna_R, lenna, 'lenna')
    change_color_green(shapes_R, shapes, 'shapes')

    # c) Implemment compute local maximas by non-maximum suppression
    winSize = 11
    apply_non_maximum_suppression_win(lenna_R, lenna, 'lenna', winSize)
    apply_non_maximum_suppression_win(shapes_R, shapes, 'shapes', winSize)

if __name__ == "__main__":
    main()