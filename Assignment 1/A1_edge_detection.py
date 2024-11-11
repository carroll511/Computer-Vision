import CV_Assignment_1.A1_image_filtering as imf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import CV_Assignment_1.image_module as im

def compute_image_gradient(img):
    # Apply Sobel filter
    s_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    s_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    df_dx = imf.cross_correlation_2d(img, s_x)
    df_dy = imf.cross_correlation_2d(img, s_y)

    # Each pixel, compute magnitude and direction of gradient
    mag = np.sqrt(df_dx ** 2 + df_dy ** 2)
    dir = np.arctan2(df_dy, df_dx)

    return mag, dir

def apply_gradient_show_save(img, img_name):
    start_time = time.time()
    mag, dir = compute_image_gradient(img)
    end_time = time.time()
    print(f"Computational time for compute_image_gradient with {img_name}: {end_time - start_time:.4f} sec")

    mag = np.clip(mag, 0, 255).astype(np.uint8)
    plt.imshow(mag, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    output_dir = './result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/part_2_edge_raw_{img_name}.png', bbox_inches='tight', pad_inches = 0)
    plt.show()

def non_maximum_suppression_dir(mag, dir):
    # a) angle(deg), map the direction to the closest representative angle [0, 45, 90, 135, 180, 225, 270, 315]
    angle = np.rad2deg(dir) + 180 # atan2 범위가 -pi ~ pi
    suppressed_mag = np.copy(mag)

    # b) Compare against two magnitudes along quantizied direction
    # don't have to interpolate
    # If the gradient magnitude at the center position is not greater than the ones along the gradient direction,
    # it is suppressed to zero
    for i in range(1, angle.shape[0] - 1):
        for j in range(1, angle.shape[1] - 1):
            tmp = angle[i, j]
            if ((0 <= tmp < 22.5) or (337.5 <= tmp <= 360)) or (157.5 <= tmp < 202.5):
                if mag[i, j] < mag[i, j - 1] or mag[i, j] < mag[i, j + 1]:
                    suppressed_mag[i, j] = 0
    
            elif (22.5 <= tmp < 67.5) or (202.5 <= tmp < 247.5):
                if mag[i, j] < mag[i - 1, j - 1] or mag[i, j] < mag[i + 1, j + 1]:
                    suppressed_mag[i, j] = 0
        
            elif (67.5 <= tmp < 112.5) or (247.5 <= tmp < 292.5):
                if mag[i, j] < mag[i - 1, j] or mag[i, j] < mag[i + 1, j]:
                    suppressed_mag[i, j] = 0
        
            elif (112.5 <= tmp < 157.5) or (292.5 <= tmp < 337.5):
                if mag[i, j] < mag[i - 1, j + 1] or mag[i, j] < mag[i + 1, j - 1]:
                    suppressed_mag[i, j] = 0
    return suppressed_mag

def apply_suppression(img, img_name):
    mag, dir = compute_image_gradient(img)

    start_time = time.time()
    suppressed_mag = non_maximum_suppression_dir(mag, dir)
    end_time = time.time()
    print(f"Computational time for non_maximum_suppression_dir with {img_name}: {end_time - start_time:.4f} sec")

    suppressed_mag = np.clip(suppressed_mag, 0, 255).astype(np.uint8)
    plt.imshow(suppressed_mag, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    output_dir = './result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/part_2_edge_sup_{img_name}.png', bbox_inches='tight', pad_inches = 0)
    plt.show()


def main():
    lenna = im.load_lenna()
    shapes = im.load_shapes()

    # 2-1 Get filter & apply filter
    filter = imf.get_gaussian_filter_2d(7, 1.5)
    lenna_filtered = imf.cross_correlation_2d(lenna, filter)
    shapes_filtered = imf.cross_correlation_2d(shapes, filter)

    # 2-2 Report computational time show the magnitude map and store to image file
    apply_gradient_show_save(lenna_filtered, 'lenna')
    apply_gradient_show_save(shapes_filtered, 'shpaes')

    # 2-3 NMS (Non-maximum Suprression)
    apply_suppression(lenna_filtered, 'lenna')
    apply_suppression(shapes_filtered, 'shapes')



if __name__ == "__main__":
    main()