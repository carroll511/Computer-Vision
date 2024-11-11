import cv2

def load_lenna():
    lenna = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    return lenna

def load_shapes():
    shapes = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
    return shapes
