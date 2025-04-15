import cv2
import numpy as np
def binaied(path):
    img = cv2.imread(path)
    #img = cv2.resize(img, (700, 700))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.bilateralFilter(img, 11, 15, 15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    alpha = 1.5  # Коэффициент контраста (>1 увеличивает контраст)
    beta = 0  # Смещение (изменяет яркость)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    binary = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
    karnel = np.ones((4, 4), np.uint8)
    binary = cv2.dilate(binary, karnel, iterations=1)
    return binary

binary1 = binaied('NVIDIA_Overlay_SL3FNLo3yC.png')
cv2.imwrite('binary_output.png', binary1)  # Save the image
cv2.waitKey(0)