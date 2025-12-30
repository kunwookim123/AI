import cv2
import numpy as np
import matplotlib.pyplot as plt

# 실습1
img = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)

kernel = kernel / kernel.sum()

result = cv2.filter2D(img, -1, kernel)

cv2.imshow("Original", img)
cv2.imshow("Filtered", result)


# 실습2
blur_methods = ['blur', 'gaussian', 'median']
kernel_sizes = [3, 7, 15]

fig, axes = plt.sublpots(3, 3, figsize=(12, 12))

for i, method in enumerate(blur_methods):
    for j, ksize in enumerate(kernel_sizes):
        if method == "blur":
            result = cv2.blur(img, (ksize, ksize))
            title = f'평균 블러 ({ksize} x {ksize})'
        elif method == 'gaussian':
            result = cv2.GaussianBlur(img, (ksize, ksize), 0)
            title = f'가우시안 블러 ({ksize} x {ksize})'
        else:
            result = cv2.medianBlur(img, ksize)
            title = f'미디안 블러 ({ksize} x {ksize})'

        axes[i, j]




cv2.waitKey(0)
cv2.destroyAllWindows()