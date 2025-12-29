import cv2
import numpy as np

# 체크보드
h, w  = 500, 500

square = 50

y = np.arange(h) // square
x = np.arange(w) // square

# Numpy 출력 생략 끄기
np.set_printoptions(threshold=np.inf)

board = (y[:,None] + x[None, :]) % 2

checkerboard = (board * 255).astype(np.uint8)