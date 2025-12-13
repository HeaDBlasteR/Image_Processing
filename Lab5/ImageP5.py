import numpy as np
import cv2 as cv
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt

# Реализуйте высокочастотную фильтрацию на основе ядра Гаусса
# Реализуйте удаление периодического шума

# Высокочастотный фильтр
img = cv.imread('D:\Image_Processing\Lab5\periodic_noise.jpg', cv.IMREAD_GRAYSCALE)

dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1))

sigma = 50  # степень сглаживания
gaussian_lowpass = np.exp(-((x - (cols // 2))**2 + (y - (rows // 2))**2) / (2 * sigma**2))
gaussian_highpass = 1 - gaussian_lowpass  # из низкочастотного в высокочастотный

dft_shift = dft_shift * gaussian_highpass[:, :, np.newaxis]
img_back = cv.idft(np.fft.ifftshift(dft_shift))
img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])

plt.figure(figsize = (10, 5))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Оригинал'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Модифицированное (высокочастотная)'), plt.xticks([]), plt.yticks([])
plt.show()



# Удаление периодического шума
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
plt.figure(figsize=(10,5))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Спектр с шумом")
plt.show()

local_max = peak_local_max(magnitude_spectrum, min_distance=15, threshold_abs=200)
rows, cols = img.shape
mask = np.copy(dft_shift)
mask[:, :, :] = 1

for i in local_max:
    r = np.int32(((i[0] - rows // 2)**2 + (i[1] - cols // 2)**2)**(1 / 2))
    cv.circle(mask, (rows // 2, cols // 2), r, (0,0))

fshift = dft_shift * mask

# Вывод спектра после удаления шума
magnitude_spectrum_cleaned = 20*np.log(cv.magnitude(fshift[:,:,0], fshift[:,:,1]))
plt.figure(figsize=(10,5))
plt.imshow(magnitude_spectrum_cleaned, cmap='gray')
plt.title("Спектр после удаления периодического шума")
plt.show()

img_back = cv.idft(np.fft.ifftshift(fshift))
img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])

plt.figure(figsize = (15, 6))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Оригинал'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Модифицированное (удаление периодического шума)'), plt.xticks([]), plt.yticks([])
plt.show()