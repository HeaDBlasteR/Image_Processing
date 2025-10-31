import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from skimage.metrics import structural_similarity, mean_squared_error

# 1. Загрузите изображение в оттенках серого sar_1_gray.jpg.
# 2. постройте гистограмму
# 3. реализуйте алгоритм гамма коррекции с параметром гамма <1, >1.
# 4. Сравните исходное изображение, скорректированное при помощи гамма-фильтра. MSE, SSIM.
# 5. реализуйте алгоритм статистической цветокоррекции на основе статистики eq_gray.
# 6. Протестируйте работу алгоритмов пороговой фильтрации с различными параметрами.
# Для каждого решения - напечатайте результат


# 1
image = cv2.imread('D:\Image_Processing\Lab1\sar_1.jpg')
b = image[:, :, 0]


# 2
histSize = 256
histRange = (0, 256)
accumulate = False

b_hist = cv2.calcHist([b], [0], None, [histSize], histRange, accumulate=accumulate)

plt.plot(b_hist)
plt.show()

b_hist_cum = b_hist.cumsum()
plt.plot(b_hist_cum)
plt.show()

b_hist_norm = b_hist/(image.shape[0] * image.shape[1])
plt.plot(b_hist_norm)
plt.show()


# 3
gamma = 1.1

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v_float = v.astype(np.float32) / 255.0
v_corrected = np.power(v_float, gamma)
v = np.uint8(v_corrected * 255)
hsv_corrected = cv2.merge([h, s, v])

image_gamma_hsv = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)


# 4
(ssim, diff) = structural_similarity(image, image_gamma_hsv, full=True, channel_axis=-1)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(ssim))

plt.imshow(diff)
plt.show()

mse = mean_squared_error(image, image_gamma_hsv)
print("MSE = ", mse)


# 5
mean = image.mean()
std = image.std()
print(mean, std)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eq_gray = cv2.equalizeHist(gray)

plt.imshow(eq_gray, cmap="gray")
plt.show()

plt.imshow(gray, cmap="gray")
plt.show()

mean_src, std_src = gray.mean(), gray.std()
mean_ref, std_ref = eq_gray.mean(), eq_gray.std()

gray_corrected = (gray - mean_src) / (std_src + 1e-8) * std_ref + mean_ref
gray_corrected = np.clip(gray_corrected, 0, 255).astype(np.uint8)
plt.imshow(gray_corrected, cmap="gray")
plt.show()

# 6
for i in range(50, 250, 50):
    _, thresh1 = cv2.threshold(gray, i, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh1, cmap='gray')
    plt.show()

    result_white = (thresh1 == 255).sum()
    result_black = (thresh1 == 0).sum()
    print("w =", result_white, "b =", result_black)