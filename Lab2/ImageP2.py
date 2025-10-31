import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error
import copy


# Зашумить изображение при помощи шума гаусса, постоянного шума.
# Протестировать медианный фильтр, фильтр гаусса, билатериальный фильтр,
# фильтр нелокальных средних с различными параметрами.
# Выяснить, какой фильтр показал лучший результат фильтрации шума.


# Шум Гаусса
image = cv2.imread('D:\Image_Processing\Lab2\sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap="gray")
plt.title("Серое исходное изображение")
plt.show()

mean = 0
stddev = 100
noise_gauss = np.zeros(image_gray.shape, np.uint8)
cv2.randn(noise_gauss, mean, stddev)

image_noise_gauss = cv2.add(image_gray,noise_gauss)
plt.imshow(image_noise_gauss, cmap="gray")
plt.title("Добавленный Гаусс шум")
plt.show()

mse_gauss = mean_squared_error(image_gray, image_noise_gauss)
(ssim, diff) = structural_similarity(image_gray, image_noise_gauss, full=True)
print("Гаусс шум MSE, SSIM: ", mse_gauss, ssim)


image_sp_median = cv2.medianBlur(image_noise_gauss, 3)
plt.imshow(image_sp_median, cmap="gray")
plt.title("Медианный фильтр")
plt.show()

mse_sp_median = mean_squared_error(image_gray, image_sp_median)
(ssim_sp_median, diff11) = structural_similarity(image_gray, image_sp_median, full=True)
print("Медианный: ", mse_sp_median, ssim_sp_median)


image_sp_gauss = cv2.GaussianBlur(image_noise_gauss, (5, 5), 0)
plt.imshow(image_sp_gauss, cmap="gray")
plt.title("Фильтр Гаусса")
plt.show()

mse_sp_gauss = mean_squared_error(image_gray, image_sp_gauss)
(ssim_sp_gauss, diff22) = structural_similarity(image_gray, image_sp_gauss, full=True)
print("Гауссовский: ", mse_sp_gauss, ssim_sp_gauss)


image_sp_bilat = cv2.bilateralFilter(image_noise_gauss, 9, 75, 75)
plt.imshow(image_sp_bilat, cmap="gray")
plt.title("Билатериальный фильтр")
plt.show()

mse_sp_bilat = mean_squared_error(image_gray, image_sp_bilat)
(ssim_sp_bilat, diff33) = structural_similarity(image_gray, image_sp_bilat, full=True)
print("Билатериальный: ", mse_sp_bilat, ssim_sp_bilat)


image_sp_nlm = cv2.fastNlMeansDenoising(image_noise_gauss, h=20)
plt.imshow(image_sp_nlm, cmap="gray")
plt.title("Фильтр нелокальных средних")
plt.show()

mse_sp_nlm = mean_squared_error(image_gray, image_sp_nlm)
(ssim_sp_nlm, diff44) = structural_similarity(image_gray, image_sp_nlm, full=True)
print("nlm: ", mse_sp_nlm, ssim_sp_nlm)



# 2 Постоянный шум
noise_intensity = 50
uniform_noise = np.random.uniform(-noise_intensity, noise_intensity, image_gray.shape)
uniform_noise = uniform_noise.astype(np.int16)
image_uniform_noise = image_gray.astype(np.int16) + uniform_noise
image_uniform_noise = np.clip(image_uniform_noise, 0, 255).astype(np.uint8)

plt.imshow(image_uniform_noise, cmap="gray")
plt.title("Добавленный постоянный шум")
plt.show()

mse_noise = mean_squared_error(image_gray, image_uniform_noise)
(ssim, diff55) = structural_similarity(image_gray, image_uniform_noise, full=True)
print("Постоянный шум MSE, SSIM: ", mse_noise, ssim)


image_noise_median = cv2.medianBlur(image_uniform_noise, 3)
plt.imshow(image_noise_median, cmap="gray")
plt.title("Медианный фильтр")
plt.show()

mse_noise_median = mean_squared_error(image_gray, image_noise_median)
(ssim_noise_median, diff66) = structural_similarity(image_gray, image_noise_median, full=True)
print("Медианный: ", mse_noise_median, ssim_noise_median)


image_noise_gauss = cv2.GaussianBlur(image_uniform_noise, (5, 5), 0)
plt.imshow(image_noise_gauss, cmap="gray")
plt.title("Фильтр Гаусса")
plt.show()

mse_noise_gauss = mean_squared_error(image_gray, image_noise_gauss)
(ssim_noise_gauss, diff77) = structural_similarity(image_gray, image_noise_gauss, full=True)
print("Гауссовский: ", mse_noise_gauss, ssim_noise_gauss)


image_noise_bilat = cv2.bilateralFilter(image_uniform_noise, 9, 75, 75)
plt.imshow(image_noise_bilat, cmap="gray")
plt.title("Билатеральный фильтр")
plt.show()

mse_noise_bilat = mean_squared_error(image_gray, image_noise_bilat)
(ssim_noise_bilat, diff88) = structural_similarity(image_gray, image_noise_bilat, full=True)
print("Билатериальный: ", mse_noise_bilat, ssim_noise_bilat)


image_noise_nlm = cv2.fastNlMeansDenoising(image_uniform_noise, h=20)
plt.imshow(image_noise_nlm, cmap="gray")
plt.title("Фильтр нелокальных средних")
plt.show()

mse_noise_nlm = mean_squared_error(image_gray, image_noise_nlm)
(ssim_noise_nlm, diff99) = structural_similarity(image_gray, image_noise_nlm, full=True)
print("nlm: ", mse_noise_nlm, ssim_noise_nlm)