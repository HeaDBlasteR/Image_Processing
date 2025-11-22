# 1. Выполните сохранение монохромного изображения в виде текстового или бинарного файла.
# 2. Реализуйте алгоритм вейвлет-преобразования Хаара для изображения.
# 3. Выполните квантование высокочастотных компонент (прим., количество квантов = 4).
# 4. Сохраните получившийся массив значений в текстовый или бинарный файл в порядке LL, LH, HL, HH
# вейвлет-преобразования Хафа. Компоненты LH, HL, HH храните в виде пар (значение, количество повторений).
# 5. Сравните объем памяти, занимаемый исходным изображением (попиксельное хранение), и изображение, полученным после преобразования Хафа и сжатием длин серий.


import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt

# 1
image_path = 'D:\Image_Processing\Lab6\sar_1_gray.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

img = image.astype(np.float32)
np.savetxt('mono_image.txt', img, fmt="%d")

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Исходное изображение')
plt.xticks([]), plt.yticks([])

# 2
def haar_transform(image):
    h, w = image.shape
    row_transform = np.zeros_like(image, dtype=np.float32)
    for i in range(h):
        for j in range(0, w - 1, 2):
            row_transform[i, j // 2] = (image[i, j] + image[i, j + 1]) / 2
            row_transform[i, (j // 2) + (w // 2)] = (image[i, j] - image[i, j + 1]) / 2

    result = np.zeros_like(row_transform, dtype=np.float32)
    for j in range(w):
        for i in range(0, h - 1, 2):
            result[i // 2, j] = (row_transform[i, j] + row_transform[i + 1, j]) / 2
            result[(i // 2) + (h // 2), j] = (row_transform[i, j] - row_transform[i + 1, j]) / 2

    ll = result[:h // 2, :w // 2]
    hl = result[h // 2:, :w // 2]
    lh = result[:h // 2, w // 2:]
    hh = result[h // 2:, w // 2:]

    return ll, lh, hl, hh

ll, lh, hl, hh = haar_transform(img)

plt.subplot(122)

h, w = img.shape
display_img = np.zeros((h, w), dtype=np.float32)
display_img[:h//2, :w//2] = ll
display_img[:h//2, w//2:] = lh
display_img[h//2:, :w//2] = hl
display_img[h//2:, w//2:] = hh
plt.imshow(np.abs(display_img), cmap='gray')
plt.title('Вейвлет-преобразование Хаара (LL, LH, HL, HH)')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(figsize=(15, 10))
plt.subplot(221)
plt.imshow(ll, cmap='gray')
plt.title('LL компонента (низкие частоты)')
plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(lh, cmap='gray')
plt.title('LH компонента (вертикальные детали)')
plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(hl, cmap='gray')
plt.title('HL компонента (горизонтальные детали)')
plt.xticks([]), plt.yticks([])

plt.subplot(224)
plt.imshow(hh, cmap='gray')
plt.title('HH компонента (диагональные детали)')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

n_quants = 4

# 3
def quantize(coeffs, n_quants):
    min_val = np.min(coeffs)
    max_val = np.max(coeffs)
    step = (max_val - min_val) / n_quants
    quantized = np.round((coeffs - min_val) / step).astype(int)
    return quantized

lh_q = quantize(lh, n_quants)
hl_q = quantize(hl, n_quants)
hh_q = quantize(hh, n_quants)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(lh_q, cmap='gray')
plt.title('Квантованная LH (n_quants=4)')
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(hl_q, cmap='gray')
plt.title('Квантованная HL (n_quants=4)')
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(hh_q, cmap='gray')
plt.title('Квантованная HH (n_quants=4)')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

# 4
def run_length_encode(data):
    encoded = []
    for value, count in Counter(data.flatten()).items():
        encoded.append((value, count))
    return encoded

lh_rle = run_length_encode(lh_q)
hl_rle = run_length_encode(hl_q)
hh_rle = run_length_encode(hh_q)

with open('wvlt_data.txt', 'w') as f:
    np.savetxt(f, ll, fmt='%d')
    f.write('\nLH:\n')
    for value, count in lh_rle:
        f.write(f"{value} {count}\n")
    f.write('\nHL:\n')
    for value, count in hl_rle:
        f.write(f"{value} {count}\n")
    f.write('\nHH:\n')
    for value, count in hh_rle:
        f.write(f"{value} {count}\n")

# 5
with open('mono_image.txt', 'r') as f:
    original_size = len(f.read().encode('utf-8'))

with open('wvlt_data.txt', 'r') as f:
    compressed_size = len(f.read().encode('utf-8'))

print(f"Размер исходного изображения: {original_size} байт")
print(f"Размер сжатого изображения: {compressed_size} байт")
print(f"Коэф. сжатия: {original_size / compressed_size:.2f}")

print(f"\nИнформация о RLE сжатии:")
print(f"LH: {len(lh_rle)} уникальных значений (исходно: {lh_q.size} элементов)")
print(f"HL: {len(hl_rle)} уникальных значений (исходно: {hl_q.size} элементов)")
print(f"HH: {len(hh_rle)} уникальных значений (исходно: {hh_q.size} элементов)")