import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(image, title: str = 'Гистограмма яркости'):
    plt.figure(figsize=(8, 4))
    plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
    plt.title(title)
    plt.xlabel('Яркость (0-255)')
    plt.ylabel('Количество пикселей')
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.show()

def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Параметр гамма должен быть > 0")
    lut = np.clip((np.linspace(0, 1, 256) ** gamma) * 255 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)

def apply_stat_color_correction(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    eq_gray = cv2.equalizeHist(gray)
    m_src, s_src = float(gray.mean()), float(gray.std() + 1e-6)
    m_eq, s_eq = float(eq_gray.mean()), float(eq_gray.std() + 1e-6)
    scale = s_eq / s_src
    shift = m_eq - m_src * scale
    channels = cv2.split(image_bgr)
    corrected = []
    for ch in channels:
        f = ch.astype(np.float32)
        f = f * scale + shift
        f = np.clip(f, 0, 255).astype(np.uint8)
        corrected.append(f)
    corrected_bgr = cv2.merge(corrected)
    return corrected_bgr

def main():
    possible_paths = [
        'D:/ImageP/Lab1/sar_1_gray.jpg'
    ]
    
    image = None
    image_path = None
    
    for path in possible_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_path = path
            print(f"Изображение найдено: {path}")
            break
    
    if image is None:
        print("Файл sar_1_gray.png не найден")
    
    plot_histogram(image, 'Гистограмма (оригинал)')
    
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    
    for gamma in gamma_values:
        corrected = apply_gamma(image, gamma)
        plot_histogram(corrected, f'Гистограмма (gamma={gamma})')
        
        output_path = f"gamma_{gamma}.png"
        cv2.imwrite(output_path, corrected)
        print(f"Сохранено: {output_path}")
    
    color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_img is not None:
        corrected_color = apply_stat_color_correction(color_img)
        
        plot_histogram(cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY), 
                      'Гистограмма (исходное цветное -> gray)')
        plot_histogram(cv2.cvtColor(corrected_color, cv2.COLOR_BGR2GRAY), 
                      'Гистограмма (stat корректированное -> gray)')
        
        cv2.imwrite("stat_corrected.png", corrected_color)
        print("Сохранено: stat_corrected.png")
    else:
        print("Цветное изображение не найдено, пропускаем статистическую коррекцию")

if __name__ == '__main__':
    main()