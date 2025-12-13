import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, jaccard_score
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def analyze_intensity(gray_img, num_bins=256):
    hist = cv2.calcHist([gray_img], [0], None, [num_bins], [0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()

    values = np.arange(num_bins)
    stats_results = {}

    stats_results['mean'] = np.sum(hist * values)

    variance = np.sum(hist * (values - stats_results['mean']) ** 2)
    stats_results['std'] = np.sqrt(variance)

    if stats_results['std'] > 0:
        stats_results['skewness'] = np.sum(
            hist * ((values - stats_results['mean']) / stats_results['std']) ** 3)
    else:
        stats_results['skewness'] = 0

    if stats_results['std'] > 0:
        stats_results['kurtosis'] = np.sum(
            hist * ((values - stats_results['mean']) / stats_results['std']) ** 4) - 3
    else:
        stats_results['kurtosis'] = -3

    non_zero_probs = hist[hist > 0]
    stats_results['entropy'] = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    cmlt_hist = np.cumsum(hist)
    stats_results['q1'] = np.argmax(cmlt_hist >= 0.25)
    stats_results['q2'] = np.argmax(cmlt_hist >= 0.5)
    stats_results['q3'] = np.argmax(cmlt_hist >= 0.75)

    occupied_bins = np.where(hist > 0)[0]
    if len(occupied_bins) > 0:
        stats_results['min'] = occupied_bins[0]
        stats_results['max'] = occupied_bins[-1]
    else:
        stats_results['min'] = 0
        stats_results['max'] = 255

    return stats_results, hist

def compute_texture_feat(gray_img, num_bins=256):
    stats, hist = analyze_intensity(gray_img, num_bins)

    feat = [
        stats['mean'],
        stats['std'],
        stats['skewness'],
        stats['kurtosis'],
        stats['entropy'],
        stats['q1'],
        stats['q2'],
        stats['q3'],
        stats['min'],
        stats['max']
    ]

    return np.array(feat)

def dataset_setup(img_dir):
    features_list = []
    categories_list = []

    for category_name in os.listdir(img_dir):
        category_path = os.path.join(img_dir, category_name)

        if os.path.isdir(category_path) and not category_name.startswith('.'):
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_path, img_file)

                    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_data is not None:
                        feature_vector = compute_texture_feat(img_data)
                        features_list.append(feature_vector)
                        categories_list.append(category_name)

    return np.array(features_list), np.array(categories_list)


def visualize_samples(imgs_dir, samples_num):
    categories = [f for f in os.listdir(imgs_dir)]

    fig, axes = plt.subplots(samples_num, 3, figsize=(15, 5 * samples_num))

    for i in range(min(samples_num, len(categories))):
        category = categories[i]
        category_path = os.path.join(imgs_dir, category)

        image_files = [f for f in os.listdir(category_path)]

        sample_path = os.path.join(category_path, image_files[4])
        sample_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

        stats, intensity_plot = analyze_intensity(sample_img)
        axes[i, 0].imshow(sample_img, cmap='gray')
        axes[i, 0].set_title(f'{category}')
        axes[i, 0].axis('off')

        axes[i, 1].bar(range(256), intensity_plot, alpha=0.7, color='purple')
        axes[i, 1].set_title('Гистограмма')

        stats_text = (
            f"Mean: {stats['mean']:.2f}\n"
            f"Std: {stats['std']:.2f}\n"
            f"Skewness: {stats['skewness']:.2f}\n"
            f"Kurtosis: {stats['kurtosis']:.2f}\n"
            f"Entropy: {stats['entropy']:.2f}\n"
            f"Min/Max: {stats['min']}/{stats['max']}"
        )

        axes[i, 2].text(0.1, 0.9, stats_text, transform=axes[i, 2].transAxes,
                          fontsize=10, verticalalignment='top')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()



img_dir = "D:\\Image_Processing\\Lab10\\10_texture_analysis\\KTH_TIPS"
features_data, target_labels = dataset_setup(img_dir)

print(f"Размер данных: {features_data.shape}")
print(f"Количество категорий: {len(np.unique(target_labels))}")

visualize_samples(img_dir, 3)



def laws_features(img_data, normalize=True):
    L5 = np.array([1, 4, 6, 4, 1])
    E5 = np.array([-1, -2, 0, 2, 1])
    S5 = np.array([-1, 0, 2, 0, -1])
    W5 = np.array([-1, 2, 0, -2, 1])
    R5 = np.array([1, -4, 6, -4, 1])

    kernels_1d = [L5, E5, S5, W5, R5]
    kernel_names = ['L5', 'E5', 'S5', 'W5', 'R5']

    filters_2d = []
    filter_names = []

    for i, kernel_x in enumerate(kernels_1d):
        for j, kernel_y in enumerate(kernels_1d):
            filter_kernel = np.outer(kernel_x, kernel_y)
            filters_2d.append(filter_kernel)
            filter_names.append(f"{kernel_names[i]}{kernel_names[j]}")

    feature_vector = []
    feature_names = []

    for kernel, name in zip(filters_2d, filter_names):
        filtered_img = cv2.filter2D(img_data.astype(np.float32), -1, kernel)
        texture_energy = np.mean(filtered_img ** 2)
        feature_vector.append(texture_energy)
        feature_names.append(f"laws_{name}")

    feature_vector = np.array(feature_vector)

    if normalize and np.sum(feature_vector) > 0:
        feature_vector = feature_vector / np.sum(feature_vector)

    return feature_vector, feature_names


def laws_categories(imgs_dir, samples_num):
    texture_categories = [f for f in os.listdir(imgs_dir)]

    fig, axes = plt.subplots(samples_num, 2, figsize=(12, 4 * samples_num))

    for i in range(min(samples_num, len(texture_categories))):
        texture_name = texture_categories[i]
        texture_path = os.path.join(imgs_dir, texture_name)

        img_files = [f for f in os.listdir(texture_path)]

        img_path = os.path.join(texture_path, img_files[4])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        features, names = laws_features(img)

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'{texture_name}')
        axes[i, 0].axis('off')

        sort_indices = np.argsort(features)[-7:][::-1]
        laws_text = "Топ характеристик:\n\n"
        for j, idx in enumerate(sort_indices):
            laws_text += f"{names[idx]}: {features[idx]:.4f}\n"

        axes[i, 1].text(0.1, 0.95, laws_text, transform=axes[i, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[i, 1].set_title('Laws features')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


laws_categories(img_dir, 3)


def calc_glcm_feat(img_data, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    img_data = (img_data // 8).astype(np.uint8)

    glcm_matrix = graycomatrix(img_data, distances=distances, angles=angles,
                               levels=32, symmetric=True, normed=True)

    feat = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    for prop in properties:
        feat_value = np.mean(graycoprops(glcm_matrix, prop))
        feat.append(feat_value)

    return np.array(feat)


def glcm_categories(imgs_dir, samples_num):
    texture_categories = [f for f in os.listdir(imgs_dir)]

    fig, axes = plt.subplots(samples_num, 2, figsize=(12, 4 * samples_num))

    for i in range(min(samples_num, len(texture_categories))):
        texture_name = texture_categories[i]
        texture_path = os.path.join(imgs_dir, texture_name)

        img_files = [f for f in os.listdir(texture_path)]

        img_path = os.path.join(texture_path, img_files[4])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        features = calc_glcm_feat(img)

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'{texture_name}')
        axes[i, 0].axis('off')

        glcm_names = ['contrast', 'homogeneity', 'energy', 'entropy']
        glcm_text = "GLCM характеристики:\n\n"
        for name, value in zip(glcm_names, features):
            glcm_text += f"{name}: {value:.4f}\n"

        axes[i, 1].text(0.1, 0.95, glcm_text, transform=axes[i, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[i, 1].set_title('GLCM features')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


glcm_categories(img_dir, 3)



image_paths = []
for category_dir in os.listdir(img_dir):
    category_path = os.path.join(img_dir, category_dir)
    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(category_path, file))

print(f"Количество изображений: {len(image_paths)}")


histogram_feat = []
laws_feat = []
glcm_feat = []
img_categories = []

for i, path in enumerate(image_paths):
    if i % 100 == 0:
        print(f"Обработано: {i}/{len(image_paths)}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    hist_feats = compute_texture_feat(img)
    laws_feats, _ = laws_features(img)
    glcm_feats = calc_glcm_feat(img)

    histogram_feat.append(hist_feats)
    laws_feat.append(laws_feats)
    glcm_feat.append(glcm_feats)

    category = os.path.basename(os.path.dirname(path))
    img_categories.append(category)

histogram_feat = np.array(histogram_feat)
laws_feat = np.array(laws_feat)
glcm_feat = np.array(glcm_feat)
img_categories = np.array(img_categories)

print(f"Размер данных: {histogram_feat.shape}")
print(f"Количество категорий: {len(np.unique(img_categories))}")

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(img_categories)

feat_sets = {
    'Histogram': histogram_feat,
    'Laws': laws_feat,
    'GLCM': glcm_feat
}

classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Trees': DecisionTreeClassifier(random_state=42, max_depth=10)
}


trained_classifiers = {}

for feat_name, feat_matrix in feat_sets.items():
    print(f"\n{feat_name.upper()}")

    X_train, X_test, y_train, y_test = train_test_split(
        feat_matrix, encoded_labels, test_size=0.25, random_state=42, stratify=encoded_labels)

    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    current_models = {}

    for clsf_name, clsf_template in classifiers.items():
        if clsf_name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        elif clsf_name == 'SVM':
            model = SVC(kernel='rbf', random_state=42)
        elif clsf_name == 'Trees':
            model = DecisionTreeClassifier(random_state=42, max_depth=10)

        if clsf_name == 'SVM':
            model.fit(X_train_scale, y_train)
            predictions = model.predict(X_test_scale)
            current_models[clsf_name] = (model, scaler)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            current_models[clsf_name] = model

        print(f"{clsf_name}")
        print(classification_report(y_test, predictions, target_names=label_encoder.classes_))
        print()

    trained_classifiers[feat_name] = current_models



selected_models = {
    'Histogram_KNN': trained_classifiers['Histogram']['KNN'],
    'Laws_Trees': trained_classifiers['Laws']['Trees'],
    'Laws_SVM': trained_classifiers['Laws']['SVM']
}


def texture_segmentation(model_data, model_id, img_path, window_size=32, step=16, num_classes=3):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape
    segmentation_result = np.zeros((height, width), dtype=np.uint8)

    all_predictions = []
    positions = []

    for y in range(0, height - window_size + 1, step):
        for x in range(0, width - window_size + 1, step):
            img_patch = img[y:y + window_size, x:x + window_size]

            if 'Histogram' in model_id:
                patch_feats = compute_texture_feat(img_patch)
            elif 'Laws' in model_id:
                patch_feats, _ = laws_features(img_patch)
            elif 'GLCM' in model_id:
                patch_feats = calc_glcm_feat(img_patch)

            if isinstance(model_data, tuple):
                classifier, feat_scaler = model_data
                feats_scale = feat_scaler.transform(patch_feats.reshape(1, -1))
                predicted_class = classifier.predict(feats_scale)[0]
            else:
                classifier = model_data
                predicted_class = classifier.predict(patch_feats.reshape(1, -1))[0]

            segmentation_result[y:y + window_size, x:x + window_size] = predicted_class
            all_predictions.append(predicted_class)
            positions.append((y, x))

    unique_classes, class_counts = np.unique(all_predictions, return_counts=True)

    top_classes = unique_classes[np.argsort(class_counts)[-num_classes:]]

    filtered_segmentation = np.zeros_like(segmentation_result)

    for i, class_id in enumerate(top_classes):
        filtered_segmentation[segmentation_result == class_id] = i

    remaining_pixels = ~np.isin(segmentation_result, top_classes)
    if np.any(remaining_pixels):
        filtered_segmentation[remaining_pixels] = 1

    print(f"Классы после фильтрации: {np.unique(filtered_segmentation)}")

    return filtered_segmentation



target_image_path = "D:\\Image_Processing\\Lab10\\origin_bread.png"
input_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

if input_image is not None:
    segmentation_results = []
    model_ids = []

    for model_id, model_data in selected_models.items():
        print(f"Модель: {model_id}")

        segmentation_map = texture_segmentation(model_data, model_id, target_image_path,
                                                        window_size=16, step=8)

        if segmentation_map is not None:
            segmentation_results.append(segmentation_map)
            model_ids.append(model_id)
            print(f"Успешно")
        else:
            print(f"Ошибка обработки")

    if segmentation_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, (seg_map, model_id) in enumerate(zip(segmentation_results, model_ids)):
            if 'Histogram' in model_id:
                method_name = 'Histogram'
                classifier_type = 'KNN'
            elif 'Laws' in model_id:
                method_name = 'Laws'
                if 'Trees' in model_id:
                    classifier_type = 'Trees'
                else:
                    classifier_type = 'SVM'

            axes[i].imshow(input_image, cmap='gray', alpha=0.7)
            axes[i].imshow(seg_map, cmap='tab10', alpha=0.5)
            axes[i].set_title(f'{method_name} + {classifier_type}', fontsize=12)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Успешно моделей обработано: {len(segmentation_results)}/3")
    else:
        print("Ошибка обработки")



def simplify_color_scheme(mask_array, max_colors=3):
    unique, counts = np.unique(mask_array, return_counts=True)

    sorted_colors = np.argsort(-counts)
    dominant_colors = unique[sorted_colors[:max_colors]]

    print(f"Частые цвета: {dominant_colors}")
    print(f"Количество пикселей: {counts[sorted_colors[:max_colors]]}")

    result = np.zeros_like(mask_array)

    for new_color, original_color in enumerate(dominant_colors):
        result[mask_array == original_color] = new_color

    print(f"Установлено: {dominant_colors} -> {list(range(len(dominant_colors)))}")

    return result


manual_mask = cv2.imread("D:\\Image_Processing\\Lab10\\bread.png", cv2.IMREAD_GRAYSCALE)
fixed_manual_mask = simplify_color_scheme(manual_mask, 3)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(manual_mask, cmap='gray')
plt.title(f'Изначальная маска\n{len(np.unique(manual_mask))} значений')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(fixed_manual_mask, cmap='tab10')
plt.title('Исправленная маска\n3 класса')
plt.colorbar()

plt.tight_layout()
plt.show()


def compare_simple(manual_mask, predicted_mask, original_img):
    manual_classes = np.unique(manual_mask)
    pred_classes = np.unique(predicted_mask)

    print(f"Ручные классы: {manual_classes}")
    print(f"Классы при предсказании: {pred_classes}")

    class_mapping = {0: 2, 1: 1, 2: 0}

    pred_mapped = predicted_mask.copy()
    for pred_class, manual_class in class_mapping.items():
        pred_mapped[predicted_mask == pred_class] = manual_class

    accuracy = accuracy_score(manual_mask.flatten(), pred_mapped.flatten())

    print(f"\nТочность: {accuracy:.1%}")
    print(f"Классовые IoU:")

    for class_id in manual_classes:
        iou = jaccard_score(manual_mask.flatten(), pred_mapped.flatten(),
                            average=None, labels=[class_id])[0]
        print(f"Класс {class_id}: {iou:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')

    axes[1].imshow(manual_mask, cmap='tab10')
    axes[1].set_title('Ручная разметка')
    axes[1].axis('off')

    axes[2].imshow(pred_mapped, cmap='tab10')
    axes[2].set_title(f'Предсказанная разметка')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return accuracy


model_name = 'Histogram_KNN'
model_info = selected_models[model_name]

print("-------------------------------------------")
predicted_mask = texture_segmentation(
    model_info, model_name, target_image_path,
    32, 4)

print(f"Предсказание размер: {predicted_mask.shape}")
print(f"Классы в предсказании: {np.unique(predicted_mask)}")

if fixed_manual_mask is not None and predicted_mask is not None and input_image is not None:
    accuracy = compare_simple(fixed_manual_mask, predicted_mask, input_image)
    print(f"В итоге верных пикселей: {accuracy*100:.1f}%")