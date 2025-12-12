import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.neighbors import KNeighborsClassifier

data_pth = 'D:\\Image_Processing\\Lab9\\09_image_date_in_data_analysis\\data'

class CharacterImageCreator:
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.character_mapping = {
            '(': '(', ')': ')',
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
            't': 't', 'x': 'X', 'y': 'y', 'h': 'h', 'w': 'w', 
            ',': ',', '+': '+', '-': '-', '*': 'times'
        }
        self.supported_characters = list(self.character_mapping.keys())
        print("Поддерживаемые символы:", self.supported_characters)

    def fetch_character_image(self, character):
        folder_name = self.character_mapping[character]
        character_folder = os.path.join(self.dataset_directory, folder_name)

        image_files = [f for f in os.listdir(character_folder) if os.path.isfile(os.path.join(character_folder, f))]
        image_files.sort()

        # Обучение модели на 85% данных / тестировка на 15%
        split_index = int(len(image_files) * 0.85)
        validation_images = image_files[split_index:]

        selected_image = random.choice(validation_images)
        image_path = os.path.join(character_folder, selected_image)

        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image_data

    def create_character_chain(self, chain_length=5):
        character_chain = ''.join(random.choices(self.supported_characters, k=chain_length))
        print(f"Создание цепочки: {character_chain}")
        character_images = []

        for char in character_chain:
            char_image = self.fetch_character_image(char)
            if char_image is not None:
                character_images.append(char_image)

        return character_chain, character_images

# Комбинирование тестовых изображений
def combine_images_horizontally(image_list, gap=20):
    total_width = sum(img.shape[1] for img in image_list) + gap * (len(image_list) - 1)
    max_height = max(img.shape[0] for img in image_list)

    combined_image = np.ones((max_height, total_width), dtype=np.uint8) * 255

    current_x = 0
    for img in image_list:
        height, width = img.shape
        y_start = (max_height - height)
        combined_image[y_start:y_start + height, current_x:current_x + width] = img
        current_x += width + gap

    return combined_image

def prepare_training_data(data_pth, character_list, training_ratio=0.85):
    training_features, training_labels = [], []

    for character in character_list:
        char_path = os.path.join(data_pth, character)
        image_list = [f for f in os.listdir(char_path)
                      if os.path.isfile(os.path.join(char_path, f))]
        image_list.sort()

        split_point = int(len(image_list) * training_ratio)
        training_images = image_list[:split_point]

        for img_file in training_images:
            img_path = os.path.join(char_path, img_file)
            img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_data is not None:
                img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img_data, (45, 45))
                training_features.append(img_resized.flatten())
                training_labels.append(character)

    return np.array(training_features), np.array(training_labels)

def classify_character(model, input_image):
    target_size = (45, 45)
    resized_image = cv2.resize(input_image, target_size)
    processed_image = cv2.bitwise_not(resized_image)
    flattened_image = processed_image.flatten()

    prediction_result = model.predict([flattened_image])
    return prediction_result[0]


image_creator = CharacterImageCreator(data_pth)
seq_num = 3
seq_len = 5
generated_sequences = []
generated_image_sets = []

for idx in range(seq_num):
    sequence, images = image_creator.create_character_chain(chain_length=seq_len)
    generated_sequences.append(sequence)
    generated_image_sets.append(images)

final_combined_images = []
for image_set in generated_image_sets:
    combined_img = combine_images_horizontally(image_set, gap=30)
    final_combined_images.append(combined_img)


figure, axes = plt.subplots(seq_num, 1, figsize=(8, 10))

for idx, (image_data, sequence_text) in enumerate(zip(final_combined_images, generated_sequences)):
    axes[idx].imshow(image_data, cmap='gray')
    axes[idx].set_title(f'Цепочка {idx+1}: "{sequence_text}"')
    axes[idx].axis('off')
plt.tight_layout()
plt.show()

# Выделение объектов/символов
processed_sequences_data = []

print("\n")
for img_index, source_image in enumerate(final_combined_images):
    print(f"Обработка цепочки {img_index + 1}: '{generated_sequences[img_index]}'")

    # Предобработка
    processed_image = source_image.copy()
    structuring_element = np.ones((3, 3), np.uint8)
    processed_image = cv2.erode(processed_image, structuring_element, iterations=1)
    processed_image = cv2.bitwise_not(processed_image)

    base_dimensions = processed_image.shape[0], processed_image.shape[1]
    base_image = np.zeros(base_dimensions, dtype=np.uint8)
    base_image[0:processed_image.shape[0], 0:processed_image.shape[1]] = processed_image
    base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)

    rgb_image = base_image

    plt.figure(figsize=(10, 3))
    plt.imshow(rgb_image)
    plt.title(f'Цепочка {img_index + 1}: "{generated_sequences[img_index]}" (предобработанное)')
    plt.show()

    # Бинаризация
    threshold_value, binary_image = cv2.threshold(rgb_image, 127, 255, 0)
    print(f"Размер бинарного изображения: {binary_image.shape}")

    plt.figure(figsize=(10, 3))
    binary_image = np.uint8(binary_image)
    plt.imshow(binary_image)
    plt.title(f'Цепочка {img_index + 1} (бинарное)')
    plt.show()

    detected_contours, hierarchy = cv2.findContours(binary_image[:, :, 0],
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    character_bboxes = []
    detected_contours = sorted(detected_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    for current_contour in detected_contours:
        contour_area = cv2.contourArea(current_contour)
        x_pos, y_pos, width, height = cv2.boundingRect(current_contour)

        while (width > 45):
            width -= 1

        if width * height > 50:
            cv2.rectangle(rgb_image, (x_pos, y_pos), (x_pos + width, y_pos + height),
                          (203, 192, 255), 1)
            character_bboxes.append([x_pos, y_pos, width, height])

    plt.figure(figsize=(10, 3))
    plt.imshow(rgb_image)
    plt.title(f'Цепочка {img_index + 1} (найденные символы)')
    plt.show()

    extracted_characters = []
    num_chars = len(character_bboxes)

    if num_chars > 0:
        fig, axes = plt.subplots(1, num_chars, figsize=(15, 3))
        if num_chars == 1:
            axes = [axes]

        for bbox_idx, bbox in enumerate(character_bboxes):
            x_pos, y_pos, width, height = bbox
            char_image = processed_image[y_pos:y_pos + height, x_pos:x_pos + width]
            char_data = char_image.copy()

            axes[bbox_idx].imshow(char_data, cmap="gray")
            axes[bbox_idx].set_title(f'Символ {bbox_idx + 1}')
            axes[bbox_idx].axis('off')
            extracted_characters.append((x_pos, char_data))

        plt.tight_layout()
        plt.show()

    processed_sequences_data.append({
        'sequence_text': generated_sequences[img_index],
        'characters': extracted_characters
    })

    print(f"Обработка завершена; Количество найденных символов: {len(extracted_characters)}\n")

print("-" * 30)
print("Цепочки после обработки:")
for seq_idx, sequence_info in enumerate(processed_sequences_data):
    print(f"{seq_idx + 1}) '{sequence_info['sequence_text']}' / {len(sequence_info['characters'])} символа(-ов)")
print("-" * 30)


character_set = ['(', ')', '0','1','2','3','4','5','6','7','8','9',',',
                '+','-','times','t','X','y','h','w']

# Обучающие данные
X_training, y_training = prepare_training_data(data_pth, character_set)
print(f"Обучающие данные - / {len(X_training)} / изображений\n")

# Тренировки разных K-nn
neighbor_counts = [1, 3, 5, 7]
knn_models = {}

for k_val in neighbor_counts:
    classifier = KNeighborsClassifier(n_neighbors=k_val)
    classifier.fit(X_training, y_training)
    knn_models[k_val] = classifier
    print(f"Обучение модели knn с k={k_val} завершено")

# Сравнение моделей
performance_results = {}

for k_val, model in knn_models.items():
    print(f"\nМодель при k = {k_val}")

    correct_predictions = 0
    total_characters = 0
    sequence_success = []

    for seq_idx, sequence_info in enumerate(processed_sequences_data):
        original_sequence = sequence_info['sequence_text']
        character_data = sequence_info['characters']

        print(f"\nЦепочка {seq_idx + 1}:")
        print(f"Изначальные символы: '{original_sequence}'")

        predicted_chars = []
        for char_idx, (x_coord, char_img) in enumerate(character_data):
            resized_char = cv2.resize(char_img, (45, 45))
            predicted_char = classify_character(model, resized_char)
            predicted_chars.append(predicted_char)

        predicted_sequence = ''.join(predicted_chars)
        print(f"Найдены символы: '{predicted_sequence}'")

        # Подсчет точности
        correct_count = sum(1 for orig, pred in zip(original_sequence, predicted_sequence)
                            if orig == pred)
        sequence_accuracy = correct_count / len(original_sequence)
        correct_predictions += correct_count
        total_characters += len(original_sequence)

        if original_sequence == predicted_sequence:
            print("Результат - 100% корректность")
            sequence_success.append(1.0)
        else:
            print("Результат - найдены ошибки")
            sequence_success.append(0.0)
            for pos, (orig_char, pred_char) in enumerate(zip(original_sequence, predicted_sequence)):
                if orig_char != pred_char:
                    print(f"Ошибка в символе {pos + 1}: исходное - '{orig_char}', найдено - '{pred_char}'")

    overall_accuracy = correct_predictions / total_characters if total_characters > 0 else 0
    performance_results[k_val] = {
        'accuracy': overall_accuracy,
        'sequence_success_rate': np.mean(sequence_success),
        'correct_chars': correct_predictions,
        'total_chars': total_characters
    }

    print("\n" + "-" * 30)
    print(f"Результаты при k = {k_val}:")
    print(f"Верно найденных символов: {correct_predictions} из {total_characters}")
    print(f"Точность классификации: {overall_accuracy:.1%}")
    print(f"Количество идеальных цепочек: {int(sum(sequence_success))} из {len(sequence_success)}")
    print("-" * 30)

print("\n")
opt_k = None
highest_acc = 0

for k_val, result in sorted(performance_results.items()):
    accuracy = result['accuracy']
    print(f"k = {k_val}: Точность классификации {accuracy:.1%} ({result['correct_chars']} из {result['total_chars']} символов)")
    if accuracy > highest_acc:
        highest_acc = accuracy
        opt_k = k_val

print(f"\nИтог: Наилучший результат при k = {opt_k} при точности нахождения {highest_acc:.1%}")
print(f"Данная задача эффективно решается при k = {opt_k}")