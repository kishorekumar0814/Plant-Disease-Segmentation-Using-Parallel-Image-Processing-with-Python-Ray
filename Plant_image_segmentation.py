import os
from glob import glob
import time
import cv2
import ray
import tensorflow as tf
from prettytable import PrettyTable
from tensorflow.keras import preprocessing
import numpy as np
import matplotlib.pyplot as plt

ray.init()

@ray.remote
def preprocess_image(image_path):
    start_time = time.time()
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))  # Resize to match MobileNetV2 input shape
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Preprocessing time for {image_path}: {execution_time:.4f} seconds")
    return resized_image, execution_time

@ray.remote
def segment_image_cnn(image):
    start_time = time.time()

    # Load the MobileNetV2 model
    model = tf.keras.applications.MobileNetV2(weights='imagenet')

    # Preprocess the image for the model
    img_array = preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Get the segmentation mask using CNN
    cnn_predictions = model.predict(img_array)
    cnn_segmented_mask = cnn_predictions[0]

    # Convert the mask to a binary image
    cnn_segmented_image = np.where(cnn_segmented_mask > 0.5, 255, 0).astype(np.uint8)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"CNN Segmentation time: {execution_time:.4f} seconds")

    return cnn_segmented_image, execution_time

def process_images_sequential(image_paths, output_folder):
    image_names = []
    sequential_times = []
    start_sequential_time = time.time()

    for i, image_path in enumerate(image_paths):
        start_time = time.time()

        img, preproc_execution_time = ray.get(preprocess_image.remote(image_path))
        segmented_img, segmentation_execution_time = ray.get(segment_image_cnn.remote(img))

        output_path = os.path.join(output_folder, f"segmented_image_{i}.jpg")
        cv2.imwrite(output_path, segmented_img)

        end_time = time.time()
        total_execution_time = end_time - start_time
        print(f"Total processing time for {image_paths[i]} (Sequential): {total_execution_time:.4f} seconds")

        image_names.append(os.path.basename(image_path))
        sequential_times.append(total_execution_time)

    end_sequential_time = time.time()
    total_sequential_time = end_sequential_time - start_sequential_time
    print(f"Total execution time for all images (Sequential): {total_sequential_time:.4f} seconds")

    return image_names, sequential_times, total_sequential_time

def process_images_parallel(image_paths, output_folder):
    parallel_times = []
    start_parallel_time = time.time()

    preprocessed_images = ray.get([preprocess_image.remote(path) for path in image_paths])

    for i, (img, preproc_execution_time) in enumerate(preprocessed_images):
        start_time = time.time()

        segmented_img, segmentation_execution_time = ray.get(segment_image_cnn.remote(img))

        output_path = os.path.join(output_folder, f"segmented_image_{i}.jpg")
        cv2.imwrite(output_path, segmented_img)

        end_time = time.time()
        total_execution_time = end_time - start_time
        print(f"Total processing time for {image_paths[i]} (Parallel - CNN): {total_execution_time:.4f} seconds")

        parallel_times.append(total_execution_time)

    end_parallel_time = time.time()
    total_parallel_time = end_parallel_time - start_parallel_time
    print(f"Total execution time for all images (Parallel): {total_parallel_time:.4f} seconds")

    return parallel_times, total_parallel_time

def plot_execution_times_line(sequential_times, parallel_times):
    x = list(range(1, len(sequential_times) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(x, sequential_times, marker='o', label='Sequential', linestyle='-', color='blue')
    plt.plot(x, parallel_times, marker='o', label='Parallel', linestyle='--', color='orange')
    plt.xlabel('Image Index Count')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Sequential vs Parallel Execution Times')
    plt.legend()
    plt.grid(True)
    plt.show()

# Specify image and output folders
image_folder = "C:/Users/kishore kumar/Desktop/sem 1/CAO/plants/"
image_paths = glob(os.path.join(image_folder, "*.jpg"))

output_folder = "C:/Users/kishore kumar/Desktop/sem 1/CAO/segmented_images/"
os.makedirs(output_folder, exist_ok=True)

# Sequential processing
image_names_seq, sequential_times_seq, _ = process_images_sequential(image_paths, output_folder)

# Parallel processing
parallel_times_par, total_parallel_time = process_images_parallel(image_paths, output_folder)

# Calculate speedup
total_sequential_time = sum(sequential_times_seq)
speedup = total_sequential_time / total_parallel_time
print(f"Speedup: {speedup:.4f}")

# Calculate efficiency
num_processors = ray.cluster_resources()["CPU"]
efficiency = speedup / int(num_processors)
print(f"Efficiency: {efficiency:.4f}")

# Create a table to display results
table = PrettyTable(["Image Name", "Sequential (seconds)", "Parallel (seconds)"])
for name, seq_time, par_time in zip(image_names_seq, sequential_times_seq, parallel_times_par):
    table.add_row([name, f"{seq_time:.4f}", f"{par_time:.4f}"])

# Add total execution times to the table
table.add_row(["Total Execution Time", f"{total_sequential_time:.4f}", f"{total_parallel_time:.4f}"])

# Print the table
print(table)

# Create a table for total execution times
total_times_table = PrettyTable(["Execution Mode", "Total Time (seconds)"])
total_times_table.add_row(["Sequential", f"{total_sequential_time:.4f}"])
total_times_table.add_row(["Parallel", f"{total_parallel_time:.4f}"])

# Print the total times table
print(total_times_table)

# Create a table for speedup and efficiency
speedup_efficiency_table = PrettyTable(["Speedup", "Efficiency"])
speedup_efficiency_table.add_row([f"{speedup:.4f}", f"{efficiency:.4f}"])

# Print the speedup and efficiency table
print(speedup_efficiency_table)

# Plotting Sequential vs Parallel Execution Times as Line Graphs
plot_execution_times_line(sequential_times_seq, parallel_times_par)

# Shutdown Ray
ray.shutdown()