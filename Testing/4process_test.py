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

ray.init(num_cpus=4)  # Specify the number of CPU processes

@ray.remote(num_cpus=2)  # Use 2 CPU process for each remote function
def preprocess_image(image_path):
    start_time = time.time()
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))
    end_time = time.time()
    execution_time = end_time - start_time
    return resized_image, execution_time

@ray.remote(num_cpus=2)
def segment_image_cnn(image):
    start_time = time.time()
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    img_array = preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    cnn_predictions = model.predict(img_array)
    cnn_segmented_mask = cnn_predictions[0]
    cnn_segmented_image = np.where(cnn_segmented_mask > 0.5, 255, 0).astype(np.uint8)
    end_time = time.time()
    execution_time = end_time - start_time
    return cnn_segmented_image, execution_time

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

def main():
    # Specify image and output folders
    image_folder = "C:/Users/kishore kumar/Desktop/sem 1/CAO/CAO_project/input_img/"
    image_paths = glob(os.path.join(image_folder, "*.jpg"))

    output_folder = "C:/Users/kishore kumar/Desktop/sem 1/CAO/CAO_project/output_img/"
    os.makedirs(output_folder, exist_ok=True)

    # Parallel processing
    parallel_times_par, total_parallel_time = process_images_parallel(image_paths, output_folder)

    # Shutdown Ray
    ray.shutdown()

    print(f"Total Parallel Execution Time: {total_parallel_time:.4f} seconds")
    print("Parallel Processing Times for Each Image:")
    for i, time_val in enumerate(parallel_times_par):
        print(f"Image {i}: {time_val:.4f} seconds")

if __name__ == "__main__":
    main()
