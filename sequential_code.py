import os
from glob import glob
import time
import cv2
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# Function for image preprocessing
def preprocess_image(image_path):
    start_time = time.time()
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (512, 512))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Preprocessing time for {image_path}: {execution_time:.4f} seconds")
    return resized_image, execution_time

# Function for image segmentation
def segment_image(image):
    start_time = time.time()
    segmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Segmentation time: {execution_time:.4f} seconds")
    return segmented_image, execution_time

image_folder = "C:/Users/kishore kumar/Desktop/sem 1/CAO/plants/"
image_paths = glob(os.path.join(image_folder, "*.jpg"))

output_folder = "C:/Users/kishore kumar/Desktop/sem 1/CAO/segmented_images/"
os.makedirs(output_folder, exist_ok=True)

# Create lists to store data
image_names = []
sequential_times = []

# Record start time for sequential processing
start_sequential_time = time.time()

# Sequential image preprocessing and segmentation
for i, image_path in enumerate(image_paths):
    start_time = time.time()

    img, preproc_execution_time = preprocess_image(image_path)
    segmented_img, segmentation_execution_time = segment_image(img)

    # Save the segmented image to the output folder
    output_path = os.path.join(output_folder, f"segmented_image_{i}.jpg")
    cv2.imwrite(output_path, segmented_img)

    end_time = time.time()
    total_execution_time = end_time - start_time
    print(f"Total processing time for {image_paths[i]} (Sequential): {total_execution_time:.4f} seconds")

    # Add data to lists
    image_names.append(os.path.basename(image_path))
    sequential_times.append(total_execution_time)

# Calculate and print total execution time for sequential processing
end_sequential_time = time.time()
total_sequential_time = end_sequential_time - start_sequential_time
print(f"Total execution time for all images (Sequential): {total_sequential_time:.4f} seconds")

# Create a line graph for sequential execution times
plt.plot(image_names, sequential_times, marker='o', linestyle='-', label='Individual Image Times')
plt.axhline(y=total_sequential_time, color='r', linestyle='--', label='Total Sequential Time')
plt.title('Sequential Execution Times')
plt.xlabel('Image Name')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
plt.legend()
plt.tight_layout()

# Save the graph as an image or display it
output_graph_path = "C:/Users/kishore kumar/Desktop/sem 1/CAO/sequential_execution_times.png"
plt.savefig(output_graph_path)
plt.show()  # Uncomment this line if you want to display the graph instead of saving it

print(f"Sequential execution times graph saved to: {output_graph_path}")

# Print the table
table = PrettyTable(["Image Name", "Sequential (seconds)"])
for name, seq_time in zip(image_names, sequential_times):
    table.add_row([name, f"{seq_time:.4f}"])

print(table)

# Create a table for total execution times
total_times_table = PrettyTable(["Execution Mode", "Total Time (seconds)"])
total_times_table.add_row(["Sequential", f"{total_sequential_time:.4f}"])

print(total_times_table)
