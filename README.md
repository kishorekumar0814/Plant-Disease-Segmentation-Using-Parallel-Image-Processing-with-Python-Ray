# Plant Disease Segmentation Using Parallel Image Processing with Python-Ray
To Identify Plant Disease Segmentation Using Parallel  Image Processing with Python Ray

## Description:
<p> This topic focuses on accelerating the segmentation of plant diseases from images using Python-Ray, a parallel computing framework. By distributing image processing tasks across multiple processors, Python-Ray enhances the speed of disease identification in agricultural datasets, providing a scalable solution for efficient analysis. The workflow involves image preprocessing, parallel segmentation, post-processing refinement, and visualization. Challenges include load balancing, minimizing communication overhead, and ensuring scalability for large datasets. Overall, this approach optimizes disease detection in plants through the synergy of parallel processing and advanced image analysis techniques. </p>

## Working of the Project
<p>
This Python script demonstrates the parallel processing of plant images for disease segmentation using Python-Ray, particularly with a focus on comparing sequential and parallel execution times. Let's break down the code:

1. <i>Initialization and Imports:</i>
   - The script starts by importing necessary libraries, including OpenCV for image processing, TensorFlow for machine learning operations, and Ray for parallel computing.

2. <i>Function Definitions:</i>
   - `preprocess_image`: Resizes input images and measures preprocessing time.
   - `segment_image_cnn`: Uses a pre-trained MobileNetV2 model to perform image segmentation based on a convolutional neural network (CNN).
   - `process_images_sequential`: Sequentially processes images, measuring and printing execution times.
   - `process_images_parallel`: Parallelizes image processing using Python-Ray and measures and prints parallel execution times.
   - `plot_execution_times_line`: Plots a line graph comparing sequential and parallel execution times.

3. <i>Image Processing Workflow:</i>
   - Images are loaded from a specified folder.
   - The sequential processing function (`process_images_sequential`) processes each image one by one, measuring and printing execution times.
   - The parallel processing function (`process_images_parallel`) uses Python-Ray to parallelize image processing, measuring and printing parallel execution times.

4. <i>Performance Metrics:</i>
   - The script calculates and prints the speedup and efficiency achieved through parallel processing.
   - A PrettyTable is created to display image-wise execution times for both sequential and parallel processing, as well as total execution times.

5. <i>Plotting:</i>
   - The script generates a line graph using Matplotlib to visually compare the sequential and parallel execution times for each image.

6. <i>Ray Shutdown:</i>
   - The Ray framework is shut down after the completion of the parallel processing tasks.

7. <i>Output:</i>
   - The script outputs detailed information, including image-wise execution times, total execution times, speedup, and efficiency. Additionally, it generates a line graph for visual analysis.

8. <i>Note:</i>
   - Ensure that the specified image and output folders exist.
   - The script relies on the availability of a pre-trained MobileNetV2 model for image segmentation.

This code showcases the implementation of parallel image processing with Python-Ray, providing insights into the performance gains achieved through parallelization in the context of plant disease segmentation.</p>

<p>
Requirements

1) ray - pip install ray
2) opencv-python - pip install opencv-python
3) tensorflow - pip install tensorflow
4) prettytable - pip install prettytable
5) matplotlib - pip install matplotlib
6) Numpy - pip install numpy
</p>
<br>

## SEGMENTED IMAGES OF PLANT DISEASED LEAVES

![1-s2 0-S221431732100024X-gr5](https://github.com/kishorekumar0814/Plant-Disease-Segmentation-Using-Parallel-Image-Processing-with-Python-Ray/assets/105975105/cc341971-cc3c-4716-a120-11b1ff9e6a71)

## SEQUENTIAL LINE-GRAPH

![Figure_seq](https://github.com/kishorekumar0814/Plant-Disease-Segmentation-Using-Parallel-Image-Processing-with-Python-Ray/assets/105975105/9400e7f5-8000-48bb-a658-e4857b42b90f)


## SEQUENTIAL vs PARALLEL PROCESS EXECUTION TIME(in seconds) LINE-GRAPH 

![Figure_both](https://github.com/kishorekumar0814/Plant-Disease-Segmentation-Using-Parallel-Image-Processing-with-Python-Ray/assets/105975105/12d05f3e-208a-47a1-8566-c112a207146e)

<div align="center">
  <img src="https://github.com/kishorekumar0814/Plant-Disease-Segmentation-Using-Parallel-Image-Processing-with-Python-Ray/assets/105975105/12d05f3e-208a-47a1-8566-c112a207146e" alt="Figure_both" width="500"/>
</div>


<br><br>

<body>

  <h2>Advantages and Disadvantages of Parallel Processing</h2>

  <h3>Advantages</h3>
  <table>
    <tr>
      <th>Advantage</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>Increased Performance and Speed</td>
      <td>Parallel processing allows multiple tasks to be executed simultaneously, leading to a significant improvement in overall system performance.</td>
    </tr>
    <tr>
      <td>Enhanced Throughput and Scalability</td>
      <td>Parallel processing enables the efficient utilization of resources, making it easier to scale up computational power as needed.</td>
    </tr>
    <tr>
      <td>Improved Resource Utilization</td>
      <td>Parallel processing optimizes resource utilization by allowing multiple tasks to run concurrently, making better use of available hardware resources.</td>
    </tr>
    <tr>
      <td>Effective Problem Solving for Parallelizable Tasks</td>
      <td>Certain problems in fields like scientific computing, data analysis, and simulations are inherently parallelizable.</td>
    </tr>
    <tr>
      <td>Energy Efficiency and Cost Savings</td>
      <td>Parallel processing can lead to energy savings and cost efficiency by completing tasks more quickly and dynamically provisioning resources.</td>
    </tr>
  </table>

  <h3>Disadvantages</h3>
  <table>
    <tr>
      <th>Disadvantage</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>Complex Programming and Debugging</td>
      <td>Developing parallel programs is often more challenging than their sequential counterparts, leading to complex code that is harder to write and debug.</td>
    </tr>
    <tr>
      <td>Limited Parallelization Potential</td>
      <td>Not all tasks can be effectively parallelized, limiting the potential benefits of parallel processing.</td>
    </tr>
    <tr>
      <td>Scalability Challenges</td>
      <td>Scalability can be limited by factors such as communication overhead, contention for shared resources, and the nature of the problem being solved.</td>
    </tr>
    <tr>
      <td>Increased Complexity of System Design</td>
      <td>Implementing parallel processing systems requires careful consideration of hardware architecture, communication protocols, and load balancing.</td>
    </tr>
    <tr>
      <td>Potential for Unpredictable Performance</td>
      <td>The performance of parallel processing systems can be unpredictable due to factors such as varying workloads, contention for shared resources, and communication delays.</td>
    </tr>
  </table>

</body>

</html>


