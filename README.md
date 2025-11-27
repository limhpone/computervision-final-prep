  # Lab 6: ML Foundations

  ## Objective
  In this lab, you will learn:
  - Generate two-dimensional synthetic data
  - Download and subsample CIFAR dataset
  - Implement KNN and Linear Classifiers from scratch
  - Use `scikit-learn` to apply both classifiers to a dataset
  - Visualize decision boundaries and evaluate model performance

  ---

  ## Lab 07 (CNN)
  We will start with the implementation of a linear classifier and MLP
  from scratch. Topics covered:
  - Initialization of weights and bias
  - Matrix multiplication of inputs (X) and weights (theta) with bias
  - Activation (Sigmoid, Softmax)
  - Loss (cost) function calculation
  - Gradient Descent (batch and stochastic)
  - Weight update
  - Training

  Then, we will advance to CNN:
  - Learn how to build and train a CNN model using PyTorch
  - Learn about MNIST dataset
  - Experiment with hyper-parameters tuning

  Model development life-cycle:
  1. Prepare the data
  2. Define the model architecture
  3. Train the model
  4. Evaluate the model

  ---

  ## Lab 08 (YOLO)
  We will go through YOLO implementation from scratch.

  Link to download PASCAL VOC dataset:
  https://www.kaggle.com/
  datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af
  8d2?resource=download
  Once you download the zip file, upload it to Puffer and unzip it. In
  the code, the path expects the unzipped folder to be named `data`.

  ---

  ## Lab 09 (Tracking)
  Learn to implement multiple tracking algorithms from simple color
  tracking to GOTURN tracking.

  For GOTURN, download the model files (goturn.caffemodel and
  goturn.prototext) and place them in the same directory with the
  provided files.

  ---

  ## Lab 10 (Segmentation)
  Learn classical segmentation techniques using mean shift and k-means
  clustering. Then implement a U-Net architecture.

  Takeaway: Implement SAM architectures (from scratch or using
  libraries).

  ---

  ## Lab 11 (Generative Network)

  ---

  ## Lab 12 (3D Vision)
  Completion requirements:
  - Learn Pytorch3D and Open3D to visualize and interpret 3D geometric
  shapes, followed by the PointNet tutorial.
  - Use Google Colab for tutorials 12.1 and 12.3. For 12.2, use the
  local PC. Tutorial 12.2 needs the `open3d` library (`pip install
  open3d`).

  ---

  # Assignment #2: From Classification to Tracking
  **Opened:** Saturday, 18 October 2025, 12:00 AM
  **Due:** Saturday, 1 November 2025, 12:05 AM
  **Due Date:** Nov 1, 2025

  ## Overview
  Implement and compare deep learning methods for image classification,
  object detection, and object tracking.

  By the end, you should be able to:
  - Build and train CNN architectures for classification
  - Evaluate and compare modern object detectors
  - Implement classical and deep learning–based tracking methods
  - Analyze trade-offs in speed, accuracy, and robustness

  ---

  ## Task 1: Image Classification with CNNs
  **Dataset Options:** CIFAR-10 or Tiny ImageNet

  Instructions:
  1. Implement and train two CNN architectures in PyTorch:
     - A simple baseline (e.g., custom CNN or LeNet)
     - A modern deep model (e.g., VGG, ResNet, or MobileNet; optional
  transfer learning)
  2. Experiment with:
     - Optimizers: SGD vs Adam
     - Learning rate schedules: `StepLR`, `ReduceLROnPlateau`
  3. Visualize:
     - Training/validation accuracy and loss curves
     - Confusion matrix
     - Example misclassified images
     - Example activation maps from CNN layer1 and layer2 of each model

  Deliverables:
  - Notebook with model training and evaluation
  - Plots and visualizations
  - Short discussion comparing performance and convergence behavior

  ---

  ## Task 2: Object Detection
  **Goal:** Compare a two-stage detector and a single-stage detector.
  **Recommended Models:** Faster R-CNN (TorchVision) and YOLO v`x`
  (version of your choice)
  **Dataset Options:** COCO 2017 (Mini subset) or Pascal VOC 2007

  Instructions:
  1. Run both detectors on the same dataset.
  2. Measure:
     - Detection accuracy (mAP or precision-recall curves)
     - Inference speed (FPS)
     - Model size and memory usage
  3. Visualize 5–10 images with predicted bounding boxes.
  4. Optionally, test reduced input resolutions or lightweight variants.

  Deliverables:
  - Notebook with detection and evaluation pipeline
  - Quantitative comparison table
  - Example detection results
  - Short discussion on performance trade-offs

  ---

  ## Task 3: Object Tracking
  **Goal:** Implement traditional tracking algorithms using OpenCV.
  **Suggested Trackers:** KCF, CSRT, MOSSE

  Instructions:
  1. Use a short video (e.g., pedestrians, cars, sports) from Pexels
  Videos or MOT Challenge.
  2. Define an initial bounding box.
  3. Apply two different OpenCV trackers (e.g., KCF and CSRT).
  4. Compare:
     - Tracking stability
     - Frame rate (FPS)
     - Failure cases (drift, loss, occlusion)

  Deliverables:
  - Output video or GIF with tracking visualization
  - Comparison table (FPS, success rate, drift cases)
  - Short discussion on performance trade-offs

  ---

  ## Submission Instructions (Assignment #2)
  Submit a folder containing:
  1. Three notebooks named `<your_studentID>_notebook_task_{#}.ipynb`
  (`#` = task number) with code, results, tables, and discussion.
  2. Output videos or GIFs.

  ---

  # Assignment #3
  **Opened:** Monday, 3 November 2025, 12:00 AM
  **Due:** Sunday, 16 November 2025, 12:05 AM
  **Due Date:** Nov 16, 2025

  ## How to Submit
  - Push all codes (Jupyter notebooks) to your GitHub repo.
  - Submit the link to your repo.

  ---

  ## Task 1: Graph Cut Segmentation
  Instructions:
  - Given 2 images (asm-1, asm2), generate bounding boxes for a person
  using any pretrained detector.
  - Using the bounding boxes, implement graph-based segmentation with
  `cv2.grabCut`.
  - Run GrabCut for 1, 3, and 5 iterations; report qualitative and
  quantitative differences.
  - Visualize: original images, user masks, final segmentation
  (foreground only and overlay).

  Deliverable:
  - Jupyter notebook with clear code and comments.

  ---

  ## Task 2: Fully Convolutional Network (FCN)
  Implement an FCN for semantic segmentation, train on a small dataset,
  and analyze architectural and training choices.

  Instructions:
  - Dataset: subset of Pascal VOC, COCO, or small custom (10–20 images).
  Split train/test; preprocess (resize, normalize, masks to class
  indices).
  - Model: FCN-32s/16s/8s variants; pretrained ResNet/VGG backbone
  (remove final FC layers); upsampling via transpose conv vs bilinear
  interpolation.
  - Training: CrossEntropyLoss; Adam or SGD; metrics: mIoU, pixel
  accuracy; train ~20 epochs; log curves.

  Deliverables:
  - Jupyter notebook with code and comments
  - Visualization of segmentation results (min. 3 test images)
  - Table comparing transpose convolution vs bilinear interpolation
  - Summary with visuals and short analysis

  ---

  ## Task 3: Variational Autoencoder (VAE)
  Implement a VAE to learn latent representations, generate new samples,
  and analyze latent dimensionality.

  Instructions:
  - Dataset: MNIST; preprocess.
  - Model: Encoder with 3–4 conv layers → flatten → Linear → outputs μ
  and log(σ²); latent dim 128. Decoder: transpose convs → reconstruct
  image.
  - Training: Loss = reconstruction (MSE or BCE) + KL divergence;
  optimizer Adam; train 50 epochs.
  - Visualize: reconstruction and latent space evolution; sample random
  z to generate images; interpolate between two z vectors.
  - Change latent dim to 256, retrain, visualize generated images and
  reconstruction quality.

  Deliverables:
  - Jupyter notebook with clear code and comments.
