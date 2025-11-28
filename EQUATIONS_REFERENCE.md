# Computer Vision Equations Reference Guide

A comprehensive collection and explanation of mathematical equations from computer vision concepts, deep learning, and machine learning fundamentals.

---

## Table of Contents
1. [Object Detection (R-CNN Family)](#1-object-detection-r-cnn-family)
2. [Generative Models](#2-generative-models)
3. [Image Segmentation](#3-image-segmentation)
4. [Classification & Distance Metrics](#4-classification--distance-metrics)
5. [Image Processing & Transformations](#5-image-processing--transformations)
6. [3D Computer Vision](#6-3d-computer-vision)
7. [Convolutional Neural Networks](#7-convolutional-neural-networks)
8. [Mathematical Foundations](#8-mathematical-foundations)

---

## 1. Object Detection (R-CNN Family)

### 1.1 Fast R-CNN Multi-Task Loss
**Equation:**
$$L(p, u, t^u, v) = L_{cls}(p, u) + \lambda [u \geq 1] L_{loc}(t^u, v)$$

**Explanation:**
This is the combined loss function for Fast R-CNN that jointly trains classification and bounding box regression:
- $L_{cls}(p, u)$: Classification loss comparing predicted class probabilities $p$ with ground truth class $u$
- $L_{loc}(t^u, v)$: Localization loss for bounding box coordinates
- $t^u$: Predicted bounding box for class $u$
- $v$: Ground truth bounding box
- $\lambda$: Weight balancing term (typically set to 1)
- $[u \geq 1]$: Indicator function (equals 1 for object classes, 0 for background)

**Purpose:** Enables end-to-end training by combining two objectives - correctly classifying objects and accurately localizing them in the image.

---

### 1.2 Faster R-CNN Loss Function
**Equation:**
$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)$$

**Explanation:**
The Region Proposal Network (RPN) loss function in Faster R-CNN:
- $i$: Index of an anchor box in a mini-batch
- $p_i$: Predicted probability that anchor $i$ contains an object
- $p_i^*$: Ground truth label (1 if positive anchor, 0 if negative)
- $t_i$: Predicted bounding box coordinates
- $t_i^*$: Ground truth box coordinates
- $N_{cls}$: Normalization term (mini-batch size, typically 256)
- $N_{reg}$: Normalization term (number of anchor locations, typically ~2400)
- $\lambda$: Balancing parameter (typically 10)

**Purpose:** Trains the RPN to generate high-quality region proposals by predicting both objectness scores and refined bounding box coordinates.

---

## 2. Generative Models

### 2.1 GAN Minimax Objective
**Equation:**
$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$$

**Explanation:**
The fundamental objective function for Generative Adversarial Networks:
- $G$: Generator network that creates fake samples
- $D$: Discriminator network that classifies real vs fake
- $x \sim p_{data}$: Real samples from the data distribution
- $z \sim p_z$: Random noise vectors from latent space (typically Gaussian)
- $\mathbb{E}[\cdot]$: Expected value
- $D(x)$: Discriminator's probability that $x$ is real
- $G(z)$: Generator's output (fake sample) from noise $z$

**Purpose:** Creates a two-player game where:
- **Discriminator $D$** tries to maximize the objective (distinguish real from fake)
- **Generator $G$** tries to minimize the objective (fool the discriminator)

**Interpretation:**
- First term: Discriminator correctly identifies real samples (wants $D(x) \rightarrow 1$)
- Second term: Discriminator correctly identifies fake samples (wants $D(G(z)) \rightarrow 0$)
- Generator wants $D(G(z)) \rightarrow 1$ (fool discriminator into thinking fakes are real)

---

### 2.2 DCGAN Convolutional Layer
**Equation:**
$$h^l = \sigma(W^l * h^{l-1} + b^l)$$

**Explanation:**
Convolution operation in Deep Convolutional GAN:
- $h^l$: Output feature map at layer $l$
- $h^{l-1}$: Input feature map from previous layer
- $W^l$: Convolutional filter weights at layer $l$
- $b^l$: Bias term
- $*$: Convolution operator (not multiplication)
- $\sigma(\cdot)$: Activation function (e.g., ReLU, LeakyReLU)

**Purpose:** Preserves 2D spatial structure of images through convolution, unlike fully connected layers. Each spatial location shares the same weights, creating translation invariance.

**Advantage:** Convolutional architecture learns hierarchical features - from edges at lower layers to complete objects at higher layers.

---

### 2.3 Batch Normalization
**Equation:**
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**Explanation:**
Normalizes layer inputs to have zero mean and unit variance:
- $x$: Input value
- $\mu_B$: Mean of values in the batch
- $\sigma_B^2$: Variance of values in the batch
- $\epsilon$: Small constant for numerical stability (typically $10^{-5}$)
- $\hat{x}$: Normalized output

**Purpose:** 
- Stabilizes training by reducing internal covariate shift
- Allows higher learning rates
- Reduces sensitivity to weight initialization
- Prevents gradient vanishing/explosion
- Acts as a regularizer (slight noise from batch statistics)

**When Used:** Between convolutional/linear layers and activation functions in deep networks, especially GANs.

---

### 2.4 VAE Reparameterization Trick
**Equation:**
$$z = \mu + \sigma \cdot \epsilon, \quad \text{where } \sigma = e^{0.5 \cdot \log\sigma^2}$$

**Explanation:**
Enables backpropagation through stochastic sampling in Variational Autoencoders:
- $z$: Latent code sampled from learned distribution
- $\mu$: Mean of the latent distribution (learned)
- $\sigma$: Standard deviation (derived from learned log-variance)
- $\epsilon \sim \mathcal{N}(0, 1)$: Random noise from standard normal distribution
- $\log\sigma^2$: Log-variance (learned parameter for numerical stability)

**Purpose:** 
- Makes sampling differentiable by separating deterministic ($\mu, \sigma$) and stochastic ($\epsilon$) components
- Allows gradients to flow through the sampling operation during backpropagation
- Without this trick, we cannot train VAE end-to-end

**Why It Works:** The gradient with respect to $\mu$ and $\sigma$ can be computed, but not with respect to $\epsilon$ (which we don't need since it's fixed noise).

---

### 2.5 VAE Loss Function (ELBO)
**Equation:**
$$L = L_{reconstruction} + L_{KL} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

**Expanded KL Divergence:**
$$D_{KL} = -0.5 \sum_{j=1}^{d} (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

**Explanation:**
The Evidence Lower Bound (ELBO) that VAE maximizes:
- $L_{reconstruction}$: How well decoder reconstructs input (typically MSE or binary cross-entropy)
- $L_{KL}$: KL divergence regularizing the latent distribution
- $q(z|x)$: Encoder's learned distribution (approximate posterior)
- $p(x|z)$: Decoder's reconstruction likelihood
- $p(z)$: Prior distribution (typically $\mathcal{N}(0, I)$)
- $d$: Dimensionality of latent space
- $\mu_j, \sigma_j$: Mean and std dev of latent dimension $j$

**Purpose:**
- **Reconstruction term**: Ensures generated images look like training data
- **KL term**: Ensures latent space has good properties (continuous, structured)
- Balances reconstruction quality with latent space regularization

**Intuition:** 
- Minimize KL divergence = make learned distribution close to simple prior (Gaussian)
- Maximize reconstruction = generate accurate outputs
- Trade-off creates smooth, interpolatable latent space

---

## 3. Image Segmentation

### 3.1 Mean Shift Algorithm
**Equation:**
$$m(x) = \frac{\sum_{x_i \in N(x)} K(x_i - x) x_i}{\sum_{x_i \in N(x)} K(x_i - x)}$$

**Explanation:**
Iteratively updates each point's position toward the mode of its local distribution:
- $m(x)$: New position (mean) to move point $x$ toward
- $x_i$: Neighboring data points
- $N(x)$: Neighborhood around $x$ (points within bandwidth)
- $K(\cdot)$: Kernel function measuring influence (typically Gaussian)
- Numerator: Weighted sum of neighboring positions
- Denominator: Normalization factor (sum of weights)

**Kernel Function (Gaussian):**
$$K(x_i - x) = e^{-\frac{\|x_i - x\|^2}{2h^2}}$$
where $h$ is the bandwidth parameter.

**Purpose:** 
- Non-parametric clustering algorithm
- Each point climbs the gradient of the density function
- Points converge to local maxima (modes) = cluster centers
- No need to specify number of clusters beforehand

**Applications:**
- Image segmentation (RGB space or XYRGB space)
- Object tracking
- Mode detection in any feature space

---

### 3.2 RGB Feature Vector
**Equation:**
$$f_i = [R_i, G_i, B_i]^T \in \mathbb{R}^3$$

**Explanation:**
Feature vector for color-based segmentation:
- $R_i, G_i, B_i$: Red, Green, Blue color values of pixel $i$
- Range: [0, 255] for 8-bit images
- $\mathbb{R}^3$: 3-dimensional real space

**Purpose:** Represents each pixel as a point in 3D color space. Pixels with similar colors cluster together.

**Limitation:** Ignores spatial information - pixels far apart but with same color are treated as neighbors.

---

### 3.3 XYRGB Feature Vector  
**Equation:**
$$f_i = [X_i, Y_i, R_i, G_i, B_i]^T \in \mathbb{R}^5$$

**Explanation:**
Extended feature vector including both spatial and color information:
- $X_i, Y_i$: Spatial coordinates of pixel $i$ (position in image)
- $R_i, G_i, B_i$: Color values
- $\mathbb{R}^5$: 5-dimensional real space

**Purpose:**
- Combines color similarity AND spatial proximity
- Pixels must be both similar in color AND close in position to be grouped
- Creates spatially coherent segments that respect object boundaries

**Advantage:** Prevents grouping spatially distant regions with similar colors, producing more meaningful segmentation.

---

## 4. Classification & Distance Metrics

### 4.1 Euclidean Distance
**Equation:**
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

**Explanation:**
Straight-line distance between two points in n-dimensional space:
- $x, y$: Two feature vectors
- $x_i, y_i$: Values at dimension $i$
- $n$: Number of dimensions

**Purpose:** Most common distance metric in KNN and clustering algorithms. Measures how "far apart" two data points are.

**Alternative form:**
$$d(x, y) = \|x - y\|_2 = \sqrt{(x - y)^T(x - y)}$$

---

### 4.2 Manhattan Distance
**Equation:**
$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

**Explanation:**
Sum of absolute differences along each dimension (L1 norm):
- Also called "city block" or "taxicab" distance
- Measures distance if you can only move along axes (like streets in a grid city)

**Purpose:** More robust to outliers than Euclidean distance. Used when features have different scales or when diagonal movement isn't meaningful.

---

### 4.3 Minkowski Distance
**Equation:**
$$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

**Explanation:**
Generalization of Euclidean and Manhattan distances:
- $p = 1$: Manhattan distance (L1 norm)
- $p = 2$: Euclidean distance (L2 norm)
- $p = \infty$: Chebyshev distance (maximum difference in any dimension)

**Purpose:** Flexible distance metric allowing different notions of similarity based on the value of $p$.

---

## 5. Image Processing & Transformations

### 5.1 Linear System - Shift Invariance (LSIS)
**Equation:**
$$f(x - x_0) \rightarrow g(x - x_0)$$

**Explanation:**
If input is shifted by $x_0$, output is shifted by the same amount:
- $f(x)$: Input function/signal
- $g(x)$: Output function/signal
- $x_0$: Shift amount

**Purpose:** Fundamental property of linear, shift-invariant systems. The system's response doesn't depend on position - same filter produces same effect everywhere.

**Example:** Applying a blur filter to an image - the blur effect is the same whether applied to top-left or bottom-right.

---

### 5.2 Linearity (Superposition)
**Equation:**
$$S[a \cdot f_1(x) + b \cdot f_2(x)] = a \cdot S[f_1(x)] + b \cdot S[f_2(x)]$$

**Explanation:**
System operator $S$ respects linear combinations:
- $f_1, f_2$: Two input signals
- $a, b$: Scaling constants
- System response to sum = sum of individual responses

**Purpose:** Allows decomposing complex inputs into simpler components, processing separately, then combining results.

---

### 5.3 Convolution
**Equation:**
$$g(x) = (f * h)(x) = \sum_{m}^{M}\sum_{n}^{N} f[m,n] \cdot h[x+m, y+n]$$

**Explanation:**
Mathematical operation combining two functions:
- $f$: Input image
- $h$: Convolution kernel/filter
- $g$: Output image
- $*$: Convolution operator
- Sum over kernel size $M \times N$

**Purpose:**
- Implements LSIS systems completely
- Used for filtering: blurring, sharpening, edge detection
- Core operation in CNNs

**Physical Interpretation:** Each output pixel is a weighted combination of input pixels in the kernel's neighborhood.

---

### 5.4 Gamma Correction (Power Law Transform)
**Equation:**
$$I_{out}(x,y) = \left(\frac{I_{in}(x,y)}{255}\right)^{\gamma} \times 255$$

**Explanation:**
Non-linear intensity transformation:
- $I_{in}$: Input pixel intensity
- $I_{out}$: Output pixel intensity
- $\gamma$: Gamma value (power exponent)
- Normalized to [0,1] range before applying power, then scaled back

**Cases:**
- $\gamma < 1$: Brightens image (expands dark regions)
- $\gamma = 1$: No change (identity transform)
- $\gamma > 1$: Darkens image (compresses dark regions)

**Purpose:** 
- Compensates for display device non-linearity
- Adjusts image brightness/contrast
- Enhances visibility of details in shadows or highlights

**Example (Brightening with $\gamma = 0.5$):**
$$I_{out} = \left(\frac{I_{in}}{255}\right)^{1/2} \times 255 = \sqrt{\frac{I_{in}}{255}} \times 255$$

**Example (Darkening with $\gamma = 2$):**
$$I_{out} = \left(\frac{I_{in}}{255}\right)^{2} \times 255$$

---

## 6. 3D Computer Vision

### 6.1 3D to 2D Projection (Camera Model)
**Equation:**
$$x_i = f\frac{x_c}{z_c}, \quad y_i = f\frac{y_c}{z_c}$$

**Explanation:**
Perspective projection from 3D world coordinates to 2D image coordinates:
- $(x_c, y_c, z_c)$: 3D point in camera coordinates
- $(x_i, y_i)$: 2D point in image plane
- $f$: Focal length of camera (distance from lens to image plane)
- Division by $z_c$ creates perspective effect (farther objects appear smaller)

**Purpose:** Models how cameras capture 3D scenes as 2D images.

---

### 6.2 Intrinsic Camera Parameters
**Equation:**
$$x = \frac{u \phi_x}{w} + \delta_x, \quad y = \frac{v \phi_y}{w} + \delta_y$$

**Explanation:**
Transformation from homogeneous image coordinates to pixel coordinates:
- $(u, v, w)$: Homogeneous coordinates
- $(x, y)$: Pixel coordinates
- $\phi_x, \phi_y$: Scale factors (pixels per unit distance)
- $\delta_x, \delta_y$: Principal point offsets (optical center)

**Purpose:** Accounts for sensor geometry - pixel size, aspect ratio, and image center offset.

---

### 6.3 Stereo Vision - Epipolar Constraint
**Equation:**
$$X_l \cdot X_r = 0$$

**Explanation:**
Fundamental constraint in stereo vision:
- $X_l, X_r$: Corresponding points in left and right images
- Dot product = 0 means vectors are perpendicular
- Points lie on corresponding epipolar lines

**Purpose:** Reduces stereo matching from 2D search to 1D search along epipolar line.

---

### 6.4 Rigid Body Transformation
**Equation:**
$$X_l = RX_r + t$$

**Explanation:**
Transforms point from right camera frame to left camera frame:
- $R$: 3×3 rotation matrix
- $t$: 3×1 translation vector
- $X_r$: Point in right camera coordinates
- $X_l$: Same point in left camera coordinates

**Purpose:** Relates coordinates between two camera viewpoints in stereo systems.

---

## 7. Convolutional Neural Networks

### 7.1 Convolutional Layer Output Size
**Equation:**
$$O = \frac{W - K + 2P}{S} + 1$$

**Explanation:**
Calculates spatial dimensions of conv layer output:
- $O$: Output size (height or width)
- $W$: Input size
- $K$: Kernel/filter size
- $P$: Padding
- $S$: Stride

**Purpose:** Design CNN architectures by calculating how spatial dimensions change through layers.

**Example:** 
- Input: 28×28, Kernel: 3×3, Padding: 1, Stride: 1
- Output: $(28 - 3 + 2(1))/1 + 1 = 28$ (same size)

---

### 7.2 Receptive Field
**Equation:**
$$RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$$

**Explanation:**
Receptive field is the region of input that affects one output neuron:
- $RF_l$: Receptive field at layer $l$
- $K_l$: Kernel size at layer $l$
- $S_i$: Stride at layer $i$
- Product accumulates stride effects from previous layers

**Purpose:** Understanding how much context each neuron "sees" - deeper layers have larger receptive fields.

---

## 8. Mathematical Foundations

### 8.1 Matrix-Vector Multiplication
**Equation:**
$$\begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix} = \begin{bmatrix} \omega_{11} & \omega_{12} & \omega_{13}\\ \omega_{21} & \omega_{22} & \omega_{23}\\ \omega_{31} & \omega_{32} & \omega_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} + \begin{bmatrix} \tau_1 \\ \tau_2 \\ \tau_3 \end{bmatrix}$$

**Explanation:**
Affine transformation in homogeneous coordinates:
- Rotation/scaling by 3×3 matrix $\omega$
- Translation by vector $\tau$
- Homogeneous coordinates allow representing translation as matrix multiplication

**Purpose:** General 2D/3D transformations including rotation, scaling, shearing, and translation.

---

### 8.2 Definite Integral
**Equation:**
$$\int_{a}^{b} f(x) \,dx = F(b) - F(a)$$

**Explanation:**
Fundamental theorem of calculus:
- $f(x)$: Function to integrate
- $F(x)$: Antiderivative of $f$ (where $F'(x) = f(x)$)
- $[a, b]$: Integration interval
- Result: Area under curve from $a$ to $b$

**Purpose:** Calculates accumulated change, area, volume, or probability depending on context.

---

### 8.3 Multiple Integrals
**Equation:**
$$\iint_V \mu(u,v) \,du\,dv$$

**Explanation:**
Double integral over region $V$:
- $\mu(u,v)$: Function of two variables
- $du, dv$: Infinitesimal changes in $u$ and $v$
- Integration over 2D region

**Purpose:** 
- Computing area
- Calculating probability over 2D distribution
- Finding mass/charge over surface

---

### 8.4 Differential Equations
**Equation:**
$$\frac{d^2y}{dx^2} + k^2y = 0$$

**Explanation:**
Second-order linear differential equation:
- $\frac{dy}{dx}$: First derivative (rate of change)
- $\frac{d^2y}{dx^2}$: Second derivative (curvature/acceleration)
- $k$: Constant parameter

**Solution:** $y = A\cos(kx) + B\sin(kx)$ (harmonic oscillator)

**Purpose:** Models oscillating systems, wave propagation, signal processing.

---

### 8.5 Expected Value
**Equation:**
$$\mathbb{E}_{x \sim p(x)}[f(x)] = \int f(x)p(x)\,dx$$

**Explanation:**
Average value of function $f$ under probability distribution $p$:
- $x \sim p(x)$: Random variable $x$ drawn from distribution $p$
- $f(x)$: Function of interest
- Weighted average where weights are probabilities

**Discrete version:**
$$\mathbb{E}[f(x)] = \sum_x f(x)p(x)$$

**Purpose:** 
- Fundamental concept in probability and statistics
- Loss functions in machine learning are often expected values
- Describes "average case" behavior

---

## Summary Table: Loss Functions

| Model | Loss Function | Purpose |
|-------|---------------|---------|
| **Fast R-CNN** | $L = L_{cls} + \lambda L_{loc}$ | Multi-task: classification + localization |
| **Faster R-CNN (RPN)** | $L = \frac{1}{N_{cls}}\sum L_{cls} + \frac{\lambda}{N_{reg}}\sum p^* L_{reg}$ | Region proposal generation |
| **GAN** | $\min_G \max_D [\mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]]$ | Adversarial training |
| **VAE** | $L = -\mathbb{E}[\log p(x\|z)] + D_{KL}(q(z\|x) \|\| p(z))$ | Reconstruction + regularization |
| **BCE (GAN)** | $L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$ | Binary classification |
| **MSE** | $L = \frac{1}{N}\sum(y - \hat{y})^2$ | Regression, reconstruction |

---

## Key Activation Functions

| Function | Equation | Range | Usage |
|----------|----------|-------|-------|
| **ReLU** | $f(x) = \max(0, x)$ | $[0, \infty)$ | Hidden layers (general) |
| **LeakyReLU** | $f(x) = \max(\alpha x, x)$ | $(-\infty, \infty)$ | Prevents dying neurons |
| **Sigmoid** | $f(x) = \frac{1}{1+e^{-x}}$ | $(0, 1)$ | Binary classification, GANs |
| **Tanh** | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | GAN generator output |
| **Softmax** | $f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)$, sum=1 | Multi-class classification |

---

## Common Notation Guide

| Symbol | Meaning |
|--------|---------|
| $x, y$ | Input and output/target |
| $\hat{y}$ | Predicted output |
| $W, b$ | Weights and biases |
| $\theta$ | Model parameters (general) |
| $z$ | Latent variable / noise vector |
| $\alpha, \lambda$ | Hyperparameters |
| $\epsilon$ | Small constant (numerical stability) or noise |
| $\mu, \sigma$ | Mean and standard deviation |
| $*$ | Convolution operator |
| $\odot$ | Element-wise multiplication |
| $\|\cdot\|$ | Norm (magnitude) |
| $\nabla$ | Gradient operator |
| $\mathbb{E}[\cdot]$ | Expected value |
| $\mathbb{R}^n$ | n-dimensional real space |
| $\sim$ | "Distributed as" / "drawn from" |

---

## References and Resources

### Papers
- Fast R-CNN (Girshick, 2015)
- Faster R-CNN (Ren et al., 2016)
- GAN (Goodfellow et al., 2014)
- DCGAN (Radford et al., 2016)
- VAE (Kingma & Welling, 2014)

### Books & Courses
- Deep Learning (Goodfellow, Bengio, Courville)
- Computer Vision: Algorithms and Applications (Szeliski)
- CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)

### Key Concepts by Topic

**Object Detection:**
- Region proposals → Feature extraction → Classification + Regression
- Evolution: R-CNN → Fast R-CNN → Faster R-CNN
- Trade-offs: Accuracy vs Speed vs Memory

**Generative Models:**
- GANs: Adversarial game between generator and discriminator
- VAEs: Learn continuous latent space with probabilistic encoding
- Applications: Image synthesis, style transfer, data augmentation

**Segmentation:**
- Clustering: K-means, Mean Shift
- Deep learning: U-Net, Mask R-CNN, FCN
- Feature spaces: RGB, Lab, XYRGB

---

## Tips for Understanding Equations

1. **Break down components**: Identify each variable and what it represents
2. **Check dimensions**: Ensure matrix/vector dimensions are compatible
3. **Understand the intuition**: What problem does this equation solve?
4. **Trace through examples**: Use small numerical examples
5. **Visualize**: Draw diagrams showing data flow
6. **Compare variants**: How does this differ from related equations?

---

## Appendix: Derivation Examples

### A.1 KL Divergence Simplification for VAE

Starting with:
$$D_{KL}(q(z|x) \| p(z)) = \int q(z|x) \log\frac{q(z|x)}{p(z)}dz$$

Where:
- $q(z|x) = \mathcal{N}(\mu, \sigma^2I)$ (encoder output)
- $p(z) = \mathcal{N}(0, I)$ (prior)

After integration and simplification:
$$D_{KL} = -\frac{1}{2}\sum_{j=1}^d (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

This closed-form allows efficient computation without sampling.

---

### A.2 Convolution as Matrix Multiplication

A 2D convolution can be reformulated as matrix multiplication:
$$y = Wx$$

Where:
- $x$: Vectorized input image (flattened)
- $W$: Sparse Toeplitz matrix encoding the kernel
- $y$: Vectorized output

This formulation shows convolution is a linear operation, enabling efficient GPU implementation.

---

## Conclusion

This reference guide covers fundamental equations across computer vision and deep learning. Each equation represents a key concept:
- **Loss functions** guide optimization
- **Activation functions** introduce non-linearity
- **Distance metrics** measure similarity
- **Transformations** manipulate data

Understanding these equations deeply enables:
- Implementing models from scratch
- Debugging issues
- Designing novel architectures
- Interpreting research papers

**Remember:** Equations are tools. The goal is to solve real-world problems, not just manipulate symbols. Always connect mathematical formalism to practical intuition.

---

**Last Updated:** November 28, 2025  
**Author:** Generated from Computer Vision course materials  
**Version:** 1.0
