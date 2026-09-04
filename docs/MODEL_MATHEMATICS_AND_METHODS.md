# 📐 BharatANPR: Deep Mathematical Formulations, Loss Functions, and Algorithmic Reference

> **Document Classification**: *Rigorous Technical & Theoretical Specification*  
> **Target Audience**: *Deep Learning Researchers, Computer Vision Scientists, and Quantitative AI Engineers*

---

## 📑 Table of Contents
1. [Introduction & Notation Conventions](#1-introduction--notation-conventions)
2. [YOLO11 Object Detection Architecture & Loss Formulations](#2-yolo11-object-detection-architecture--loss-formulations)
   - [2.1 TaskAligned Assignor (TAL) Mechanics](#21-taskaligned-assignor-tal-mechanics)
   - [2.2 Complete Intersection-over-Union (CIoU) Loss](#22-complete-intersection-over-union-ciou-loss)
   - [2.3 Distribution Focal Loss (DFL)](#23-distribution-focal-loss-dfl)
   - [2.4 Total Composite Multi-Task Loss](#24-total-composite-multi-task-loss)
3. [Multi-Object Tracking: Discrete Kalman Filter & ByteTrack Mathematics](#3-multi-object-tracking-discrete-kalman-filter--bytetrack-mathematics)
   - [3.1 Discrete State-Space Formulation](#31-discrete-state-space-formulation)
   - [3.2 Time Update (Prediction Steps)](#32-time-update-prediction-steps)
   - [3.3 Measurement Update (Correction Steps)](#33-measurement-update-correction-steps)
   - [3.4 Bipartite Matching via the Hungarian Algorithm](#34-bipartite-matching-via-the-hungarian-algorithm)
   - [3.5 ByteTrack Two-Stage Association Algorithm](#35-bytetrack-two-stage-association-algorithm)
4. [Neural Sequence Recognition: CRNN + BiLSTM + CTC Loss](#4-neural-sequence-recognition-crnn--bilstm--ctc-loss)
   - [4.1 CNN Feature Extraction & Map Slicing](#41-cnn-feature-extraction--map-slicing)
   - [4.2 Bidirectional LSTM Recurrent Modeling](#42-bidirectional-lstm-recurrent-modeling)
   - [4.3 Connectionist Temporal Classification (CTC) Theory](#43-connectionist-temporal-classification-ctc-theory)
   - [4.4 The Forward-Backward Algorithm for CTC](#44-the-forward-backward-algorithm-for-ctc)
   - [4.5 Exact Analytical Gradient Derivation](#45-exact-analytical-gradient-derivation)
   - [4.6 Prefix Beam Search Decoding with Language Priors](#46-prefix-beam-search-decoding-with-language-priors)
5. [Permutation Autoregressive Sequence (PARSeq) Vision Transformer](#5-permutation-autoregressive-sequence-parseq-vision-transformer)
   - [5.1 Multi-Head Self-Attention Mechanics](#51-multi-head-self-attention-mechanics)
   - [5.2 Permutation Language Modeling Formulation](#52-permutation-language-modeling-formulation)
6. [Image Preprocessing & Morphological Mathematics](#6-image-preprocessing--morphological-mathematics)
   - [6.1 Contrast Limited Adaptive Histogram Equalization (CLAHE)](#61-contrast-limited-adaptive-histogram-equalization-clahe)
   - [6.2 Otsu's Global Binarization via Variance Maximization](#62-otsus-global-binarization-via-variance-maximization)
   - [6.3 Bilateral Filtering for Edge-Preserving Denoising](#63-bilateral-filtering-for-edge-preserving-denoising)

---

## 1. Introduction & Notation Conventions

This document provides the exhaustive mathematical, probabilistic, and algorithmic foundation underlying the **BharatANPR** intelligent transportation system.

### Mathematical Notations:
- Bold uppercase letters ($\mathbf{X}, \mathbf{F}, \mathbf{H}$) denote matrices.
- Bold lowercase letters ($\mathbf{x}, \mathbf{z}, \mathbf{y}$) denote vectors.
- Scalar variables are italicized ($t, s, \alpha, \lambda$).
- Calligraphic letters ($\mathcal{L}, \mathcal{B}, \mathcal{S}$) denote operators, losses, or sets.
- $\mathbb{E}[\cdot]$ denotes the mathematical expectation operator.
- $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ denotes a multivariate Gaussian distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$.

---

## 2. YOLO11 Object Detection Architecture & Loss Formulations

YOLO11 is an anchor-free single-stage object detector featuring a modified CSPNet backbone with **C3k2** (Cross Stage Partial with 2 convolutions) residual blocks, an optimized **SPPF** (Spatial Pyramid Pooling Fast) neck, and decoupled anchor-free detection heads for classification and bounding box regression.

---

### 2.1 TaskAligned Assignor (TAL) Mechanics

Traditional anchor-based detectors (e.g., YOLOv3/v4) assign ground-truth labels to anchor boxes purely based on spatial IoU overlap:
$$\text{Anchor is Positive} \iff \text{IoU}(\text{Anchor}, \text{GT}) \ge \tau$$
This introduces a severe structural flaw: an anchor might overlap spatially with the ground truth, but its convolutional feature vector might reside near a class decision boundary, resulting in a low classification score. Conversely, a feature point with high classification confidence might have poor localization regression.

YOLO11 eliminates heuristic IoU assignment by employing the **TaskAligned Assignor (TAL)**. TAL computes a unified alignment metric $t$ for every prediction-target pair:

$$t = s^\alpha \times \text{IoU}^\beta$$

where:
- $s \in [0, 1]$ is the predicted classification probability for the target class.
- $\text{IoU} \in [0, 1]$ is the Intersection-over-Union between the predicted bounding box and the ground-truth box.
- $\alpha$ is the classification weighting factor (default: $\alpha = 0.5$).
- $\beta$ is the localization weighting factor (default: $\beta = 6.0$).

The alignment metric $t$ dynamically balances classification and localization. For each ground truth bounding box $g_j$, the top $K$ anchor points with the highest alignment scores $t$ within the ground truth spatial mask are selected as positive samples.

To supervise the classification head, the discrete binary label $y \in \{0, 1\}$ is replaced with the **normalized alignment target** $\hat{t}$:

$$\hat{t}_i = \frac{t_i}{\max_{j \in \mathcal{P}} t_j} \times \max_{j \in \mathcal{P}} \text{IoU}_j$$

where $\mathcal{P}$ is the set of positive candidates assigned to that ground truth.

#### Classification Loss (Binary Cross-Entropy with Soft Targets):
$$\mathcal{L}_{cls} = -\sum_{i=1}^N \left[ \hat{t}_i \log(\hat{p}_i) + (1 - \hat{t}_i) \log(1 - \hat{p}_i) \right]$$

where $\hat{p}_i = \sigma(c_i)$ is the sigmoid output of the classification logit $c_i$.

---

### 2.2 Complete Intersection-over-Union (CIoU) Loss

Bounding box regression minimizes the spatial discrepancy between the predicted box $b = (x, y, w, h)$ and the ground truth box $b^{gt} = (x^{gt}, y^{gt}, w^{gt}, h^{gt})$.

Standard Smooth-$L_1$ or MSE losses optimize each coordinate independently, ignoring the geometric correlations between $(x, y)$ and $(w, h)$. BharatANPR utilizes **CIoU (Complete IoU)** loss, which simultaneously optimizes three geometric factors:
1. **Overlap Area** ($\text{IoU}$)
2. **Normalized Central Point Distance** ($\rho^2 / c^2$)
3. **Aspect Ratio Consistency** ($v$)

#### Mathematical Formulation:
$$\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v$$

where:
- $\mathbf{b} = (x_c, y_c)$ and $\mathbf{b}^{gt} = (x_c^{gt}, y_c^{gt})$ represent the center coordinates of the predicted and ground truth boxes.
- $\rho^2(\mathbf{b}, \mathbf{b}^{gt}) = (x_c - x_c^{gt})^2 + (y_c - y_c^{gt})^2$ is the squared Euclidean distance between the center points.
- $c$ is the diagonal length of the smallest enclosing box covering both $\mathbf{b}$ and $\mathbf{b}^{gt}$.
- $v$ quantifies aspect ratio similarity:
  $$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2$$
- $\alpha$ is a dynamic trade-off parameter giving higher priority to overlap when IoU is low:
  $$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

When $w/h = w^{gt}/h^{gt}$, $v = 0$ and CIoU gracefully reduces to Distance-IoU (DIoU).

---

### 2.3 Distribution Focal Loss (DFL)

Traditional bounding box regression models the four offsets $(\delta x, \delta y, \delta w, \delta h)$ as single deterministic scalar Dirac delta distributions:
$$P(y) = \delta(y - \hat{y})$$
However, under real-world traffic conditions—where license plate edges are degraded by motion blur, shadows, or dust—the exact pixel boundary is inherently uncertain and ambiguous.

**Distribution Focal Loss (DFL)** treats each continuous coordinate distance $y$ as a continuous random variable whose value is modeled by a general probability distribution over a discrete set of bins:
$$\mathcal{Y} = \{y_0, y_1, y_2, \dots, y_n\}, \quad \text{where } y_0 \le y \le y_n$$
In YOLO11, $n = 16$ with uniform step size $\Delta = 1.0$.

The predicted continuous coordinate $\hat{y}$ is computed as the statistical expectation over the discrete probability distribution $\mathbf{S} = [S_0, S_1, \dots, S_n]^T$:

$$\hat{y} = \sum_{i=0}^n S_i y_i, \quad \text{where } S_i = \frac{\exp(\mathcal{S}_i)}{\sum_{j=0}^n \exp(\mathcal{S}_j)}$$

where $\mathcal{S}_i$ are the raw network output logits.

To supervise the distribution $\mathbf{S}$ such that the probability mass concentrates immediately around the true continuous target $y \in [y_i, y_{i+1}]$, DFL penalizes the two adjacent discrete bins $y_i$ and $y_{i+1}$ using focal cross-entropy:

$$\mathcal{L}_{DFL}(S_i, S_{i+1}) = - \left[ (y_{i+1} - y) \log(S_i) + (y - y_i) \log(S_{i+1}) \right]$$

where $y_i = \lfloor y \rfloor$ and $y_{i+1} = \lfloor y \rfloor + 1$.

#### Theoretical Guarantee:
The global minimum of $\mathcal{L}_{DFL}$ is uniquely attained when:
$$S_i = y_{i+1} - y, \quad S_{i+1} = y - y_i$$
which guarantees that the expectation $\mathbb{E}[\hat{y}]$ equals the true continuous ground truth $y$:
$$\mathbb{E}[\hat{y}] = S_i y_i + S_{i+1} y_{i+1} = (y_{i+1} - y) y_i + (y - y_i) y_{i+1} = y$$

---

### 2.4 Total Composite Multi-Task Loss

The complete objective function minimized during training of `indian_vehicles_yolo11.pt` and `indian_plate_yolo11.pt` is:

$$\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{box} \mathcal{L}_{CIoU} + \lambda_{dfl} \mathcal{L}_{DFL}$$

In BharatANPR, the hyperparameters are set to standard production values:
$$\lambda_{cls} = 0.5, \quad \lambda_{box} = 7.5, \quad \lambda_{dfl} = 1.5$$

---

## 3. Multi-Object Tracking: Discrete Kalman Filter & ByteTrack Mathematics

Tracking multiple vehicles across video frames requires estimating temporal trajectories under noisy, incomplete, or occluded detections. BharatANPR employs **ByteTrack** with a discrete **Linear Kalman Filter**.

---

### 3.1 Discrete State-Space Formulation

The motion of each tracked vehicle is modeled in an 8-dimensional state space:

$$\mathbf{x}_k = [u_k, v_k, s_k, r_k, \dot{u}_k, \dot{v}_k, \dot{s}_k, 0]^T$$

where:
- $(u_k, v_k)$ is the horizontal and vertical center of the bounding box.
- $s_k = w_k \times h_k$ is the bounding box area (scale).
- $r_k = w_k / h_k$ is the bounding box aspect ratio (assumed constant over short horizons).
- $(\dot{u}_k, \dot{v}_k, \dot{s}_k)$ are the linear velocities of center position and scale.

#### Discrete State Transition Equation:
$$\mathbf{x}_k = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$

where the state transition matrix $\mathbf{F} \in \mathbb{R}^{8 \times 8}$ under sampling interval $\Delta t = 1$ is:

$$\mathbf{F} = \begin{bmatrix}
\mathbf{I}_{4 \times 4} & \Delta t \mathbf{I}_{4 \times 4} \\
\mathbf{0}_{4 \times 4} & \mathbf{I}_{4 \times 4}
\end{bmatrix}$$

#### Discrete Measurement Equation:
When a YOLO detection is matched to the track, the observation vector is 4-dimensional:
$$\mathbf{z}_k = [u_m, v_m, s_m, r_m]^T$$
$$\mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$$

where the observation matrix $\mathbf{H} \in \mathbb{R}^{4 \times 8}$ is:
$$\mathbf{H} = \begin{bmatrix} \mathbf{I}_{4 \times 4} & \mathbf{0}_{4 \times 4} \end{bmatrix}$$

The process noise covariance $\mathbf{Q}$ and measurement noise covariance $\mathbf{R}$ are formulated based on the bounding box height $h$:
$$\mathbf{Q} = \operatorname{diag}\left( (\sigma_p h)^2, (\sigma_p h)^2, (\sigma_p h)^2, (\sigma_p h)^2, (\sigma_v h)^2, (\sigma_v h)^2, (\sigma_v h)^2, (\sigma_v h)^2 \right)$$
$$\mathbf{R} = \operatorname{diag}\left( (\sigma_m h)^2, (\sigma_m h)^2, (\sigma_m h)^2, (\sigma_m h)^2 \right)$$

where $\sigma_p = \frac{1}{20}$, $\sigma_v = \frac{1}{160}$, and $\sigma_m = \frac{1}{20}$.

---

### 3.2 Time Update (Prediction Steps)

Prior to receiving new detections at frame $k$, the prior state estimate $\mathbf{x}_{k|k-1}$ and prior error covariance $\mathbf{P}_{k|k-1}$ are extrapolated forward:

$$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1}$$
$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$$

---

### 3.3 Measurement Update (Correction Steps)

When a detection $\mathbf{z}_k$ is associated with the track:
1. **Measurement Innovation (Residual)**:
   $$\mathbf{y}_k = \mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1}$$
2. **Innovation Covariance**:
   $$\mathbf{S}_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R}$$
3. **Optimal Kalman Gain**:
   $$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}_k^{-1}$$
4. **Posterior State Estimate**:
   $$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k$$
5. **Posterior Error Covariance**:
   $$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$$

---

### 3.4 Bipartite Matching via the Hungarian Algorithm

Let $\mathcal{T} = \{t_1, t_2, \dots, t_M\}$ denote the active predicted tracks, and let $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$ denote the set of detected bounding boxes in the current frame.

The cost matrix $\mathbf{C} \in \mathbb{R}^{M \times N}$ is defined by the IoU distance:
$$C_{i,j} = 1 - \operatorname{IoU}(t_i, d_j)$$

The optimal assignment matrix $\mathbf{X} \in \{0, 1\}^{M \times N}$ is found by solving the classical assignment problem:
$$\min_{\mathbf{X}} \sum_{i=1}^M \sum_{j=1}^N C_{i,j} X_{i,j} \quad \text{subject to} \quad \sum_{j=1}^N X_{i,j} \le 1, \quad \sum_{i=1}^M X_{i,j} \le 1$$

This is solved in polynomial time $O(V^3)$ using the **Kuhn-Munkres (Hungarian)** algorithm or the **Jonker-Volgenant** shortest augmenting path solver. Matches with $C_{i,j} > (1 - \text{match\_thresh})$ are rejected.

---

### 3.5 ByteTrack Two-Stage Association Algorithm

Standard trackers discard all detections below a rigid confidence threshold (e.g., $<0.50$). ByteTrack retains all detections by partitioning them into two tiers:

$$\mathcal{D} = \mathcal{D}_{high} \cup \mathcal{D}_{low}$$
$$\mathcal{D}_{high} = \{d \in \mathcal{D} \mid \text{score}(d) \ge \tau_{high}\}$$
$$\mathcal{D}_{low} = \{d \in \mathcal{D} \mid \tau_{low} \le \text{score}(d) < \tau_{high}\}$$

In BharatANPR: $\tau_{high} = 0.50, \tau_{low} = 0.10$.

#### Algorithm Flow:
1. **Stage 1 Association**: Match all active tracks $\mathcal{T}_{active}$ with high-confidence detections $\mathcal{D}_{high}$ using IoU distance cost matrix $\mathbf{C}_1$.
   - Matched tracks $\to$ Kalman update.
   - Unmatched detections $\mathcal{D}_{remain} \to$ candidates for initializing new tracks.
   - Unmatched tracks $\mathcal{T}_{remain} \to$ passed to Stage 2.
2. **Stage 2 Association**: Match remaining tracks $\mathcal{T}_{remain}$ with **low-confidence detections** $\mathcal{D}_{low}$ using IoU distance cost matrix $\mathbf{C}_2$.
   - Matched tracks $\to$ Kalman update (preserves tracks through occlusion/blur!).
   - Unmatched tracks $\to$ state changed to `Lost` (retained in `track_buffer = 120` frames).
   - Unmatched low-confidence detections $\to$ discarded as background noise (never used to create new tracks).

---

## 4. Neural Sequence Recognition: CRNN + BiLSTM + CTC Loss

For character recognition on cropped license plate images, BharatANPR deploys a **Convolutional Recurrent Neural Network (CRNN)** trained with **Connectionist Temporal Classification (CTC)** loss.

---

### 4.1 CNN Feature Extraction & Map Slicing

The input image $\mathbf{I} \in \mathbb{R}^{H_{in} \times W_{in} \times C}$ (normalized to $32 \times 160 \times 1$ grayscale) passes through 7 convolutional layers with batch normalization and max-pooling:
- Standard pooling: $2 \times 2$ downsampling.
- Later pooling layers: $2 \times 1$ (asymmetric) downsampling, preserving horizontal resolution while compressing height.

The output feature tensor has dimensions:
$$\mathbf{X}_{feat} \in \mathbb{R}^{H' \times W' \times C'} \quad \text{where } H' = 1, \quad W' = T = 40, \quad C' = 512$$

Since $H' = 1$, each horizontal column slice along $W'$ represents a 512-dimensional feature vector $\mathbf{x}_t \in \mathbb{R}^{512}$ corresponding to a receptive field slice of the input plate from left to right:
$$\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T), \quad T = 40$$

---

### 4.2 Bidirectional LSTM Recurrent Modeling

To capture long-range contextual dependencies across character sequences, the feature sequence is fed into a 2-layer **Bidirectional Long Short-Term Memory (BiLSTM)** network with hidden size $H_{hid} = 256$.

At time step $t$:
- **Forward LSTM**:
  $$\overrightarrow{\mathbf{h}}_t = \operatorname{LSTM}_{fwd}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$
- **Backward LSTM**:
  $$\overleftarrow{\mathbf{h}}_t = \operatorname{LSTM}_{bwd}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
- **Concatenated Hidden State**:
  $$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \,;\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{512}$$

A linear projection layer followed by Softmax maps $\mathbf{h}_t$ to the alphabet probability distribution:
$$y_k^t = \frac{\exp(\mathbf{w}_k^T \mathbf{h}_t + b_k)}{\sum_{j=1}^{|\mathcal{L}'|} \exp(\mathbf{w}_j^T \mathbf{h}_t + b_j)}, \quad k \in \mathcal{L}'$$

where the alphabet $\mathcal{L}' = \mathcal{L} \cup \{\epsilon\}$ contains 36 alphanumeric characters (`0-9`, `A-Z`) plus the special **blank token** $\epsilon$ ($|\mathcal{L}'| = 37$).

---

### 4.3 Connectionist Temporal Classification (CTC) Theory

Let $\boldsymbol{\pi} = (\pi_1, \pi_2, \dots, \pi_T) \in \mathcal{L}'^T$ represent a sequence of frame-level label predictions. Assuming conditional independence across time steps given the feature representation $\mathbf{x}$:

$$P(\boldsymbol{\pi} \mid \mathbf{x}) = \prod_{t=1}^T y_{\pi_t}^t$$

Define the **CTC Collapse Operator** $\mathcal{B}: \mathcal{L}'^T \to \mathcal{L}^{\le T}$ as the sequence homomorphism that:
1. Collapses all consecutive identical non-blank characters into a single character.
2. Removes all blank tokens $\epsilon$.

#### Example:
$$\mathcal{B}(\text{--U-PP--7--88-}) = \text{UP78}$$

The conditional probability of a target license plate string $\mathbf{l} \in \mathcal{L}^{\le T}$ is the sum over all possible frame-level paths $\boldsymbol{\pi}$ that map to $\mathbf{l}$ under $\mathcal{B}$:

$$P(\mathbf{l} \mid \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{l})} P(\boldsymbol{\pi} \mid \mathbf{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^T y_{\pi_t}^t$$

The **CTC Objective Function** is the negative log-likelihood:

$$\mathcal{L}_{CTC} = -\ln P(\mathbf{l} \mid \mathbf{x}) = -\ln \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^T y_{\pi_t}^t$$

---

### 4.4 The Forward-Backward Algorithm for CTC

Directly summing over all paths in $\mathcal{B}^{-1}(\mathbf{l})$ is computationally intractable because the number of valid paths grows exponentially $O(|\mathcal{L}'|^T)$. CTC solves this in $O(T \times |\mathbf{l}'|)$ using **dynamic programming**.

Construct a modified target label sequence $\mathbf{l}'$ of length $L' = 2|\mathbf{l}| + 1$ by inserting the blank token $\epsilon$ between every character and at both ends:
$$\mathbf{l}' = (\epsilon, l_1, \epsilon, l_2, \epsilon, \dots, \epsilon, l_{|\mathbf{l}|}, \epsilon)$$

#### Forward Variable $\alpha_t(s)$:
$\alpha_t(s)$ represents the total probability of all path prefixes of length $t$ that map onto the prefix $\mathbf{l}'_{1:s}$:

$$\alpha_t(s) = \sum_{\substack{\boldsymbol{\pi}_{1:t}: \\ \mathcal{B}(\boldsymbol{\pi}_{1:t}) = \mathbf{l}'_{1:s}}} \prod_{\tau=1}^t y_{\pi_\tau}^\tau$$

#### Initialization ($t = 1$):
$$\alpha_1(1) = y_\epsilon^1, \quad \alpha_1(2) = y_{l_1}^1, \quad \alpha_1(s) = 0 \quad \forall s > 2$$

#### Recursive Transition ($t > 1$):
$$\alpha_t(s) = \begin{cases}
\left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) \right) y_{l'_s}^t & \text{if } l'_s = \epsilon \text{ or } l'_s = l'_{s-2} \\
\left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2) \right) y_{l'_s}^t & \text{otherwise}
\end{cases}$$

#### Termination:
$$P(\mathbf{l} \mid \mathbf{x}) = \alpha_T(L') + \alpha_T(L' - 1)$$

---

### 4.5 Exact Analytical Gradient Derivation

During backpropagation, we compute the analytical gradient of $\mathcal{L}_{CTC}$ with respect to the unnormalized activation logits $a_k^t$ at time step $t$ for character class $k$:

$$\frac{\partial \mathcal{L}_{CTC}}{\partial a_k^t} = y_k^t - \frac{1}{P(\mathbf{l} \mid \mathbf{x})} \sum_{s \in \{s \mid l'_s = k\}} \alpha_t(s) \beta_t(s)$$

where $\beta_t(s)$ is the backward variable computed from $t = T$ down to $1$:
$$\beta_t(s) = \sum_{\substack{\boldsymbol{\pi}_{t:T}: \\ \mathcal{B}(\boldsymbol{\pi}_{t:T}) = \mathbf{l}'_{s:L'}}} \prod_{\tau=t}^T y_{\pi_\tau}^\tau$$

This elegant gradient formula shows that backpropagation simply computes the difference between the network's predicted probability $y_k^t$ and the posterior probability that character $k$ occurs at time $t$ conditioned on the true sequence $\mathbf{l}$.

---

### 4.6 Prefix Beam Search Decoding with Language Priors

Instead of simple greedy argmax decoding ($\pi_t^* = \arg\max_k y_k^t$), which often fails on ambiguous character boundaries, BharatANPR utilizes **Prefix Beam Search Decoding**.

Let $\ell$ be a candidate prefix string. At each time step $t$, we maintain two separate path probabilities for each prefix $\ell$:
- $P_b(\ell, t)$: The probability that the prefix ends in a **blank token** $\epsilon$.
- $P_{nb}(\ell, t)$: The probability that the prefix ends in a **non-blank character**.

The total probability is:
$$P_{tot}(\ell, t) = P_b(\ell, t) + P_{nb}(\ell, t)$$

At step $t$, for each candidate prefix $\ell$ in the beam of width $B = 10$, we evaluate two branching operations for every character $c \in \mathcal{L}'$:
1. If $c = \epsilon$:
   $$P_b(\ell, t) = P_{tot}(\ell, t-1) \times y_\epsilon^t$$
2. If $c \ne \epsilon$:
   - If $c = \operatorname{last}(\ell)$:
     $$P_{nb}(\ell \cdot c, t) = P_b(\ell, t-1) \times y_c^t \quad \text{(blank separated identical letters)}$$
     $$P_{nb}(\ell, t) = P_{nb}(\ell, t-1) \times y_c^t \quad \text{(consecutive repeat collapsed)}$$
   - If $c \ne \operatorname{last}(\ell)$:
     $$P_{nb}(\ell \cdot c, t) = P_{tot}(\ell, t-1) \times y_c^t$$

Pruning retains only the top $B$ highest $P_{tot}(\ell, t)$ prefixes at each step, yielding superior decoding on noisy plate boundaries.

---

## 5. Permutation Autoregressive Sequence (PARSeq) Vision Transformer

For degraded, partially occluded, or low-contrast plates, BharatANPR activates **PARSeq (Permutation Autoregressive Sequence)** as an ensemble/fallback OCR engine.

---

### 5.1 Multi-Head Self-Attention Mechanics

PARSeq utilizes a 12-layer Vision Transformer (ViT) encoder. The plate image is split into non-overlapping patches $\mathbf{x}_p \in \mathbb{R}^{P^2 \times C}$, linearly projected into $D = 384$ embedding dimensions, and added to 1D learnable position embeddings $\mathbf{E}_{pos}$:

$$\mathbf{z}_0 = [\mathbf{x}_{p}^1 \mathbf{E}; \dots; \mathbf{x}_{p}^N \mathbf{E}] + \mathbf{E}_{pos}$$

The core computation within each transformer block is **Multi-Head Self-Attention (MHSA)**:

$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

where $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q$, $\mathbf{K} = \mathbf{X} \mathbf{W}_K$, $\mathbf{V} = \mathbf{X} \mathbf{W}_V$, and $d_k = D / h$ is the feature dimension per head.

---

### 5.2 Permutation Language Modeling Formulation

Standard autoregressive language models predict tokens strictly in left-to-right order:
$$P(\mathbf{y} \mid \mathbf{x}) = \prod_{i=1}^K P(y_i \mid y_{<i}, \mathbf{x})$$
However, on damaged license plates, the leftmost characters (e.g., the State code `UP`) might be legible, while an intermediate digit is blurred, but the rightmost digits (`7951`) are sharp. Left-to-right decoding fails to utilize right-side visual context to predict intermediate characters.

PARSeq resolves this through **Permutation Language Modeling (PLM)**. During training, it optimizes the model over all $K!$ possible permutations of token factorization order $\mathbf{z} \in \mathcal{S}_K$:

$$\mathcal{L}_{PARSeq} = - \mathbb{E}_{\mathbf{z} \sim \mathcal{S}_K} \left[ \sum_{k=1}^K \log P(y_{z_k} \mid y_{z_{<k}}, \mathbf{x}) \right]$$

#### Dual Decoding Capability:
1. **Non-Autoregressive Mode (1-step)**: Generates all characters simultaneously in parallel ($O(1)$ forward pass) for ultra-low latency ($<5\text{ ms}$).
2. **Autoregressive Iterative Refinement (k-steps)**: Re-feeds predictions back into the cross-attention decoder, using bidirectional language context to resolve visually ambiguous letters.

---

## 6. Image Preprocessing & Morphological Mathematics

Before passing plate crops to OCR, BharatANPR applies a deterministic image conditioning pipeline to maximize high-frequency character edge gradients.

---

### 6.1 Contrast Limited Adaptive Histogram Equalization (CLAHE)

Standard Global Histogram Equalization flattens the cumulative distribution function (CDF) across the entire image:
$$s_k = T(r_k) = (L-1) \sum_{j=0}^k p_r(r_j)$$
In license plate images, headlights, streetlights, or sunlight cause local over-exposure. Global equalization amplifies background noise and washes out character edges.

**CLAHE** partitions the image into non-overlapping contextual tiles (e.g., $8 \times 8$ grid). In each tile, the local histogram $h(i)$ is clipped at a predetermined threshold $\beta$:

$$h_{clipped}(i) = \min(h(i), \beta)$$

The total number of clipped pixels $N_{clipped} = \sum_i \max(0, h(i) - \beta)$ is redistributed uniformly across all gray-level bins $L$:

$$h_{final}(i) = h_{clipped}(i) + \frac{N_{clipped}}{L}$$

Bilinear interpolation between tile boundaries eliminates artificial border seams.

---

### 6.2 Otsu's Global Binarization via Variance Maximization

To segment black characters from reflective white/yellow plate backgrounds, BharatANPR computes the mathematically optimal binarization threshold $t^* \in [0, 255]$ using **Otsu's method**.

Let the normalized gray-level histogram be $p(i)$ for $i \in [0, L-1]$. For a candidate threshold $t$, the image pixels are divided into two classes:
- Background $\mathcal{C}_0 = [0, t]$
- Foreground $\mathcal{C}_1 = [t+1, L-1]$

Class probabilities:
$$\omega_0(t) = \sum_{i=0}^t p(i), \quad \omega_1(t) = \sum_{i=t+1}^{L-1} p(i) = 1 - \omega_0(t)$$

Class means:
$$\mu_0(t) = \sum_{i=0}^t \frac{i \cdot p(i)}{\omega_0(t)}, \quad \mu_1(t) = \sum_{i=t+1}^{L-1} \frac{i \cdot p(i)}{\omega_1(t)}$$

Total mean of the image:
$$\mu_T = \omega_0(t)\mu_0(t) + \omega_1(t)\mu_1(t)$$

The **Between-Class Variance** $\sigma_B^2(t)$ is:
$$\sigma_B^2(t) = \omega_0(t) (\mu_0(t) - \mu_T)^2 + \omega_1(t) (\mu_1(t) - \mu_T)^2 = \omega_0(t)\omega_1(t) \left(\mu_0(t) - \mu_1(t)\right)^2$$

The optimal threshold $t^*$ maximizes between-class variance:
$$t^* = \arg\max_{0 \le t < L-1} \sigma_B^2(t)$$

Since total variance $\sigma_T^2 = \sigma_W^2(t) + \sigma_B^2(t)$ is constant, maximizing $\sigma_B^2(t)$ strictly minimizes the **Within-Class Variance** $\sigma_W^2(t)$, guaranteeing optimal foreground-background separation.

---

### 6.3 Bilateral Filtering for Edge-Preserving Denoising

Gaussian smoothing blurs noise, but simultaneously destroys the crisp vertical strokes of characters like `I`, `1`, `D`, and `0`. BharatANPR applies a **Bilateral Filter**, which combines spatial distance weighting with photometric intensity range weighting:

$$I^{filtered}(\mathbf{p}) = \frac{1}{W_{\mathbf{p}}} \sum_{\mathbf{q} \in \Omega} I(\mathbf{q}) \, g_s(\|\mathbf{p} - \mathbf{q}\|) \, f_r(|I(\mathbf{p}) - I(\mathbf{q})|)$$

where:
- $\Omega$ is the spatial neighborhood centered at pixel $\mathbf{p} = (x, y)$.
- $g_s(\|\mathbf{p} - \mathbf{q}\|) = \exp\left( -\frac{\|\mathbf{p} - \mathbf{q}\|^2}{2\sigma_s^2} \right)$ is the **spatial Gaussian kernel** (penalizes geometric distance).
- $f_r(|I(\mathbf{p}) - I(\mathbf{q})|) = \exp\left( -\frac{|I(\mathbf{p}) - I(\mathbf{q})|^2}{2\sigma_r^2} \right)$ is the **range Gaussian kernel** (penalizes intensity difference).
- $W_{\mathbf{p}}$ is the normalization factor:
  $$W_{\mathbf{p}} = \sum_{\mathbf{q} \in \Omega} g_s(\|\mathbf{p} - \mathbf{q}\|) \, f_r(|I(\mathbf{p}) - I(\mathbf{q})|)$$

#### Property:
Across uniform plate areas, $|I(\mathbf{p}) - I(\mathbf{q})| \approx 0 \implies f_r \approx 1$, behaving as standard Gaussian smoothing to remove noise. Across sharp character edges, $|I(\mathbf{p}) - I(\mathbf{q})| \gg 0 \implies f_r \approx 0$, suppressing smoothing and preserving razor-sharp character borders.

---

*This concludes the mathematical and algorithmic reference for BharatANPR.*