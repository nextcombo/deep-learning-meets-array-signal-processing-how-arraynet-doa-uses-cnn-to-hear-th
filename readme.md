# Deep Learning Meets Array Signal Processing: How ArrayNet-DOA Uses CNN to "Hear" the Direction of Weak Signals [with Python Code]

When Deep Learning Meets Array Signal Processing: How ArrayNet-DOA "Hears" the Direction of Weak Signals
Background: Why is direction of arrival (DoA) estimation important in radar, sonar, and 5G communication? DoA estimation is a classic problem in array signal processing. Simply put, it involves receiving electromagnetic wave signals from a distant source through an array of regularly arranged antennas and then "inversely" determining where these signals are coming from. This process is akin to your ability to roughly determine the speaker's location by only listening with one ear, using just two ears as your "ears."

As the number of sensors increases from "two ears" to "16 antenna elements," the system can more accurately distinguish multiple signal sources.
Classic algorithms and their challenges: In the past few decades, MUSIC (Multiple Signal Classification) and Root-MUSIC have been the "gold standard" for DoA estimation. Their core idea is to:
Perform feature decomposition on the covariance matrix of received signals
Utilize the orthogonality between the signal subspace and noise subspace
Locate the signal direction through search or polynomial rooting
However, a common fatal flaw of these methods is that when the signal-to-noise ratio (SNR) drops to -10 dB or even lower, the estimation of the noise subspace becomes severely biased, leading to a sharp increase in angle estimation errors. Under extreme conditions with only 100 snapshots, traditional methods almost completely fail.
This is the problem that ArrayNet-DOA aims to solve.
2. Core Innovation: Learning "Directional Codes" in Covariance Matrices with CNN ArrayNet-DOA
Based on the research of Papageorgiou et al., the core idea can be summarized in one sentence:  
Do not calculate direction angles using analytical formulas; instead, let the neural network learn to "look at the covariance matrix and guess the signal direction" directly from massive data. Specifically, the design of the entire system follows several key ideas:  
2.1 Signal Model: For a uniform linear array (ULA) system with 16 antenna elements (arranged equidistantly, with a separation of half a wavelength), each element receives a signal that can be represented as:  
Y = A(θ) · S + E  
where:  
Y: Received signal matrix (16 × number of snapshots)  
A(θ): Array manifold matrix, each column is the steering vector corresponding to an angle  
S: Signal source matrix  
E: Additive Gaussian white noise  
From this received signal, the covariance matrix Ry (16 × 16) is calculated, which contains the core information about the signal direction.  
2.2 Data Preprocessing: The three-channel "portrait" of the covariance matrix is a complex matrix consisting of individual 16×16 matrices. To enable CNN processing, the team decomposes it into three real-number channels:  
Channel  
Meaning  
Function  
Channel 1  
The real part Re{Ry} of the covariance matrix  
Space distribution of signal energy  
Channel 2  
The imaginary part Im{Ry} of the covariance matrix  
Symmetry of phase relationships  
Channel 3  
Phase angle angle{Ry} of the covariance matrix  
Information about phase differences between different elements.
The result is that each sample becomes a 16 × 16 × 3 tensor, similar to an "image" with three channels. The CNN's convolutional kernel can slide over this "image," capturing the spatial correlation between adjacent elements.
2.3 Label Design: The angle space encoding angle search range is set to -60° to +60°, gridded at a resolution of 1°, resulting in 121 grid points. The labels are a 121-dimensional vector, where the grid points corresponding to the true signal direction are marked as 1, and the rest are 0.
This design transforms the DoA estimation problem into a multi-label binary classification problem: the neural network needs to output a probability value (between 0 and 1) for each grid point, indicating whether there is a signal source in that direction.
III. Network Architecture: The ArrayNet-DOA architecture is meticulously designed following the classic paradigm of "convolution extracting features → fully connected making decisions." It consists of a total of 24 layers, approximately 282 million trainable parameters.
Architectural Panorama Input Layer: 16×16×3 Tensor (Covariance Matrix with Three Channels)
The first layer uses a 3x3 convolutional kernel with a stride of 2, reducing the spatial size from 16x16 to 7x7.
The subsequent three layers use a 2x2 convolutional kernel with a stride of 1, progressively compressing the feature maps to 4x4.
256 filters per layer to ensure sufficient feature representation capability.
After each convolution layer, batch normalization (BatchNorm) and ReLU activation are followed to prevent gradient vanishing.
2. Fully connected part: Deep Perception Decision
Three-layer fully connected (4096 → 2048 → 1024), layer by layer dimensionality reduction
Dropout (drop rate 20%) is used after each fully connected layer to prevent overfitting.
The final output layer uses a Sigmoid function to provide the probability distribution of 121 grid points.
3. Key insights
The covariance matrix inherently has spatial structure—the elements between adjacent elements in the kernel have strong correlations. Convolutional kernels are particularly adept at capturing this local spatial pattern, which is fundamental to why CNNs are better suited for this task than purely fully connected networks.
IV. Training strategy: Teach with "ideal data" and test with "real data"
4.1 A key design choice in generating training data is to use the true covariance matrix (True Covariance) instead of the sampled covariance matrix estimated from limited fast Fourier transforms.
Why do this? Because:
The true covariance matrix is an "ideal version" without sampling noise, directly derived from the signal model.
If sampling noise is mixed into the training phase, the network may learn incorrect mapping relationships.
Using the sampled covariance matrix in testing can assess the network's generalization ability and robustness.
The training set covers 5 SNR levels (-20, -15, -10, -5, 0 dB), and for each SNR level, all possible combinations of two source angles (121 grid points, randomly selected 2, totaling 7260 combinations) are traversed. In total, 36,300 training samples are used.
4.2 Training configuration parameters
Configuration settings
Description
Optimizer
Adam
Adaptive learning rate
Initial learning rate
0.001
Combined with ReduceLROnPlateau decay
Batch size
32
Moderate batch size
Maximum number of training iterations
200
Combined with early stopping mechanism
Early stopping patience value
25
Stop training if validation loss does not decrease
Corresponding to multi-label binary classification.
Validation set ratio
20%  
Randomly divide during training.
V. Performance Comparison: CNN vs MUSIC vs Root-MUSIC
5.1 RMSE comparison under fixed SNR at SNR = -10 dB, T = 2000, and two source angle intervals of 4.7°
Algorithm
RMSE  
CNN0.43°Root-MUSIC0.39°MUSIC11.05°  
Under a -10 dB low signal-to-noise ratio, the traditional MUSIC algorithm almost collapses (with an error of up to 11°), while CNN and Root-MUSIC can maintain extremely high precision.
5.2 RMSE Variation with SNR (Core Experiment) Two fixed signal sources (approximately 10° and 13°), a constant number of frames set to 1000, SNR scanned from -20 dB to +30 dB:
SNR  
CNN  
MUSIC  
R-MUSIC  
-20 dB  
18.60°24.83°  
26.31°  
-15 dB  
2.70°22.25°  
20.40°  
-10 dB  
0.42°9.62°  
0.51°  
-5 dB  
0.24°  
1.85°  
0.13°0 dB  
0.23°  
0.23°  
0.08°+10 dB  
0.23°  
0.23°  
0.02°  
Key Findings:
1  
Low SNR region (-20 ~ -15 dB): CNN is the only algorithm that can still work, MUSIC and Root-MUSIC both have errors exceeding 20°, basically failed.
2  
Middle SNR region (-10 dB): CNN achieves sub-degree accuracy, MUSIC still has nearly 10° error.
3  
High SNR region (≥ 0 dB): CNN performance tends to be stable (limited by 1° grid resolution), Root-MUSIC exhibits ultra-high precision.
4  
CNN's low SNR advantage: At -15 dB, the RMSE of CNN is only 2.70°, which is about 20° lower than MUSIC, demonstrating the strong modeling ability of deep learning for noise patterns.
5.3 RMSE remains constant at -10 dB SNR = -10 dB as the number of rapid shots varies from 100 to 10000:
Snapshot count T
CNN  
MUSIC  
R-MUSIC  
100  
6.84°21.89°  
20.83°  
200  
0.92°18.22°  
7.56°  
500  
0.59°  
12.82°  
0.59°1000  
0.47°  
7.57°  
0.32°2000  
0.39°  
2.48°  
0.19°10000  
0.38°  
0.33°  
0.08°  
Key Findings:
1  
Extremely fast-sampling scenario (T=100): CNN is the only algorithm that maintains usable accuracy (6.84° vs over 20°)
2  
CNN can achieve an accuracy of about 1 degree with only 200 frames, while MUSIC needs about 5000 frames to reach a similar level.
3  
In the case of a large number of rapid shots (T ≥ 2000), Root-MUSIC gradually surpasses CNN, but this is often unrealistic - the number of rapid shots available in actual radar and communication systems is usually extremely limited.
6. Why does CNN "dominate" traditional algorithms under low signal-to-noise ratios? This is a question worth deep reflection. The performance bottleneck of traditional MUSIC-like algorithms lies in the accuracy of subspace decomposition—when noise power far exceeds signal power (low SNR), it becomes difficult to reliably distinguish between the signal subspace and the noise subspace after feature decomposition.
The strength of CNN comes from three aspects:  
1. End-to-end implicit feature learning  
CNNs do not need to explicitly perform feature decomposition or subspace separation. They learn an end-to-end mapping directly from the raw values of the covariance matrix, from "matrix patterns" to "directional angles." This implicit learning avoids error accumulation through intermediate steps.  
2. Statistical modeling of noise distribution  
By jointly training on data at multiple SNR levels, CNNs inherently learn the pattern of noise power changes affecting the covariance matrix. In other words, the network "sees" data at various noise levels, knowing how to extract effective information from noisy inputs.  
3. Convolutional kernels capturing local spatial correlation  
In ULA arrays, the phase difference between adjacent elements receiving signals is directly related to the signal direction. The sliding operation of convolutional kernels in small ranges naturally suits capturing this local phase structure, which is more efficient and robust compared to global feature decomposition.  
7. Limitations of ArrayNet-DOA and future directions Despite ArrayNet-DOA's outstanding performance under low signal-to-noise ratios, there are some noteworthy limitations:  
Limitation  
Description  
Fixed number of source sources The current model assumes K=2 as known, making it difficult to directly handle scenarios with unknown or variable K.  
Grid resolution limitation The output precision is limited by a 1° grid interval, unable to achieve 0.001° level precision like Root-MUSIC under high SNR.  
Array structure dependence The model is trained for a 16-element ULA, requiring retraining when changing array structures.
The GPU is required for training a model with approximately 28.2 million parameters, presenting challenges in deployment to embedded devices.
Future improvements could include:
- Introducing grid refinement strategies or unstructured methods to break through the resolution bottleneck of 1°.
- Training models that adapt to an increasing number of source signals, enabling them to handle scenarios with K unknowns.
- Using model compression techniques (quantization, pruning, knowledge distillation) to reduce computational overhead.
- Expanding to two-dimensional DoA estimation (azimuth + elevation angle) scenarios.
VIII. Summary ArrayNet-DOA demonstrates an effective paradigm for integrating deep learning into classical array signal processing problems. Its core idea—training with real covariance matrices and testing with sampled covariance matrices—ensures excellent generalization capabilities under low SNR conditions.
Experimental results show that CNNs surpass traditional MUSIC and Root-MUSIC algorithms by an overwhelming margin in the extremely low SNR range from -20 to -10 dB. Especially under extreme conditions where the number of samples is insufficient (T ≤ 200), CNNs are the only solution capable of maintaining usable accuracy.
This achievement provides a novel solution approach for radar target detection, underwater sonar positioning, and beam management in 5G/6G scenarios.

## Images

![Deep Learning Meets Array Signal Processing: How ArrayNet-DOA "Hears" the Direction of Weak Signals with CNN [with Python code]](01.jpg)
![Deep Learning Meets Array Signal Processing: How ArrayNet-DOA "Hears" the Direction of Weak Signals with CNN [with Python code]](02.jpg)
![Deep Learning Meets Array Signal Processing: How ArrayNet-DOA "Hears" the Direction of Weak Signals with CNN [with Python code]](03.jpg)
![Deep Learning Meets Array Signal Processing: How ArrayNet-DOA "Hears" the Direction of Weak Signals with CNN [with Python code]](04.jpg)
![Deep Learning Meets Array Signal Processing: How ArrayNet-DOA "Hears" the Direction of Weak Signals with CNN [with Python code]](05.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
