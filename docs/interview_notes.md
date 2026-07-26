Digital Pathology Copilot:

✅ ResNet18
✅ EfficientNet
✅ Vision Transformer
✅ GradCAM
✅ Attention Rollout
✅ Medical Segmentation (MONAI UNet)
✅ FastAPI
✅ Docker
✅ AWS Deployment
✅ MLflow
✅ TensorBoard
✅ Monitoring
✅ RAG
✅ Clinical Copilot

Classification predicts a single label for the entire image, whereas segmentation assigns a class label to every pixel, enabling precise localization of anatomical structures or lesions.
The industry standard for medical segmentation is: UNet

Developed specifically for biomedical imaging.
Why UNet?

Unlike ResNet,

UNet preserves

Spatial Information

through skip connections.

Architecture

Image

↓

Encoder

↓

Bottleneck

↓

Decoder

↓

Mask

New Metrics

Classification

Accuracy

Precision

Recall

F1

ROC-AUC

Segmentation

Dice Score ⭐⭐⭐⭐⭐

IoU

Hausdorff Distance

## Modules:

1. Training

---

Train CNN

Train ViT

Compare Models

---

2. Inference

---

Predict

Confidence

JSON

---

3. Explainability

---

GradCAM

Attention Rollout

---

4. Whole Slide Images

---

OpenSlide

Patch Extraction

Aggregation

---

5. Clinical AI

---

LLM

RAG

Medical Guidelines

Clinical Report

---

6. Deployment

---

FastAPI

Docker

AWS

Monitoring

1. What problem did EfficientNet solve?

Before EfficientNet, researchers scaled CNNs in three ways:

Option 1: Increase Depth
ResNet18
↓
ResNet34
↓
ResNet50
↓
ResNet101

More layers → better features, but training becomes slower.

Option 2: Increase Width
64 Channels
↓
128 Channels
↓
256 Channels

More channels → more expressive features, but parameter count increases rapidly.

Option 3: Increase Image Resolution
224 × 224
↓
384 × 384
↓
512 × 512

Higher resolution preserves more detail but increases computation significantly.

The Problem

Researchers asked:

Which of these should we increase?

There was no principled answer.

2. EfficientNet's Big Idea

Instead of scaling one dimension, EfficientNet scales all three together.

Depth +
Width +
Resolution

This is called Compound Scaling.

Interview answer:

EfficientNet uses compound scaling to balance network depth, width, and input resolution using a single scaling coefficient, leading to better accuracy with fewer parameters and FLOPs.

3. EfficientNet Architecture

EfficientNet-B0 is built using MBConv blocks.

Input
│
Stem Conv
│
MBConv
│
MBConv
│
MBConv
│
Head Conv
│
Global Average Pool
│
Classifier

Unlike ResNet, it does not primarily use residual blocks.

4. What is an MBConv Block?

This is one of the most important interview questions.

MBConv stands for:

Mobile Inverted Bottleneck Convolution

Flow:

Input
│
1×1 Expansion Conv
│
Depthwise Conv
│
Squeeze-and-Excitation
│
1×1 Projection Conv
│
Skip Connection

Let's understand each step.

Expansion Layer

Instead of

32 Channels

we first expand

32

↓

192

Why?

The network learns richer feature representations in a higher-dimensional space.

Depthwise Convolution

Instead of one convolution operating across all channels:

64 input channels

↓

64 filters

Depthwise convolution applies one filter per input channel.

Advantages:

Much fewer parameters
Faster computation
Lower memory usage

This is why EfficientNet is computationally efficient.

Squeeze-and-Excitation (SE)

This module learns which channels are important.

Imagine 64 feature maps.

Some detect:

nuclei
tissue texture
background
blood vessels

SE learns to assign higher weights to informative channels and suppress less useful ones.

Interview answer:

SE blocks improve feature representation by recalibrating channel-wise responses using learned attention weights.

Projection Layer

After expansion:

192 Channels

Project back to

32 Channels

This keeps the model compact while preserving useful information.

5. Fine-Tuning Strategy

For medical imaging, a common approach is:

Stage 1

Freeze the backbone.

Train only the classifier.

model.freeze_backbone()

Train for 5–10 epochs.

Stage 2

Unfreeze the last few blocks.

Use a smaller learning rate.

LR

0.001

↓

0.0001

This adapts pretrained features without overwriting them too aggressively.

Stage 3

Unfreeze the entire network.

Fine-tune for a few more epochs.

6. Why is EfficientNet Better than ResNet?
   Feature ResNet18 EfficientNet-B0
   Parameters ~11.7M ~5.3M
   FLOPs Higher Lower
   Accuracy Good Better for similar compute
   Mobile Friendly Moderate Excellent

The exact numbers depend on the task, but EfficientNet generally provides a better accuracy-to-compute trade-off.

Vision Transformers:
Problem: CNNs are excellent at Local Features but they struggle with Global Relationships unless they become very deep.

Researchers asked Instead of looking at 3×3

Why not look at Everything at once?

That idea came from NLP: Transformers.

How does ViT convert an image into transformer input?

Answer: A Vision Transformer divides the image into fixed-size patches. Each patch is flattened, projected into an embedding vector, and treated as a token. The sequence of patch embeddings is then processed by the transformer encoder.

Amazing Observation

Look carefully.

GPT:

Words

↓

Embedding

↓

Position

↓

Transformer

ViT:

Patches

↓

Embedding

↓

Position

↓

Transformer

Same architecture.

Only the input changes.

Why do we need a CLS token?

Answer: The CLS token aggregates information from all image patches through self-attention and serves as the global representation used for image classification.

Shape Analysis--------

Input

(2,3,96,96)

After Conv

(2,768,6,6)

Flatten

(2,768,36)

Transpose

(2,36,768)

CLS

(2,1,768)

Concatenate

(2,37,768)

Add Position

(2,37,768)

Perfect.

The Heart of Every Transformer:

Now comes the most important block:

Q (Query)

K (Key)

V (Value)

↓

Scaled Dot Product Attention

↓

Multi-Head Attention

This is the core of:

GPT
BERT
ViT
CLIP
DINOv2
Llama
Gemma
Claude

---

## CNN vs Attention

CNN

Image

↓

3×3

↓

3×3

↓

3×3

↓

Feature

Attention

Image

↓

Every Patch

↓

Looks at Every Other Patch

↓

Feature

CNN sees locally.

Attention sees globally.

## Entire Flow

Image

↓

Patch Embedding

↓

Q

K

V

↓

QKᵀ

↓

Scale

↓

Softmax

↓

Weights

↓

Weighted Sum of V

↓

New Embeddings

This is the core of every transformer.

We need Q (Query), K (Key), and V (Value) instead of just Q because they separate the search task, the content match, and the actual data retrieval. Using only Q limits the model's ability to measure relationships and pass rich information forward.

What would happen if we removed Softmax?
Answer: Removing Softmax from a transformer's self-attention mechanism destroys the probability distribution of attention weights, causing unstable training dynamics, explosive numerical growth, and severe accuracy loss unless replaced by alternative normalizations like linear kernels or ReLU.

Why is Attention better than CNN for modeling long-range relationships in an image?
Answer: Attention mechanisms are better than Convolutional Neural Networks (CNNs) at modeling long-range relationships in images because they offer a global receptive field in a single step, dynamic content-based weighting, and better preservation of spatial context without the degradation challenges of stacking deep layers.

Why does Attention output the same shape as its input?

Answer: Attention updates the representation of each token by incorporating contextual information from all other tokens, but preserves the sequence length and embedding dimension.

Q1: Why split into multiple heads?

Answer: Multiple attention heads allow the model to learn different relationships in parallel. Each head focuses on different aspects of the input, leading to richer feature representations.

Q2: Why must embed_dim be divisible by num_heads?

Example:

768

↓

12

↓

64

If

770

↓

12

Impossible.

Every head must receive the same embedding size.

Q3: Why concatenate instead of averaging?

Answer: Concatenation preserves the information learned by each attention head. Averaging would merge them too early and reduce the model's representational capacity.

Q4: What does the final linear layer do?

Answer: The output projection mixes information from all attention heads into a single embedding representation of the original dimension.

Why residual connections?

Residual connections preserve the original representation, improve gradient flow during backpropagation, and make it easier to train deep transformer networks.

Why LayerNorm?

Imagine these feature values: 100 -250 450 -900

Training becomes unstable because feature magnitudes vary widely.

LayerNorm normalizes each token's feature vector.

Unlike BatchNorm:

BatchNorm normalizes across the batch.
LayerNorm normalizes within each sample.

That's why transformers use LayerNorm.

Why LayerNorm instead of BatchNorm?

LayerNorm normalizes features within each token independently and does not depend on batch statistics, making it suitable for variable sequence lengths and transformer architectures.

Feed Forward Network (MLP)

After attention, each token passes through an MLP.

Architecture:

768

↓

3072

↓

GELU

↓

768

Notice:

Attention mixes information between tokens.

The MLP processes each token independently.

Both are needed.

Q1. Why do we need both Attention and an MLP?

Answer: Self-attention models interactions between different tokens, while the feed-forward network applies nonlinear transformations independently to each token. Together they provide both contextual reasoning and feature transformation.

Q2. Why does the MLP expand from 768 to 3072?

Answer: Expanding to a higher-dimensional hidden space increases the model's representational capacity. The projection back to the original embedding size keeps the interface between transformer blocks consistent.

Q3. Why use GELU instead of ReLU?

Answer: GELU provides a smoother activation by weighting inputs based on their value instead of applying a hard threshold like ReLU. It has been shown to improve optimization in transformer models.

Q4. Why use Pre-LayerNorm?

Answer: Pre-LayerNorm improves gradient flow and training stability, especially in deep transformer architectures, making optimization easier than the original Post-LayerNorm design.

How is ViT different from ResNet?

ResNet-------------------------------------------- ViT
Learns local features using convolutions --------- Learns global relationships using self-attention
Strong inductive bias ---------------------------- Relies more on data
Better for smaller datasets ---------------------- Excels with large-scale pretraining
Translation equivariant -------------------------- Flexible global context modeling
