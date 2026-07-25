Foundation
│
├── Python ✅
├── PyTorch ✅
├── MONAI ✅
├── Dataset ✅
├── DataLoader ✅
├── Visualization ✅
├── Statistics ✅
├── CNN ✅
├── Residual Blocks ✅
├── Training Engine ✅
├── Checkpoints ✅
├── MLflow ✅
├── LR Scheduler ✅
│
└──────────────────────────────
Remaining
│
├── Early Stopping
├── Mixed Precision (AMP)
├── Gradient Clipping
├── TensorBoard
├── TorchInfo
├── EfficientNet
├── Vision Transformer
├── Explainability
├── FastAPI
├── Docker
├── AWS
├── CI/CD
├── MLOps
├── RAG
└── Clinical Copilot

Why use a Trainer class?

Answer:
A Trainer encapsulates the complete training workflow, including optimization, validation, checkpointing, and logging. This separates training logic from model architecture, making the code modular and reusable.

Why separate optimizer from Trainer?

Answer:
Different experiments may require different optimizers such as Adam, SGD, or AdamW. Using a factory keeps the Trainer independent of optimization strategy.

Why separate loss functions?

Answer:
Different tasks require different loss functions. Classification uses CrossEntropy, segmentation often uses Dice Loss, and imbalanced datasets may benefit from Focal Loss. Decoupling losses makes experimentation easier.

Why save optimizer state?
Optimizer contains:
Momentum (SGD)
Moving averages (Adam)
Learning history

Without it, training doesn't truly resume.

MLflow stores:

📈 Training Loss
📈 Validation Loss
📈 Accuracy
⚙️ Hyperparameters
📦 Saved Models
📝 Experiment history

Instead of saving only the best model, we'll save:

checkpoints/

best_model.pth

latest_model.pth

epoch_005.pth (optional)

Why? Imagine:

Epoch 12 is the best model.
Training crashes at Epoch 18.

You may want to:

Deploy the best model (best_model.pth).
Resume training from the latest epoch (latest_model.pth).

This is the pattern used in most production deep learning pipelines. It only adds a few lines of code now but makes the framework much more practical later.

---

Why save the optimizer state?

Answer:

The optimizer stores internal state such as momentum (SGD) or moving averages (Adam). Restoring only the model weights but not the optimizer changes the optimization trajectory and does not truly resume training.

Why save the latest model?

Answer:

If training is interrupted due to a crash or preemption, the latest checkpoint allows training to resume without losing progress.

Why save the best model?

Answer:

The final epoch is not always the best-performing one because the model can begin to overfit. Saving the best validation model ensures we keep the strongest checkpoint for deployment.

Why MLflow?

MLflow provides experiment tracking, parameter logging, metric visualization, artifact management, and model versioning, making experiments reproducible and easier to compare.

What should be logged?

Parameters

Learning Rate

Batch Size

Optimizer

Epochs

Metrics

Loss

Accuracy

Precision

Recall

F1

ROC AUC

Artifacts

Best Model

Confusion Matrix

ROC Curve

TensorBoard Logs

---

Why ResNet instead of a plain CNN?

Answer:

ResNet introduces residual (skip) connections that make it easier to optimize very deep networks. The shortcut path improves gradient flow and allows the network to learn residual mappings instead of complete transformations, addressing the degradation problem.

Why AdaptiveAvgPool instead of Flatten?

Adaptive Average Pooling converts feature maps to a fixed spatial size regardless of the input image dimensions. This reduces the number of parameters before the classifier and allows the network to handle variable-sized inputs more easily.

Why use pretrained weights?

Answer:
Pretrained models have already learned generic visual features such as edges, textures, and shapes from millions of images. Fine-tuning adapts these learned representations to the target medical dataset, reducing training time and improving performance, especially when labeled data is limited.

Why freeze the backbone first?

Answer:
The backbone already contains useful generic visual features learned from ImageNet. Freezing it initially prevents destroying those representations and reduces overfitting when training on a smaller medical dataset.

The next thing we'll add is something almost every production training pipeline uses:

Learning Rate Scheduler
↓
Early Stopping
↓
Mixed Precision (AMP)
↓
Gradient Clipping

## These four features can improve convergence, stability, and training speed without changing the model architecture.

Why use a Learning Rate Scheduler?

A scheduler gradually adjusts the learning rate during training. A larger learning rate helps the model learn quickly in the beginning, while a smaller learning rate later enables more stable convergence and better final performance.

Why not always use a very small learning rate?

Training would converge extremely slowly and might get stuck before reaching a good solution.

Why not always use a very large learning rate?

The optimizer may overshoot the optimum, causing unstable training or divergence.

Why Early Stopping?

Answer
Early stopping monitors validation performance and terminates training when the model stops improving. This helps prevent overfitting and reduces unnecessary computation.

---

AMP Pipeline(Mixed Precision)
Forward

↓

Autocast

↓

Loss

↓

GradScaler

↓

Backward

↓

Optimizer

We'll implement

torch.cuda.amp.autocast()

GradScaler()

Why not always use FP16?

Because FP16 cannot accurately represent very small numbers.

Example

Gradient

0.00000000001

FP16 rounds it to

0

Training breaks.

Solution

PyTorch introduced

GradScaler

Pipeline

Forward

↓

Autocast

↓

Loss

↓

Scale Loss

↓

Backward

↓

Unscale

↓

Optimizer Step

Everything happens automatically.

Why GradScaler?

## Answer: FP16 can underflow very small gradients to zero. GradScaler scales the loss before backpropagation so gradients remain representable, then unscales them before the optimizer update.

Without AMP

Forward

↓

Backward

↓

Optimizer

With AMP

Forward FP16

↓

Loss Scaling

↓

Backward FP16

↓

Unscale

↓

Optimizer FP32

The optimizer still maintains FP32 master weights.

Does AMP reduce accuracy?

Answer: Usually no. Mixed precision maintains FP32 master weights while performing many computations in FP16, providing significant speedups with little or no loss in model accuracy.

---

Gradient Clipping

Another interview favorite.

Problem

Gradient

↓

1000

↓

Weight Update

↓

Explosion

Solution

torch.nn.utils.clip*grad_norm*()

Why Gradient Clipping?

Answer: Gradient clipping prevents excessively large gradients from causing unstable parameter updates. It is especially useful in deep networks and recurrent models where exploding gradients can occur.
-------------------------
EfficientNet: 
EfficientNet contains:

MBConv blocks
Squeeze-and-Excitation (SE)
Swish (SiLU) activation
Drop Connect
Compound Scaling

Implementing all of this manually would take several days.

A senior ML engineer is expected to understand why these components exist and how to fine-tune them—not to rewrite the architecture.

What is the biggest contribution of EfficientNet?
Answer: EfficientNet introduced compound scaling, which scales network depth, width, and input resolution together using a principled scaling method. This achieves better accuracy with fewer parameters and lower computational cost compared to scaling only one dimension.
