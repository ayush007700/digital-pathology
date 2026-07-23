Why use a Trainer class?

Answer:

A Trainer encapsulates the complete training workflow, including optimization, validation, checkpointing, and logging. This separates training logic from model architecture, making the code modular and reusable.

Why separate optimizer from Trainer?

Answer:

Different experiments may require different optimizers such as Adam, SGD, or AdamW. Using a factory keeps the Trainer independent of optimization strategy.

Why separate loss functions?

Answer:

Different tasks require different loss functions. Classification uses CrossEntropy, segmentation often uses Dice Loss, and imbalanced datasets may benefit from Focal Loss. Decoupling losses makes experimentation easier.
