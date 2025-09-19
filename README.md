# EfficientMnist-Under20K
Lightweight MNIST classifier designed for **maximum accuracy with minimal parameters**

This project is extension to [MNIST classification](https://github.com/divya-r-kamat/MnistNN/edit/main/README.md) with strict efficiency + accuracy goals ( <20K params, 99.4%+, in ≤20 epochs)

|Notebook | Model Parameters | Accuracy (Epoch20) | Description|
|---------|-----------------|--------------------|------------|
|[EfficientMnistNN_Iter1.ipynb](https://github.com/divya-r-kamat/EfficientMnist-Under20K/blob/main/EfficientMnistNN_Iter1.ipynb) | 24,498 | 99.39% | We start with [initial-mnist-nn-iteration9.ipynb](https://github.com/divya-r-kamat/MnistNN/blob/main/initial-mnist-nn-iteration9.ipynb) where we had less tham 25K parameters, but increased the epoch to 20k, achieved 99.44 at 17th epoch, but yeah parameters are >20k|
|[EfficientMnistNN_Iter2.ipynb](https://github.com/divya-r-kamat/EfficientMnist-Under20K/blob/main/EfficientMnistNN_Iter2.ipynb)|15,658|99.29%|Reduced number of channels to bring parameters down to ~15K. Accuracy dipped slightly below target, highlighting trade-offs between compactness and accuracy|
|[EfficientMnistNN_Iter3.ipynb](https://github.com/divya-r-kamat/EfficientMnist-Under20K/blob/main/EfficientMnistNN_Iter3.ipynb)|15,658|99.44%|Same as Iter2, but introduced a **1x1 convolution before max pooling**. This architectural tweak improved feature compression and helped cross the **99.4% target**.!|
|[EfficientMnistNN_Iter4.ipynb](https://github.com/divya-r-kamat/EfficientMnist-Under20K/blob/main/EfficientMnistNN_Iter4.ipynb)|15,794|99.59%|Added **Batch Normalization** after every convolutional layer. Slightly increased parameter count, but significantly stabilized training. Accuracy consistently reached **99.5%+ from epoch 16 onward**|
|[EfficientMnistNN_Iter5.ipynb](https://github.com/divya-r-kamat/EfficientMnist-Under20K/blob/main/EfficientMnistNN_Iter5.ipynb)|9,074|99.41%|Replaced the fully connected layers with a **Global Average Pooling (GAP) layer**, cutting parameters to under **10K**. Achieved excellent balance of compactness and accuracy.|
|[EfficientMnistNN_Iter6.ipynb](https://github.com/divya-r-kamat/EfficientMnist-Under20K/blob/main/EfficientMnistNN_Iter6.ipynb)|13,562|99.52%|Increased channel sizes moderately for richer representations while staying under 20K parameters. This yielded **stable 99.5% accuracy from epoch 16 onward**.!|


Through iterative refinement, the project demonstrates how careful architecture design (channel scaling, 1x1 convolutions, GAP layers, and BatchNorm) can achieve **state-of-the-art MNIST accuracy** while keeping the parameter count well below 20K.




# Key Learnings
- Strategic **1x1 convolutions** before pooling layers enhance feature compression and boost accuracy.  
- Combining **max pooling** with 1x1 conv achieves a strong balance of compactness and robustness.
- Careful use of **BatchNorm** stabilizes accuracy and speeds up convergence.  
- **GAP layers** are highly parameter-efficient replacements for fully connected layers.  
  
