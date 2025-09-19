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


Through iterative refinement, it demonstrates how careful architecture design (channel scaling, 1x1 convolutions, GAP layers, and BatchNorm) can achieve **state-of-the-art MNIST accuracy** while keeping the parameter count well below 20K.

## Model

    Net(
      (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
      (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(16, 28, kernel_size=(3, 3), stride=(1, 1))
      (bn3): BatchNorm2d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv4): Conv2d(28, 8, kernel_size=(1, 1), stride=(1, 1))
      (bn4): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv5): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
      (bn5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv6): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
      (bn6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv7): Conv2d(16, 28, kernel_size=(3, 3), stride=(1, 1))
      (gap): AdaptiveAvgPool2d(output_size=1)
      (fc): Linear(in_features=28, out_features=10, bias=True)
    )

## Model Parameters:

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 26, 26]              80
           BatchNorm2d-2            [-1, 8, 26, 26]              16
                Conv2d-3           [-1, 16, 24, 24]           1,168
           BatchNorm2d-4           [-1, 16, 24, 24]              32
                Conv2d-5           [-1, 28, 22, 22]           4,060
           BatchNorm2d-6           [-1, 28, 22, 22]              56
                Conv2d-7            [-1, 8, 22, 22]             232
           BatchNorm2d-8            [-1, 8, 22, 22]              16
                Conv2d-9             [-1, 16, 9, 9]           1,168
          BatchNorm2d-10             [-1, 16, 9, 9]              32
               Conv2d-11             [-1, 16, 7, 7]           2,320
          BatchNorm2d-12             [-1, 16, 7, 7]              32
               Conv2d-13             [-1, 28, 5, 5]           4,060
    AdaptiveAvgPool2d-14             [-1, 28, 1, 1]               0
               Linear-15                   [-1, 10]             290
    ================================================================
    Total params: 13,562
    Trainable params: 13,562
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.53
    Params size (MB): 0.05
    Estimated Total Size (MB): 0.58
    ----------------------------------------------------------------

## Training Output for 1st Epoch:  

    Epoch 16
    Train loss=0.0163 batch_id=468 Accuracy=99.15: 100%|██████████| 469/469 [00:23<00:00, 19.96it/s]
    Test set: Average loss: 0.0155, Accuracy: 9952/10000 (99.52%)
    
    Epoch 17
    Train loss=0.0073 batch_id=468 Accuracy=99.20: 100%|██████████| 469/469 [00:23<00:00, 19.96it/s]
    Test set: Average loss: 0.0153, Accuracy: 9952/10000 (99.52%)
    
    Epoch 18
    Train loss=0.0911 batch_id=468 Accuracy=99.21: 100%|██████████| 469/469 [00:23<00:00, 19.85it/s]
    Test set: Average loss: 0.0151, Accuracy: 9950/10000 (99.50%)
    
    Epoch 19
    Train loss=0.0104 batch_id=468 Accuracy=99.25: 100%|██████████| 469/469 [00:23<00:00, 19.98it/s]
    Test set: Average loss: 0.0146, Accuracy: 9953/10000 (99.53%)
    
    Epoch 20
    Train loss=0.0068 batch_id=468 Accuracy=99.25: 100%|██████████| 469/469 [00:23<00:00, 19.99it/s]
    Test set: Average loss: 0.0149, Accuracy: 9950/10000 (99.50%)

## Learnings
- Strategic **1x1 convolutions** before pooling layers enhance feature compression and boost accuracy.  
- Combining **max pooling** with 1x1 conv achieves a strong balance of compactness and robustness.
- Careful use of **BatchNorm** stabilizes accuracy and speeds up convergence.  
- **GAP layers** are highly parameter-efficient replacements for fully connected layers.  
  
