# FACIAL-LANDMARK-DETECTION

## OBJECTIVE
The goal of the project is to develop a highly efficient model capable of accurately detecting facial key points in input images. This model will support real-time inference for applications such as face recognition, emotion detection, virtual makeup, and augmented reality (AR).

## LEARNING PROCESS


1. The project began with an understanding of Deep Learning, Artificial Neural Networks, and Numpy, leading to the creation of a basic MNIST model.
2. Next, we studied various loss functions like cross-entropy loss and optimizers like Adam, then learned PyTorch and built an MNIST model using this framework.
3. This was followed by exploring Convolutional Neural Networks (CNNs) and studying various CNN architectures, culminating in the implementation of a CIFAR-10 model.
4. We then created a custom dataset for facial landmark detection.
5. Finally, we learned and implemented the facial landmark detection algorithm using the PyTorch framework.



### CUSTOM-DATASET
The dataset is sourced from the iBUG 300-W dataset with XML-based annotations for facial landmarks and crop coordinates. It extracts image paths, 68 landmark points, and face cropping coordinates. The preprocessing pipeline resizes images with padding, applies random augmentations such as color jitter, offset and random cropping, and random rotation, adjusting landmarks accordingly. Landmarks are normalized relative to image dimensions, and images are converted to grayscale before being transformed into tensors normalized between [-1, 1]. A dataset class handles parsing, preprocessing, and returning ready-to-use images and normalized landmarks.

#### RESULTS
Output-
![image](https://hackmd.io/_uploads/S1GGjmYkkg.png)

### FACIAL LANDMARK DETECTION

For facial landmark detection, datasets such as the *iBUG 300-W* and similar facial landmark datasets are commonly used. These datasets contain a variety of images annotated with facial key points. For example, the *iBUG 300-W* dataset consists of thousands of images, with each image labeled with 68 facial landmarks, including points around the eyes, nose, mouth, and jawline. The dataset is typically divided into training and test sets, enabling model training and evaluation for tasks like face recognition, emotion analysis, and augmented reality applications.

#### PROCEDURE




In this project, a **FaceLandmarksAugmentation** class was implemented to apply various image augmentation techniques, such as cropping, random cropping, random rotation, and color jittering, specifically tailored for facial landmark detection. ![image](https://hackmd.io/_uploads/HkrIMVtk1l.png)
This class initializes key parameters like image dimensions, brightness, and rotation limits. Methods such as `offset_crop` and `random_rotation` adjust landmark coordinates accordingly. A **Preprocessor** class initializes the augmentation methods and normalizes the data, while a **datasetlandmark** class, inheriting from `Dataset`, handles image paths, landmark coordinates, and cropping information parsed from XML files. The dataset is split into training and validation sets, with `DataLoader` objects created for each, and a function is defined to visualize images with corresponding landmarks.![image](https://hackmd.io/_uploads/HJco7EYy1e.png)
The **network design** is a modular convolutional neural network (CNN) with depthwise separable convolutions to improve efficiency. It includes an entry block for initial feature extraction, middle blocks with residual connections, and an exit block that outputs facial landmark coordinates. The CNN uses batch normalization and LeakyReLU for better performance. A training loop is implemented to run for 30 epochs, where the model computes loss, updates weights using an optimizer, validates after each epoch, and saves checkpoints to prevent overfitting and retain the best-performing model.




The project employs **efficient convolutions** using depthwise separable convolutions to reduce computational complexity and enhance efficiency. Batch normalization and LeakyReLU activation functions are used throughout the architecture to improve model performance and ensure better convergence during training. A training loop was implemented to run for 30 epochs, during which the model computes the training loss, performs backpropagation, and updates the weights using an optimizer. After each epoch, the model is validated, and checkpoints of the model's state, including weights and optimizer settings, are saved. This process helps facilitate progress recovery and prevent overfitting by retaining the best-performing model based on validation metrics.
![image](https://hackmd.io/_uploads/BJo3HEYJkl.png)
#### HYPERPARAMETERS
| Hyperparameters | Value | 
| -------- | -------- | 
|   Batch-Size| 32
Learning-rate	|0.00075
num of Epochs|30
Loss |      MSE LOSS
Optimizer |    Adam Optimizer
#### RESULTS
LOSS-
 
Loss at epoch 1:
![image](https://hackmd.io/_uploads/S1kkuNKJ1l.png)
Loss at epoch 2:
![image](https://hackmd.io/_uploads/H1C-OEKykg.png)
Loss at epoch 3:
![image](https://hackmd.io/_uploads/HJfr_4Kyyx.png)


GRAPHS -
![image](https://hackmd.io/_uploads/S1qu_VYkyg.png)


### SOFTWARE TOOLS USED 
 Python
Numpy
PyTorch
PIL
Matplotlib








































