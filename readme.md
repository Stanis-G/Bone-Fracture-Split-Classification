# Description
The dataset was taken from kaggle https://www.kaggle.com/datasets/shyamgupta196/bone-fracture-split-classification. It contains images of bone fractures, belonging to 12 different classes. Train and test data have 1202 and 523 images correspondingly

# Data preparation
Images that have a format not supported by keras were transformed to .png files. Images were read in grayscale mode as long as color does not matter for bone fracture classification. The validation dataset was taken from train in a ratio of 15% valid, 85% train.
The attempt to augment data with mirroring images was taken, but it did not improve model performance

# Model structure
The basic model is CNN with 3 conv layers followed by maxpool layers and 1 fully connected hidden layer. In order to improve accuracy on valid data model was undergone to various transformations, including additional fully connected hidden layer, dropout layers in various combinations, batch normalization layers. The final model structure is:
- conv1 + maxpool
- conv2 + maxpool
- conv3 + maxpool
- batch norm
- flatten
- fully connected + dropout
- fully connected

# Hyperparameters optimization and best model
The optimized parameters are learning rate, num of neurons in fully connected layer, probability for neuron to turn off during training (dropout probability). Hyperparameters search was performed on the grid with evaluation on valid data. After this step the model with best hyperparametes was fitted. Best model performance is 90.1% on train data, 41.7% on valid data, 35.9% on test data

# Considerations
1. **Data preparation**. Addtitonal data preparation may be useful for improving model performance. Many images from dataset contain extra elements (arrows, writings, frames), which better to be removed. The area of interest in many cases is small compared to image size, so zoom can be applied.
2. **Weights initialization**. Activation function in all layers except last one is ReLU, which works better with Kaiming initialization according to literature. For now model is initialized with glorot (Xavier) weights.
3. **Activation function**. ReLU activation makes links with negative weights inactive, because of zeroing gradients in negative range. Replacing ReLU with Leaky ReLU can solve that problem and potentially imporve model performance.
4. **Valid data**. After optimizing hyperparameters the best value of learning rate is 0.0014000000000000002, which came from np.arange function. Changing it to 0.0014 reduces accuracy on valid data, so the suspicion is that current valid data has local minimum. That minimum can be absent in general data distribution, so the best found value of learning rate may not be the best for inference of new data.
