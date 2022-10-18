# WeatherClassificationSystem
CNN is used as the choice of model for the image classification task. It is because they perform the best when it comes to tasks related to image classification, segmentation, etc as compared to normal Feedforward Neural Networks. This is because of their ability to not only look at individual pixel values but rather store the spatial information too whilst ensuring reduced model complexity. This is thanks to the fact that they share weights. We used the following layers in the model structure:

Conv2d layer - It applies Convolution operation to a 2d image in a sliding window-like manner where the window size equals the size of the kernel.
Maxpooling2d layer - Applied pooling layer on a 2D image which downsamples the image to a smaller size and picks the maximum value in the given window equal to pool size(n,n)
Dropout layer - The Dropout layer sets input units to 0 at random with a frequency rate at each training step, thereby preventing overfitting. Inputs that are not set to 0 are multiplied by 1/(1 - rate) so that the sum of all inputs remains unchanged[11]. 
Flatten layer- This layer converts the tensors to 1-dimension.
Dense layer - normal densely connected layers in neural networks

The final model is shown in fig 4. The model has 4 Conv2D layers kernel size (3,3) each followed by a MaxPooling layer of pool size(2,2). There are three Dropout layers connected to the last three MaxPooling layers to prevent overfitting. After 4 Conv2D layers, a Flatten layer is used which is connected to a fully connected dense layer and is connected to a Dense layer with 128 nodes. All the Conv2D layers and the dense layer have ‘relu’ as the activation function. The last layer is a dense layer with 4 nodes corresponding to the 4 output classes in our dataset. The activation used in this layer is ‘softmax’ which is the common choice for the output layer in case of multi-class classification problems.
