# WeatherClassificationSystem
CNN is used as the choice of model for the image classification task. It is because they perform the best when it comes to tasks related to image classification, segmentation, etc as compared to normal Feedforward Neural Networks. This is because of their ability to not only look at individual pixel values but rather store the spatial information too whilst ensuring reduced model complexity. This is thanks to the fact that they share weights. We used the following layers in the model structure:

Conv2d layer - It applies Convolution operation to a 2d image in a sliding window-like manner where the window size equals the size of the kernel.
Maxpooling2d layer - Applied pooling layer on a 2D image which downsamples the image to a smaller size and picks the maximum value in the given window equal to pool size(n,n)
Dropout layer - The Dropout layer sets input units to 0 at random with a frequency rate at each training step, thereby preventing overfitting. Inputs that are not set to 0 are multiplied by 1/(1 - rate) so that the sum of all inputs remains unchanged[11]. 
Flatten layer- This layer converts the tensors to 1-dimension.
Dense layer - normal densely connected layers in neural networks

The final model is shown in fig 4. The model has 4 Conv2D layers kernel size (3,3) each followed by a MaxPooling layer of pool size(2,2). There are three Dropout layers connected to the last three MaxPooling layers to prevent overfitting. After 4 Conv2D layers, a Flatten layer is used which is connected to a fully connected dense layer and is connected to a Dense layer with 128 nodes. All the Conv2D layers and the dense layer have ‘relu’ as the activation function. The last layer is a dense layer with 4 nodes corresponding to the 4 output classes in our dataset. The activation used in this layer is ‘softmax’ which is the common choice for the output layer in case of multi-class classification problems.
<img width="365" alt="image" src="https://user-images.githubusercontent.com/56907651/196559432-e79ab13b-d369-4f00-b919-20b75d828597.png">
# Model Training 
 Model training refers to the process of feeding input data features and labels to the network to learn features or patterns from input data. The Convolution layers extract patterns by computing weights and generating feature maps. Then pooling is used to extract the most significant values. We have used categorical cross-entropy as the loss function and adam optimizer which is known to be much more efficient than others. The model was trained with a batch size of 32. For each epoch the model goes through the data and computes loss, it also calculates the performance on the validation set provided to check how good of a task it is doing.

Keras - Callbacks API was also used along with the training loop to add additional functionality. These offer the ability to add custom actions at multiple stages during the training, before every epoch. We have used three callbacks in our training process.

ModelCheckpoint- It is used to save the whole Keras model or just a model weight Add some regular intervals. It can be done after every epoch or a fixed number of batches as checkpoints[12]. These can later be used to evaluate the model performance on new data without the need to retrain it. We saved the best-performing model, which means that after every new update if it finds a better-performing model, it would delete the previous one and only keep the latest version. It also provides an option to monitor one of the parameters in training to decide when to save the model. One of the parameters could be model accuracy. So when it sees an improvement in accuracy, it would save the model.

EarlyStopping - It is used to stop the model during the training process if it has stopped improving its performance, which could be decreased in loss values or increase in accuracy depending upon the choice of metric used to monitor the model performance. There is also a parameter called patience, which waits for a particular number of epochs before stopping the training process if no improvements are seen.

ReduceLROnPlateau - It reduces the model's learning rate if defined metrics have stopped improving, which helps the model to learn better by reducing the learning rate when learning becomes stagnated.

# Model Evaluation                                 
                 To evaluate the model we checked the performance of the model against unseen data kept separate from the training and validation dataset. We normalized these images too and input them in the network to check how accurately it can predict those correctly. There are about 115 test images belonging to the four classes. For a given image we take prediction values for all the classes and take the one which has the highest score among the four, we then multiply the score by 100 to get the score value as a percentage. The model gives an accuracy score of 95.62% when checked on all the images in the test dataset. Fig6 shows the performance of the trained model on some random samples from each of the classes.

<img width="479" alt="image" src="https://user-images.githubusercontent.com/56907651/196559666-f9d00fc9-f831-40d5-9af0-7af38c02087c.png">

The leftmost image in the figure shows almost more than 99% accuracy which means the model trained is 99% sure it is the sunrise class which is the same as the actual class. However since this score is higher than the average value of test accuracy, this means the accuracy score might not give us a very good intuition about how well our model performs on the data and we opted for some other metrics too to get a better picture. Some of the other commonly used metrics are Precision, Recall, F1-score ( combination of precision and recall), and area under the ROC curve. We selected two other ones from the scikit-learn metrics: the confusion matrix which tells us about classification counts for classes and the classification report which combines many of the mentioned metrics together 
Confusion matrix is a matrix of size nxn where n is the number of classes. It takes true class values and predicted values to give a matrix of the count of classification done by the model. So here in our case, it will be a 4 x4 matrix. So in the confusion matrix M, Mi, and j is equal to the values known to be in group i and classified to be in group j[13]. So correctly classified values for any class would be present when i=j in that row and column combination. The figure below shows us the values predicted for each class by the baseline model built. We can see from Fig7 that  3 images from the cloud class are classified as shine and 1 from the shine class is misclassified as sunrise. From the matrix, it can be seen that the sunrise and rain classes have a perfect score for classification. This shows that classes that had relatively more images performed much better than others even after applying data augmentation.


<img width="481" alt="image" src="https://user-images.githubusercontent.com/56907651/196559789-c977a474-797e-40cc-8ffb-41a087f8f133.png">

Classification report also takes the true labels for data and the predicted ones by the model to give a summary of various helpful metrics available[14]. The metrics present in it are accuracy, precision, f1-score, and support. One of the best features of these metrics is that it shows values of these metrics for individual classes which helps us get a better idea of how our model is performing. One of the metrics that work well for classification is the F1 score.

<img width="485" alt="image" src="https://user-images.githubusercontent.com/56907651/196559870-6c4126e2-e312-4530-b565-e294ae5ff7f8.png">

Precision and recall contribute equally to the F1 score, and the harmonic mean can be used to determine the optimal trade-off between the two variables. 

<img width="619" alt="image" src="https://user-images.githubusercontent.com/56907651/196560033-25e05c50-061a-4164-8197-8516d433d154.png">

Fig 8: Classification report for the image classification task
# Experiments to improve model performance
To achieve the best model performance we carried out various tests. These included changing the values of hyperparameters and adding additional layers to handle some of the problems. Some of these were done on the baseline model while others were part of building it. The parameters for the baseline model are discussed in previous sections, they are as follows:

Data Augmentation - Yes ( scaling, rotation, etc)
Layers -  4 layers Convo2d, 4 MaxPooling layers,1 Flatten, 2 dense, and 3 dropout layers.
Activation Function - Relu and softmax at output layer
Optimizer- Adam
Loss function- Categorical_crossentropy
Batch Size - 32
Epochs - 500
EalryStopping: monitor loss variable

<img width="542" alt="image" src="https://user-images.githubusercontent.com/56907651/196560469-781c3c27-5628-40b4-9fab-f54f73a624ed.png">


Result
The following table lists changes made to the baseline model to check for changes in results obtained after training is done. The second column only lists the differences from the baseline model. For example, optimizer =RMSprop means optimizer was changed from adam to rmsprop while keeping other things constant in the model architecture. 

<img width="688" alt="image" src="https://user-images.githubusercontent.com/56907651/196560526-1fcdb6a1-ad85-4e9e-85e9-a2b2f5508648.png">

From the table it can be seen that changing the value of learning rate from default (lr=0.001) didn’t help us in getting  better accuracy and in fact as can be seen from fig 14.In experiment no. 2, when adam lr=0.01, the signs of local optima have been seen. In this instance, the test accuracy had frozen to 0.317 and was not rising because the model was trapped in local optima, leading to underfitting. This occurred as a result of a very high learning rate, therefore in order to address it, we decreased the learning rate, which allowed us to remove it from the local optima and produce better outcomes..However, changing the optimizer from adam to ‘rmsprop’ gave better loss value but almost same accuracy value but ‘sgd’ ( Stochastic gradient descent) didn’t give better result than baseline with default and even when using a learning rate scheduler.After changing the Early Stopping criteria in exp 5 and 6 from loss minimization to validation accuracy maximization we gain saw better result in exp 6 , this time both in accuracy and loss value.While building the model we also trained the model without using any data augmentation techniques and in this case performance was not satisfactory as val_loss could not converge after a point while training loss improved.

<img width="554" alt="image" src="https://user-images.githubusercontent.com/56907651/196560611-3c2706c1-20bb-48ea-b86f-9a13d2f18b05.png">

<img width="497" alt="image" src="https://user-images.githubusercontent.com/56907651/196560648-c6210cca-e1e8-40ea-b77a-190a74fd938e.png">

We also tried a different activation function , Leaky_RELU which is a modified version of relu offering gradient values for negative derivative values and preventing the issue of dead neurons.Trying a high and low values for alpha parameter , we observed the model performance was almost the same.The best performance was observed in case of exp 12 when we used a batch_size of 8 where accuracy jumped to ~99% as shown in figure 16(a) .This case  had just one misclassified example which was a shine class image but was misclassified as a sunrise. These cases are very similar in the real world too so the model might not have been able to differentiate them. The final experiment was removing the dropout layers which was done while building the model. The results shown in the figure 17 justify their use as we can see that without dropout the model loss stopped converging and was a case of underfitting too as training and validation loss were not close and validation loss also started to climb after 100 epochs.

<img width="503" alt="image" src="https://user-images.githubusercontent.com/56907651/196560701-f47b30ce-7371-4bf3-aeee-f24c08b273ac.png">

<img width="500" alt="image" src="https://user-images.githubusercontent.com/56907651/196560740-aa9496e1-0211-4f54-9dee-c63993aabf97.png">

<img width="503" alt="image" src="https://user-images.githubusercontent.com/56907651/196560701-f47b30ce-7371-4bf3-aeee-f24c08b273ac.png">


