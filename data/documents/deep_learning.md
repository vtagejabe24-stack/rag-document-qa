# Deep Learning

Deep learning uses neural networks with many layers to learn increasingly abstract representations of data. It has driven recent progress in computer vision, speech recognition and natural language processing.

## Neural Networks

A neural network is built from layers of artificial neurons. Each neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a nonlinear activation function such as ReLU, sigmoid or tanh. Without nonlinear activations, a stack of layers would collapse into a single linear transformation and could not model complex patterns.

## Backpropagation and Gradient Descent

Backpropagation is the algorithm used to train neural networks. It first performs a forward pass to compute the predictions and the loss, then applies the chain rule of calculus to compute the gradient of the loss with respect to every weight, moving backward from the output layer to the input layer. Gradient descent then updates each weight by subtracting the gradient multiplied by the learning rate. Stochastic gradient descent estimates the gradient from small random mini-batches, which is faster and noisier than using the full dataset. Adam is a popular optimizer that adapts the learning rate for each parameter.

## Dropout and Batch Normalization

Dropout is a regularization technique that randomly sets a fraction of neuron outputs to zero during each training step. This prevents neurons from co-adapting and reduces overfitting. At test time dropout is turned off. Batch normalization normalizes the inputs of a layer across the mini-batch, which stabilizes and speeds up training.

## Convolutional Neural Networks

A convolutional neural network (CNN) is designed for grid-like data such as images. Convolutional layers slide small learned filters across the image to detect local patterns such as edges and textures. Pooling layers downsample the feature maps, which reduces computation and gives some translation invariance. Early layers detect simple features, and deeper layers combine them into complex shapes and objects. CNNs share weights across positions, so they need far fewer parameters than fully connected networks.

## Recurrent Neural Networks

A recurrent neural network (RNN) is designed for sequential data such as text, speech and time series. It keeps a hidden state that is updated at every time step, so information from earlier inputs can influence later outputs. Plain RNNs struggle with long sequences because of vanishing gradients. LSTM and GRU networks add gates that control what information is kept or forgotten, which lets them learn longer dependencies. RNNs process tokens one at a time, which makes them hard to parallelize.

## CNN versus RNN

CNNs exploit spatial structure and process the whole input in parallel, which makes them the standard choice for images. RNNs exploit temporal order and process input step by step, which makes them suited to sequences. Transformers have largely replaced RNNs for language tasks.
