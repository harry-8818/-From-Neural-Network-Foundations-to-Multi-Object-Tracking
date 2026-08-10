# Computer Vision — From Neural Network Foundations to Multi-Object Tracking

A week-by-week log of what I built and understood, starting from a plain
multilayer perceptron and ending with a working multi-object tracking pipeline.

# Week 1 →

I started with the two building blocks everything else rests on: the multilayer
perceptron and the convolutional neural network.

## MLP

I chose a layered structure so that each layer could learn features from the
training data, and then use those learned features to identify objects in the
test data and produce accurate outputs. Every layer carries its own weights and
biases to capture features, and I used activation functions like sigmoid and
ReLU (Rectified Linear Unit) to keep each neuron's activation within a useful
range.

$A_i = sigmoid(W_i*A_{i-1}+B_i)$ → The equation for the matrix.
Call $W_i*A_{i-1}+B_i$ to be $Z_i$.

### Gradient Descent and Cost Function :

This is how a neural network learns from the training data. It compares the
network's output with the correct output, measures how badly the network is
performing, and then adjusts the weights based on that error.

### Back - Propagation :

Backpropagation uses the cost function to set the weights and biases of a layer
according to the difference between the desired output and the actual output.
For each neuron I considered how a change in it affects all the neurons in the
previous layer, and then summed those contributions to arrive at better weights.
The goal is to see how a small change in the weights influences the change in
the activation of that layer.

$$
\frac{\partial C_0}{\partial W_L}=\frac{\partial Z_L}{\partial W_L}*\,\frac{\partial A_L}{\partial Z_L}*\,\frac{\partial C_0}{\partial A_L}=A_{L-1}*\frac{\partial A_L}{\partial Z_L}*2(A_L-y)
$$

$$
\frac{\partial C}{\partial W_L} = \frac{1}{n}*\sum_{k=0}^{n-1}\frac{\partial C_k}{\partial W_L}
$$

The biases follow the same idea: I calculate the partial derivatives with
respect to them to minimize the cost of the network.

## CNN - Convolutional Neural Networks

### Why we need them?

When we flatten an image, we lose the spatial connection each pixel has with
its neighbours. A CNN preserves those relationships by working over a 2-D
matrix instead.

Here the network uses a learnable filter matrix (the kernel) and computes the
relationship between the input matrix and this kernel. The kernel slides over
the input and extracts features from it.

### Hyperparameters :

Kernel Size (F) : the spatial dimensions of the filter, in other words how many
pixels the filter looks at a time.

Stride (S) : the gap between two consecutive positions of the filter. S=1 means
we move pixel by pixel, while S=2 means we skip the next pixel each step.

Padding (P) : used to preserve the dimensions of the input matrix, since the
filter otherwise contracts the spatial size. To avoid that, we append rows and
columns of zeroes at the edges of the input.

The depth of the input tensor after one full pass of the filters depends on how
many filters we use, and the total number of parameters equals the total weights
and biases. Every filter has one bias, and its weights form a tensor whose depth
matches the depth of the input tensor before the pass.

# Week 2 →

Having built an MLP and a CNN in week 1, I ran into the problems that appear as
networks get deeper: they start to collapse under their own weight. This week I
focused on diagnosing those problems and fixing them, which led me to residual
layers.

## Techniques Used →

### Data Augmentation :

I used data augmentation to avoid overfitting during training. The main
transformations are rotating, flipping, and changing colour-related parameters
such as brightness and contrast, so the network almost never sees the exact
same image twice across epochs.

### Vanishing Gradients and Degradation :

Vanishing gradients happen during backpropagation because multiplying many very
small numbers can decay the gradient toward zero, so in a very deep network the
early layers may fail to learn.

After fixing that, I noticed a second problem, degradation: a network with more
layers performed worse than one with fewer. In these networks it is hard to even
represent an identity mapping, which causes the degradation. Residual networks
solve this with skip connections, which I discuss below.

### Batch Normalization :

Batch normalization addresses problems like internal covariate shift, where
changing the weights in earlier layers abruptly affects a later layer and slows
learning. To avoid this I add a layer that normalizes the parameters before
passing them on, computing the mean and variance of each batch and standardizing
with the formula below (here $\epsilon$ is a very small number that prevents
division by zero).

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

## Residual Layers Network →

Earlier networks tried to learn the final non-linear output after L layers,
which involves many matrix multiplications and becomes dangerous in deep
networks. ResNet changes this: each block only learns the residual function
F(x), which represents the new features, and the final output is H(x) = x + F(x),
adding the original input back in without forcing the network through those
complex multiplications.

Writing out the layer equations shows why this solves degradation: a 1 gets
added to the second term of the partial derivative. I also used two residual
blocks per layer, one to update the weights and one to stabilize the data.

# Week 3 →

With deep networks now trainable, the next issue was that training from scratch
starts from randomly initialized, non-convex parameters. This week I addressed
that by reusing pre-trained models through two techniques: feature extraction
and fine-tuning.

### Feature Extraction →

Here I split the layers into two groups, early layers and deep layers. I trust
the early layers to extract basic features from new images just as they did on
their original data, so I only train the weights and biases of the deep layers,
which capture dataset-specific features. This saves a lot of GPU memory and time,
because I compute gradients only for the deep layers and not the early ones.

## Fine - Tuning →

For fine-tuning I use differential learning rates during training, so that large
gradient steps do not destroy the already-optimized weights. The learning rate
decreases as we go deeper into the network.

### How to choose between two?

The choice mainly depends on two factors: how large the dataset is, and how
similar it is to the dataset the model was pre-trained on. This gives four cases:

1. Small dataset, high similarity → Feature extraction, since there isn't enough
   data to fine-tune without overfitting, and the pre-trained weights already
   suit the task.
2. Large dataset, high similarity → Fine-tuning, since there is now enough data
   to avoid overfitting and tune the weights for maximum accuracy.
3. Small dataset, low similarity → Feature extraction, since the dataset is small
   and the early layers, which extract very different basic features, should stay
   frozen.
4. Large dataset, low similarity → The domains differ, so we retrain from the
   start, but a large dataset lets us retrain the whole network, using the
   pre-trained weights as a better initialization than random.

# Week 4 →

Until now I had only classified images. This week the goal was not just to say
what object is in an image but also where it is, which is object detection, and
then to carry that across video frames as tracking.

I used YOLO (You Only Look Once) models, which use a CNN to detect objects in
video frames. The model predicts the center coordinates of each bounding box
along with its width and height. The output tensor splits into two parts, one
that classifies the object in the frame and one that predicts the exact
coordinates and dimensions of the box. The cell containing the object's center
is responsible for detecting it. The confidence score reflects the IOU between
the predicted box and the actual box, where IOU (intersection over union) is the
area of intersection of the two boxes divided by the area of their union.

A YOLO model has three key components: backbone, neck, and head. The backbone is
made of CNN layers that detect key image features and is pre-trained on an image
classification dataset. The neck uses those features to form predictions, and
the head is the final output layer.

### NMS ( Non Maximum Suppression) :

NMS filters redundant predictions down to a single bounding box per object.

It takes the list of predicted boxes, each with a confidence score, sorts them
in descending order of confidence, and computes the IOU between the highest-
confidence box and the remaining boxes. If any remaining box has an IOU above a
threshold, it is treated as a duplicate of the same object and removed.

### Kalman Filter :

I used a Kalman filter to estimate the future state of an object from
measurements, assuming a constant-velocity model. It first uses kinematics to
predict the object's state at t+1, and when the model reports the actual bounding
box, the filter computes the difference and outputs a covariance matrix
representing the uncertainty.

### The Hungarian Algorithm :

The filter gives predicted boxes and YOLO gives the actual detected boxes, and
the Hungarian algorithm solves the bipartite matching problem, assigning new
detections to existing tracks optimally. It builds a cost matrix with at least
one zero in every row and column using row operations, then covers all the zeroes
with the minimum number of lines. Because the algorithm minimizes cost, I use the
trick below:

$1-\mathrm{IoU}(\mathrm{Track}_i,\mathrm{Detection}_j)$ — minimizing this
maximizes the IOU, which is what we want for matching.

# Week 5 →

This week I improved the tracker from Week 4, which broke whenever a pedestrian
was occluded or their detection confidence dropped for a few frames. I used two
techniques for this: ByteTrack for the association step, and interpolation as a
post-processing step.

### ByteTrack :

The basic tracker keeps only the high-confidence detections and discards the
rest, but a pedestrian who is partly hidden or blurred often falls to a low
confidence for a few frames, and dropping those boxes breaks the track. ByteTrack
fixes this by keeping every detection and separating them into two groups, one
with high confidence scores and one with low confidence scores. It first uses the
Hungarian algorithm to match the high-confidence detections with the boxes
predicted by the Kalman filter. Then it runs a second matching pass, taking the
tracks that were left unmatched and matching them against the low-confidence
detections, which recovers the objects that only dipped in confidence instead of
losing them. This second pass is what keeps a pedestrian on the same ID through a
short occlusion. In my code this is controlled by the low conf=0.10 threshold, so
the low-score boxes survive to be matched, and by the tracking thresholds I tuned
in the tracker YAML.

### Interpolation :

Even after ByteTrack keeps the ID stable, the output still has gaps: for the few
frames where the pedestrian was fully hidden there is no detection at all, so no
box is written for that ID. Interpolation is a post-processing step that fills
those gap frames. For each ID, whenever two consecutive detected frames have a
gap with 1 < gap <= max_gap, I take the box before the gap and the box after it
and fill each missing frame with a linear blend, using weight = j/gap so that
box = box1*(1-weight) + box2*weight slides the box smoothly from the first to the
second. I capped max_gap at 30 frames, because beyond that the straight-line
assumption stops being reliable and the pedestrian may have actually left the
scene.
