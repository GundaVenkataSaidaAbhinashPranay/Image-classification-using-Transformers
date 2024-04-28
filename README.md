# Image-classification-using-Transfer-Learning
This project aims to perform image classification using transfer learning, a technique in which a pre-trained deep learning model is fine-tuned on a new dataset. By leveraging the knowledge gained from training on a large dataset, transfer learning enables us to achieve good performance even with limited labeled data.

###Requirements
Python 3.x
TensorFlow 2.x (or TensorFlow 1.x if using an older pre-trained model)
Keras
NumPy
Matplotlib (for visualization)
GPU (recommended for faster training)


Data Preparation: Organize your image data into directories based on class labels. The directory structure should look like this:
Copy code
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
Model Selection: Choose a pre-trained model as the base architecture. Popular choices include VGG, ResNet, Inception, and MobileNet. You can either use a model provided by Keras or import one from TensorFlow Hub. We used Vision Trasnformer
Fine-tuning: Fine-tune the pre-trained model on your dataset. This involves freezing some layers (usually the early layers) to prevent them from being updated and adding custom fully connected layers on top of the pre-trained base.
Training: Train the model on your dataset. You can adjust hyperparameters such as learning rate, batch size, and number of epochs to achieve optimal performance.
Evaluation: Evaluate the trained model on a separate validation set to assess its performance. Calculate metrics such as accuracy, precision, recall, and F1-score.
Inference: Use the trained model to classify new images. Load the model weights and pass images through the model to get predictions.
