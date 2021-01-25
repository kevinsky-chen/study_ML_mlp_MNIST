from MLP_class import MLP
from keras.models import load_model
from keras.datasets import mnist

# Loading dataset
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

# Preprocessing testing data
testing_input = MLP.preProcessor()
testing_input_image = testing_input.normalize(testing_input.flatten(test_feature))
testing_input_label = testing_input.one_hot_encoding(test_label)    # default labels in one-hot encoding

# Loading pre-trained model
print("載入模型 Mnist_mlp_model.h5")
model = load_model('Mnist_mlp_model.h5')

# 載入架構的training method, 並 predict
mlp_reload = MLP.model()
prediction = mlp_reload.testing_from_reload(testing_input_image, testing_input_label, model)
MLP.preProcessor.show_images_labels_predictions(test_feature, test_label, prediction, 15)

