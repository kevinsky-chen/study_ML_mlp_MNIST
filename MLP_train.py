from MLP_class import MLP
from keras.datasets import mnist

# Loading dataset
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()
print(len(train_feature), len(train_label))
print(train_feature.shape, train_label.shape)

# 資料前處理
# training data
training_input = MLP.preProcessor()
training_input_image = training_input.normalize(training_input.flatten(train_feature))
training_input_label = training_input.one_hot_encoding(train_label)    # default labels in one-hot encoding

# testing data
testing_input = MLP.preProcessor()
testing_input_image = testing_input.normalize(testing_input.flatten(test_feature))
testing_input_label = testing_input.one_hot_encoding(test_label)    # default labels in one-hot encoding

# 模型建構
mlp = MLP.model()
mlp.layers([784, 256, 128, 10])    # 設定神經層每層神經元數
mlp.training(training_input_image, training_input_label)

# 建立好的模型進行預測
prediction = mlp.testing(testing_input_image, testing_input_label)
MLP.preProcessor.show_images_labels_predictions(test_feature, test_label, prediction, 0, 15)   # show出前15個testing結果與label的比對狀況

# Save model
mlp.save("Mnist_mlp_model.h5")
print("Mnist_mlp_model.h5 模型已儲存")
del mlp


