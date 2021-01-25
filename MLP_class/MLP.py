from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt



class preProcessor:
    def __init__(self):
        pass

    # Show image
    def show_image(self, image):
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.imshow(image, cmap='binary')
        plt.show()

    # Show images and their labels
    @staticmethod
    def show_images_labels_predictions(images, labels, predictions, start_id, num=10):
        plt.gcf().set_size_inches(12, 14)
        if num > 25: num = 25
        for i in range(num):
            ax = plt.subplot(5, 5, i + 1)
            ax.imshow(images[start_id + i], cmap="binary")
            if len(predictions) > 0:
                title = 'ai = ' + str(predictions[start_id+i])
                title += (' (o)' if predictions[start_id+i] == labels[start_id + i] else ' (x)')
                title += '\nlabel =' + str(labels[start_id + i])
            else:
                title = 'label =' + str(labels[start_id + i])

            ax.set_title(title, fontsize=12)
            ax.set_xticks([]), ax.set_yticks([])
        plt.show()

    def flatten(self, feature):
        feature_vector = feature.reshape(len(feature), 784).astype('float32')    # 28*28 -> 784
        return feature_vector    # return 1d array

    def normalize(self, feature_vector):
        return feature_vector/255       # return probability between 0~1

    def one_hot_encoding(self, label):
        return np_utils.to_categorical(label)    # return one-hot encoding in a list



class model:    # 步驟包含 架構模型(layers)->訓練(training)->測試(testing)
    def __init__(self):
        self.model = Sequential()

    def layers(self, numbers):     # numbers用list裝著每層神經元的數目(eg. 輸入層, 隱藏層*2, 輸出層)
        for i in range(len(numbers)-1):
            self.model.add(Dense(units=numbers[i+1], input_dim=numbers[i], kernel_initializer='normal', activation='relu'))
            # input or hidden layer[units本層神經元數; input_dim上一層神經元數]
        self.model.add(Dense(units=numbers[-1], kernel_initializer='normal', activation='softmax'))
        # output layer[units輸出層神經元數; input_dim上一層神經元數]

    def training(self, x, y):    # x為data, y為labels
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # loss function為categorical_crossentropy, 優化器為adam, 評估方法為accuracy
        train_history = self.model.fit(x, y, validation_split=0.2, epochs=10, batch_size=200, verbose=2)
        # x,y為設定訓練值特徵值和標籤, verbose是否顯示訓練過程(0不顯示, 1詳盡顯示, 2簡易顯示)

        # summarize history for accuracy
        plt.plot(train_history.history['accuracy'])
        plt.plot(train_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(train_history.history['loss'])
        plt.plot(train_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def testing(self, x, y):
        scores = self.model.evaluate(x, y)
        print(f"Accuracy = {scores[1]}, Loss = {scores[0]}")
        prediction = np.argmax(self.model.predict(x), axis=-1)
        #prediction = self.model.predict_classes(x)
        return prediction
    def save(self, file_name):
        self.model.save(file_name)

    def testing_from_reload(self, x, y, model):    # 若從外部載入訓練好的模型作為預測，則用此函數測試(因為訓練, 建模型都不用重複做了)
        #scores = model.evaluate(x, y)
        #print(f"Accuracy = {scores[1]}, Loss = {scores[0]}")
        prediction = np.argmax(model.predict(x), axis=-1)
        # prediction = self.model.predict_classes(x)
        return prediction
