from keras.datasets import mnist
dataset = mnist.load_data()
train,test = dataset
X_train,y_train = train
X_test, y_test = test

from keras.layers import Convolution1D, MaxPooling1D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
import pandas as pd

y_train = to_categorical(y_train)

learningRate = 0.01
count = 1
acc = 0

while acc<=97.0:
    
    model = Sequential()
    model.add(Convolution1D(filters=500, kernel_size=(3), activation='relu', input_shape=(28,28)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    for i in range(count):
        model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(optimizer=RMSprop(learning_rate=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=4)


    df_history = pd.DataFrame(model.history.history)
    accuracy = df_history['accuracy']
    acc = max(accuracy)*100
    count+=1
    learningRate/=10
