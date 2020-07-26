from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense

dataset=mnist.load_data('mymnist.db')
train , test = dataset

X_train , y_train = train
X_test, y_test = test

img1 = X_train[0]
img1_label=y_train[0]
img_label= y_train[5]
img1d=img1.reshape(28*28)

X_train_1d = X_train.reshape(-1 , 28*28)
X_train = X_train_1d.astype('float32')
y_train_object=to_categorical(y_train)

model =Sequential()

model.add(Dense(units=512, input_dim = 28*28, activation= 'relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128,activation='relu'))    
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer= RMSprop(),loss='categorical_crossentropy', metrics=['accuracy'])

awesome = model.fit(X_train, y_train_object,epochs=4)

X_test_1d=X_test.reshape(-1, 28*28)
X_test= X_train_1d.astype('float32')

y_test_object=to_categorical(y_test)

model.predict(X_test)

acc = awesome.history['accuracy']
d = str(acc)
with open('accuracy.txt', 'w') as f:
    f.write(str(d))

model.save('mnist_model.h5')

