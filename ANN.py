import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

 
model = Sequential()

 
model.add(Dense(4, input_dim=2, activation='relu'))   
model.add(Dense(1, activation='sigmoid'))  
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
model.fit(X, y, epochs=1000, batch_size=4, verbose=1)

 
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

 
predictions = model.predict(X)
print('Predictions:')
for i in range(len(X)):
    print(f'Input: {X[i]}, Predicted Output: {predictions[i]}')
