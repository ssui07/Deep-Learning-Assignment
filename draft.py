model_4 = models.Sequential()

model_4.add(Lambda(normalize, input_shape = (28, 28, 1)))
model_4.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_4.add(BatchNormalization())

model_4.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_4.add(BatchNormalization())
model_4.add(layers.Dropout(0.25))

model_4.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_4.add(layers.MaxPooling2D((2, 2)))
model_4.add(layers.Dropout(0.25))


model_4.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_4.add(BatchNormalization())
model_4.add(layers.Dropout(0.25))

model_4.add(layers.Flatten())
model_4.add(layers.Dense(512, activation='relu'))
model_4.add(BatchNormalization())
model_4.add(layers.Dropout(0.5))
model_4.add(layers.Dense(128, activation='relu'
model_4.add(BatchNormalization())))
model_4.add(layers.Dropout(0.5))
model_4.add(layers.Dense(10, activation='softmax'))