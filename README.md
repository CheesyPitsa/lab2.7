# Лабораторная работа 2.7. 
## Реализация GAN, генерирующей изображения одежды

На основе лекции была реализована GAN

Используемые слои:
```python
generator = tf.keras.Sequential([
  Dense(7 * 7 * 256, activation='relu', input_shape=(hidden_dim,)),
  BatchNormalization(),
  Reshape((7, 7, 256)),
  Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
  BatchNormalization(),
  Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
  BatchNormalization(),
  Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='sigmoid'),
])
```
```python
discriminator = tf.keras.Sequential()
discriminator.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(1))
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```

Результат:
* После обучения на 10 эпохах

![image](https://github.com/CheesyPitsa/lab2.7/assets/113666100/b790f714-96a1-4537-88ad-9562ceeb4e29)

* После обучения на 30 эпохах

![image](https://github.com/CheesyPitsa/lab2.7/assets/113666100/6c05e728-fd70-4050-b34a-2084c43a04b5)

* После обучения на 40 эпохах

![image](https://github.com/CheesyPitsa/lab2.7/assets/113666100/3e5df88d-0d48-449d-be2a-3858b510f427)
