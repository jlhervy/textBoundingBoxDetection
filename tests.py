from tensorflow.keras.applications import ResNet50V2
import tensorflow.keras.layers as ly
import tensorflow as tf
import cv2
# model = ResNet50V2(weights="imagenet", include_top=False,
#                        input_tensor=ly.Input(shape=(512, 512, 3)))
# x = model.output
#
# x = ly.Flatten()(x)
# x = ly.Dropout(0.5)(x)
# outputs = ly.Dense(3, activation="softmax")(x)
#
# model = tf.keras.Model(inputs = model.input, outputs = outputs)
#
# model.summary()

img = cv2.imread('data/dataset_gen/X51005361883/mask_3.png')
print(0)