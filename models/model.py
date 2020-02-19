#-*- coding:utf-8 -*-
import tensorflow as tf
from absl import flags
# from tensorflow.keras.applications import ResNet50
from .custom_Resnet50 import ResNet50
import tensorflow.keras.layers as ly
from ..parameters import *



flags.DEFINE_integer('text_scale', 512, '')
FLAGS = flags.FLAGS


def unpool(inputs, rate):
    return tf.image.resize(inputs, size=[tf.shape(input=inputs)[1]*rate,  tf.shape(input=inputs)[2]*rate], method=tf.image.ResizeMethod.BILINEAR)


class resize_image(tf.keras.layers.Layer):

    def __init__(self, target_tensor_shape, target_int_shape, *args, **kwargs):
        self.target_tensor_shape = target_tensor_shape
        self.target_int_shape = target_int_shape
        super(resize_image, self).__init__(*args, **kwargs)


    def build(self, input_shape):
        super(resize_image, self).build(input_shape)

    def call(self, input_tensor, **kwargs):
        return tf.image.resize(input_tensor, (self.target_tensor_shape[0], self.target_tensor_shape[1]),
                                      method=tf.image.ResizeMethod.BILINEAR)

    # def compute_output_shape(self, input_shape):
    #     # return (input_shape[0],) + (self.target_int_shape[0], self.target_int_shape[1]) + (input_shape[-1],)
    #     return (0, 10, 10, 10)


def conv_bn_relu(input_tensor, filters, kernel_size=3, bn=True,
                 relu=True, isTraining=True, weight_decay=1e-6):
    '''
    conv2d + batch normalization + relu
    '''
    x = ly.Conv2D(filters, kernel_size, strides=(1, 1),
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input_tensor)
    if (bn):
        x = ly.BatchNormalization(axis=-1)(x)

    if (relu):
        x = ly.Activation('relu')(x)
    return x


def upsample_conv(input_tensor, concat_tensor, filters, type='resize', kernel_size=3):
    t_tensor_shape = concat_tensor.shape[1:3]
    t_int_shape = tf.keras.backend.int_shape(concat_tensor)[1:3]

    if (type == 'resize'):
        output_image = resize_image(t_tensor_shape, t_int_shape)(input_tensor)
    else:
        raise ValueError('upsample_conv type not in [resize,...]')

    output_image = ly.Concatenate(axis=3)([output_image, concat_tensor])


    output_image = conv_bn_relu(output_image, filters)

    return output_image


def build_feature_pyramid(blocks):

    for i, l in enumerate(blocks):
        if (l.shape.as_list()[-1] > max_depth):
            blocks[i] = ly.Conv2D(max_depth, kernel_size=1, padding='same')(l)
    PN = []
    output_tensor = blocks[0]
    PN.append(output_tensor)
    for i in range(1, len(blocks)):
        output_tensor = upsample_conv(output_tensor, blocks[i], upsample_filters[i - 1])
        PN.append(output_tensor)

    return PN


def FC_SN(PN):
    # Fusion of the four feature maps to get feature map F with 1024 channels via the function
    # C(·) as: F = C(P2, P3, P4, P5) = P2 || Up×2(P3) || Up×4(P4) || Up×8 (P5), where “k” refers to the concatenation and Up×2
    # (·), Up×4 (·), Up×8 (·) refer to 2, 4, 8 times upsampling
    P2 = PN[-1]
    t_tensor_shape = P2.shape[1:3]
    t_int_shape = tf.keras.backend.int_shape(P2)[1:3]

    for i in range(len(PN)-1):
        PN[i] = resize_image(t_tensor_shape,t_int_shape)(PN[i])

    F = ly.Concatenate(-1)(PN)

    #F is fed into Conv(3, 3)-BN-ReLU layers and is reduced to 256 channels.
    F = conv_bn_relu(F,256)

    #Next, it passes through multiple Conv(1, 1)-Up-Sigmoid layers and produces n segmentation results
    #S1, S2, ..., Sn.
    SN = ly.Conv2D(6,(1,1))(F)

    scale = 1
    if(ns == 2):
        scale = 1

    new_shape = t_tensor_shape
    new_shape *= tf.constant(np.array([scale, scale], dtype='int32'))
    if t_int_shape[0] is None:
        new_height = None
    else:
        new_height = t_int_shape[0] * scale
    if t_int_shape[1] is None:
        new_width = None
    else:
        new_width = t_int_shape[1] * scale

    SN = resize_image(new_shape,(new_height,new_width))(SN)
    SN = ly.Activation('sigmoid')(SN)

    return SN

def psenet_model(input):
    resnet_model = ResNet50(weights='imagenet', include_top=False,
                       input_tensor=input)
    b5 = resnet_model.get_layer("conv5_block3_out").output
    b4 = resnet_model.get_layer("conv4_block5_out").output
    b3 = resnet_model.get_layer("conv3_block3_out").output
    b2 = resnet_model.get_layer("conv2_block2_out").output
    b1 = resnet_model.get_layer("conv1_relu").output
    end_points = [b5, b4, b3, b2, b1]
    PN = build_feature_pyramid(end_points)
    seg_S_pred = FC_SN(PN)

    return seg_S_pred


# input = tf.keras.layers.Input(shape=(1000, 1000, 3))
# outputs = psenet_model(input, is_training=True)
# model = tf.keras.models.Model(inputs = input, outputs = outputs)
# model.summary()
