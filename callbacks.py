from parameters import *
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "-" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

logpath = generate_unique_logpath("logs", "train")
if not os.path.exists(logpath):
    os.mkdir(logpath)


file_writer = tf.summary.create_file_writer(logpath + "/metrics")
file_writer.set_as_default()


checkpoint_filepath = os.path.join(logpath,  "best_model.h5")

checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True, save_weights_only=True)

tensorboard_callback = TensorBoard(log_dir=logpath)

earlystop_cb = EarlyStopping(monitor='val_loss',patience=3)
callbacks = [checkpoint_cb, tensorboard_callback]