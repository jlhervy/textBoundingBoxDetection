
from absl import flags, app
import sys
from .callbacks import *
from .parameters import *
from .models.metrics import build_iou
from .models.loss import build_loss
from .models.model import psenet_model
from .data.tf_data_generator import get_dataset

flags.DEFINE_integer('height', 1000, '')
flags.DEFINE_integer('width', 1000, '')
flags.DEFINE_integer('batch_size', 4, '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('epochs', 60, '')
flags.DEFINE_float('moving_average_decay', 0.997, '')
flags.DEFINE_string('checkpoint_path', 'logs/train-2/best_model.h5', '')
flags.DEFINE_boolean('load_from_checkpoint', False, 'Whether to load model from checkpoint file')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def main(argv=None):
    # tf.config.experimental_run_functions_eagerly(True)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        input = tf.keras.layers.Input(shape=(FLAGS.height, FLAGS.width, 3))
        outputs = psenet_model(input, is_training=True)
        model = tf.keras.models.Model(inputs = input, outputs = outputs)
        if FLAGS.load_from_checkpoint :
            model.load_weights(FLAGS.checkpoint_path)
        model.summary()
        optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
        ious = build_iou([0, 1], ['bk', 'txt'])
        model.compile(optimizer = optimizer, loss = build_loss, metrics=ious)
        train_dataset, val_dataset = get_dataset(FLAGS.batch_size)

        STEPS_PER_EPOCH = np.ceil(train_size / FLAGS.batch_size)
        VALIDATION_STEPS = np.ceil(val_size / FLAGS.batch_size)
        model.fit(train_dataset, epochs=FLAGS.epochs, steps_per_epoch=STEPS_PER_EPOCH, validation_data=val_dataset, validation_steps=VALIDATION_STEPS, callbacks=callbacks)


if __name__ == '__main__':
    app.run(main)
