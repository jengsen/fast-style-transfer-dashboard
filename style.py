import tensorflow as tf

from PIL import Image
import os
from networks import TransformerNet
from utils import load_img
import matplotlib.pyplot as plt


def stylize_image(image_path, style, style_weight):
    image = load_img(image_path)
    style_weights = {1: '10',
                     2: '100',
                     3: '1000'}
    print('stylise_image')
    transformer = TransformerNet()
    optimizer = tf.optimizers.Adam(learning_rate = 0.001)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer, step=tf.Variable(1))
    # ckpt = tf.train.Checkpoint(transformer=transformer)
    # ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()
    model_path = f'models/style/{style}_sw{style_weights[style_weight]}'
    ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
    # ckpt.restore(tf.train.latest_checkpoint('models/style/la_muse_contentlayer33_sw100')).expect_partial()
    print('ckpt')

    transformed_image = transformer(image)
    print('transformed_image')
    transformed_image = tf.cast(
        tf.squeeze(transformed_image), tf.uint8
    ).numpy()
    print('transformed_image cast')

    img = Image.fromarray(transformed_image, mode="RGB")
    output_path = f'images/output/output_{style}_sw{style_weight}.png'
    img.save(output_path)
    return output_path


if __name__ == "__main__":
    stylize_image("Pierre.jpg", "udnie", 3)