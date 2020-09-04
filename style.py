import tensorflow as tf

from PIL import Image
import os
from networks import TransformerNet
from utils import load_img
import matplotlib.pyplot as plt

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
    

def load_img2(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


def stylize_image(image_path, style, style_weight):
    image = load_img(image_path)
    style_weights = {1: '10',
                     2: '100',
                     3: '1000'}
    print('stylise_image')
    transformer = TransformerNet()
    ckpt = tf.train.Checkpoint(transformer=transformer)
    # ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()
    # model_path = f'models/style/{style}_sw{style_weights[style_weight]}'
    # ckpt.restore(tf.train.latest_checkpoint(model_path)).expect_partial()
    ckpt.restore(tf.train.latest_checkpoint('models/style/udnie_celeba_sw100')).expect_partial()
    print('ckpt')

    transformed_image = transformer(image)
    print('transformed_image')
    transformed_image = tf.cast(
        tf.squeeze(transformed_image), tf.uint8
    ).numpy()
    print('transformed_image cast')

    img = Image.fromarray(transformed_image, mode="RGB")
    output_path = 'test_save_image.png'
    img.save(output_path)
    return output_path


if __name__ == "__main__":
    stylize_image("Pierre.jpg", "udnie", 2)
    
    # image = load_img('Pierre.jpg')
    
    # transformer = TransformerNet()
    # ckpt = tf.train.Checkpoint(transformer=transformer)
    # # ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()
    # ckpt.restore(tf.train.latest_checkpoint('models/style/udnie_celeba_sw100')).expect_partial()

    # transformed_image = transformer(image)
    # transformed_image = tf.cast(
    #     tf.squeeze(transformed_image), tf.uint8
    # ).numpy()

    # img = Image.fromarray(transformed_image, mode="RGB")
    # img.save('la_muse_contentlayer33_sw100.png')