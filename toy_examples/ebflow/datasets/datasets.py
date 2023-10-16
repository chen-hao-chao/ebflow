import tensorflow as tf
import tensorflow_datasets as tfds

import ebflow.datasets.swirl
import ebflow.datasets.sine
import ebflow.datasets.checkerboard
import ebflow.datasets.multimodal10

def get_dataset(config):
  """Construct data loaders for training and evaluation.
  Args:
    config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
  Returns:
    train_ds: (tf dataset iter) The dataset iterator.
  """
  # Compute the batch size.
  batch_size = config['batch_size']
  eval_batch_size = config['batch_size_eval']

  # Set buffer size.
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None

  # Build datasets.
  if config['dataset'] in ['checkerboard', 'swirl', 'sine', 'multimodal10']:
    dataset_builder = tfds.builder(config['dataset'])
    train_split_name = 'train'
    test_split_name = 'test'
  else:
    raise NotImplementedError('Dataset {} not yet supported.'.format(config['dataset']))

  # Customize preprocess functions for each dataset.
  preprocess_fn = lambda x: x

  def create_dataset(dataset_builder, split):
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(split=split, shuffle_files=True) 
    else:
      ds = dataset_builder
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True) if split == "train" else ds.batch(eval_batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  test_ds = create_dataset(dataset_builder, test_split_name)
  return train_ds, test_ds