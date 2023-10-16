"""multimodal dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import numpy as np
from ebflow.datasets.mm_sampler import fn
import torch

_SIZE = 10
_NUM_POINTS = 50000
_DESCRIPTION = """
The toy experiment dataset.
"""
_CITATION = """\
Nope.
"""

class multimodal10(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the toy dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'position': tfds.features.Tensor(shape=(_SIZE,), dtype=tf.float32),
            'label': tfds.features.ClassLabel(num_classes=1),
        }),
        supervised_keys=('position','label'),  # Set to `None` to disable
        homepage='None',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(split='train')),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(split='test')),
    ]

  def _generate_examples(self, split):
    """Yields examples."""
    dim = _SIZE
    num = _NUM_POINTS if split == "train" else int(_NUM_POINTS / 5)
    noise = torch.randn((num, dim)) #np.random.normal(0, 1, size=(num, dim))
    positions = fn(noise).detach().cpu().numpy()

    labels = np.zeros(positions.shape[0], dtype=np.uint8)
    data = list(zip(positions, labels))
    for index, (position, label) in enumerate(data):
      record = {"position": position, "label": label}
      yield index, record