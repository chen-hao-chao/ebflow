"""six center dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import numpy as np

_SIZE = 2
_NUM_POINTS = 50000
_DESCRIPTION = """
The 2d toy experiment dataset.
"""
_CITATION = """\
Nope.
"""

class checkerboard(tfds.core.GeneratorBasedBuilder):
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
            'position': tfds.features.Tensor(shape=(_SIZE,), dtype=tf.float64),
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
            gen_kwargs=dict(split='all')),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(split='test')),
    ]

  def _generate_examples(self, split):
    """Yields examples."""
    num = _NUM_POINTS if split == "train" else int(_NUM_POINTS / 5)
    r = 3
    x1 = np.random.rand(num) * 4 - 2
    x2_ = np.random.rand(num) - np.random.randint(0, 2, (num,)).astype('float64') * 2
    x2 = x2_ + np.floor(x1) % 2
    positions = np.stack((x1, x2),axis=1) * r

    labels = np.zeros(positions.shape[0], dtype=np.uint8)

    data = list(zip(positions, labels))
    for index, (position, label) in enumerate(data):
      record = {"position": position, "label": label}
      yield index, record