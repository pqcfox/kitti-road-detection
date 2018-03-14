"""Input pipeline for KITTI segmentation."""

def input_fn(data_files, gt_files, is_training, params):
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(data_files), tf.constant(gt_files)))
                   .shuffle(len(data_files))
                   .batch(params.batch_size)
                   .prefetch(1))
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(data_files), tf.constant(gt_files)))
                   .batch(params.batch_size)
                   .prefetch(1))
