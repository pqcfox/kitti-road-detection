"""Define the model."""

import tensorflow as tf
import model.tensorflow_fcn.fcn8_vgg as fcn8_vgg
import model.tensorflow_fcn.loss as fcn_loss

def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']
    vgg_fcn = fcn8_vgg.FCN8VGG()
    vgg_fcn.build(images, train=is_training, num_classes=2, debug=True)
    return vgg_fcn.upscore32, vgg_fcn.pred_up


def max_f1(logits, labels):
    precisions, precision_update_op = tf.metrics.precision_at_thresholds()
    recalls, recall_update_op = tf.metrics.recall_at_thresholds()
    fscores = 2 * precisions * recalls / (precisions + recalls)
    return fscores, tf.group([precision_update_op, recall_update_op])


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    masks = inputs['masks']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits, predictions = build_model(is_training, inputs, params)

    # Define loss and accuracy
    padding = tf.constant([[0, 0], [0, 0], [0, 0], [0, 1]])
    loss = fcn_loss.loss(tf.pad(logits, padding), tf.one_hot(labels, 3), 3, head=[1, 1, 0])
    # max_f1_score = max_f1(logits, labels)

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            # 'max_f1': max_f1(logits, labels)
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    # tf.summary.scalar('max_f1', max_f1_score)
    tf.summary.image('train_image', inputs['images'])
    tf.summary.image('label_image', inputs['labels'] * 127)
    tf.summary.image('mask_image', inputs['masks'] * 255)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['predictions'] = predictions
    model_spec['loss'] = loss
    # model_spec['max_f1'] = max_f1_score
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
