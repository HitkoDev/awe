import tensorflow as tf
import tf_slim as slim


def head(endpoints, embedding_dim, is_training):
    endpoints['head_output'] = slim.fully_connected(
        endpoints['model_output'], 1024, normalizer_fn=slim.batch_norm,
        normalizer_params={
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'updates_collections': tf.compat.v1.GraphKeys.UPDATE_OPS,
        })

    endpoints['emb_raw'] = slim.fully_connected(
        endpoints['head_output'], embedding_dim, activation_fn=None,
        weights_initializer=tf.compat.v1.orthogonal_initializer(), scope='emb')
    endpoints['emb'] = tf.identity(tf.nn.dropout(endpoints['emb_raw'], 0.5), name="out_emb")

    return endpoints
