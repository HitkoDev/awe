import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = './logs/20180402-114759/saved_model.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [],
            './logs/20180402-114759/',
        )
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes = [n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        saver = tf.train.Saver()
        saver.restore('./logs/20180402-114759/model-20180402-114759.ckpt-275.data-00000-of-00001')
