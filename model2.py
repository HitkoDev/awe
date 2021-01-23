import os
import re

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_size = 299


def insert_layer_nonseq(model, layer_regex, insert_layer_factory, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update({layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update({model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            for new_layer in insert_layer_factory():
                x = new_layer(x)
                if position == 'before':
                    x = layer(x)

            break
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return tf.keras.Model(inputs=model.inputs, outputs=x)


def dropout_layer_factory():
    return[
        tf.keras.layers.Dropout(0.3),
        # tf.keras.layers.Dense(256, activation=None),
        # tf.keras.layers.Dropout(0.3),
        # tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ]


model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling='max'
)

model = insert_layer_nonseq(model, 'conv_7b_ac', dropout_layer_factory)
