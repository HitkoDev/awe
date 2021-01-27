# Siamese Network Tensorflow

Siamese network is a neural network that contain two or more identical subnetwork. The objective of this network is to find the similarity or comparing the relationship between two comparable things. Unlike classification task that uses cross entropy as the loss function, siamese network usually uses contrastive loss or triplet loss.

This model is based on InceptionResNetV2, with embedding size 256, and trained using contrastive loss.

# Train

To train the model, run `train.sh`.
