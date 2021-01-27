#/bin/bash
python train.py --train_set images/train.csv --test_set ../../images/test.csv --image_root ../../images --experiment_root ./embedding --batch_p 50 --flip_augment --rotate_augment --learning_rate 0.000001
