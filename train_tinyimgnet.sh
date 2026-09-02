#python3 pre_models_training.py --batch_size 128 --lr 3e-4 --dataset "imagenet50" --model "vit16" --lr_decay_epochs "100,150" --epochs 200 --lr_decay_rate 0.1

python3 main_supcon.py --batch_size 256 --epochs 600 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet_multi" --datasets "tinyimgnet" --method "SupCon" --trail 0 --temp1 1. --temp2 1. --temp3 1.
python3 main_supcon.py --batch_size 256 --epochs 600 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet_multi" --datasets "tinyimgnet" --method "SupCon" --trail 0 --temp1 0.5 --temp2 0.5 --temp3 0.5