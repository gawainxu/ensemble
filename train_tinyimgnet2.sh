#python3 pre_models_training.py --batch_size 128 --lr 0.1 --dataset "imagenet50" --model "vit16" --lr_decay_epochs "60,120,160,200" --epochs 300 --lr_decay_rate 0.2

python3 main_supcon.py --batch_size 256 --epochs 600 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet_multi" --datasets "tinyimgnet" --method "SupCon" --trail 0 --temp1 0.1 --temp2 0.1 --temp3 0.1
python3 main_supcon.py --batch_size 256 --epochs 600 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet_multi" --datasets "tinyimgnet" --method "SupCon" --trail 0 --temp1 0.05 --temp2 0.05 --temp3 0.05
