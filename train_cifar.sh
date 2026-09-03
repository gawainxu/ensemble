#python3 main_ce.py --batch_size 128 --learning_rate 1e-2 --dataset "cifar100_marco" --model "resnet18" --lr_decay_epochs "100,120" --epochs 300 --lr_decay_rate 0.01 --trail 0 --multiplier 4
#python3 main_ce.py --batch_size 128 --learning_rate 1e-2 --dataset "cifar100_marco" --model "resnet18" --lr_decay_epochs "100,120" --epochs 300 --lr_decay_rate 0.01 --trail 1 --multiplier 4
#python3 main_ce.py --batch_size 128 --learning_rate 1e-2 --dataset "cifar100_marco" --model "resnet18" --lr_decay_epochs "100,120" --epochs 300 --lr_decay_rate 0.01 --trail 2 --multiplier 4
#python3 main_ce.py --batch_size 128 --learning_rate 1e-2 --dataset "cifar100_marco" --model "resnet18" --lr_decay_epochs "100,120" --epochs 300 --lr_decay_rate 0.01 --trail 3 --multiplier 4

#python3 main_ce.py --batch_size 128 --learning_rate 1e-2 --dataset "cifar100_marco" --model "resnet18" --lr_decay_epochs "100,120" --epochs 300 --lr_decay_rate 0.01 --trail 4 --multiplier 2
#python3 main_ce.py --batch_size 128 --learning_rate 1e-2 --dataset "cifar100_marco" --model "resnet18" --lr_decay_epochs "100,120" --epochs 300 --lr_decay_rate 0.01 --trail 5 --multiplier 1.3
#python3 main_ce.py --batch_size 128 --learning_rate 1e-2 --dataset "cifar100_marco" --model "resnet18" --lr_decay_epochs "100,120" --epochs 300 --lr_decay_rate 0.01 --trail 6


#python3 main_supcon.py --batch_size 256 --epochs 600 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet_multi" --datasets "cifar10" --method "SupCon" --trail 0 --temp1 1. --temp2 1. --temp3 1.
python3 main_supcon.py --batch_size 256 --epochs 600 --learning_rate 0.001 --cosine True --lr_decay_rate 0.8 --model "resnet_multi" --datasets "cifar10" --method "SupCon" --trail 0 --temp1 0.5 --temp2 0.5 --temp3 0.5