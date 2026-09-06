#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_1.0/" --backbone_model_direct2 "./save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_1.0/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.5/" --backbone_model_direct2 "./save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.5/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.1/" --backbone_model_direct2 "./save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.1/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.05/" --backbone_model_direct2 "./save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.05/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.01/" --backbone_model_direct2 "./save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.01/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.005/" --backbone_model_direct2 "./save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.005/"



#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_1.0/"  --backbone_model_direct2 "./save/SupCon/tinyimgnet_models/2/tinyimgnet_resnet18_trail_0_128_1.0/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.5/"  --backbone_model_direct2 "./save/SupCon/tinyimgnet_models/2/tinyimgnet_resnet18_trail_0_128_0.5/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.1/"  --backbone_model_direct2 "./save/SupCon/tinyimgnet_models/2/tinyimgnet_resnet18_trail_0_128_0.1/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.05/"  --backbone_model_direct2 "./save/SupCon/tinyimgnet_models/2/tinyimgnet_resnet18_trail_0_128_0.05/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.01/"  --backbone_model_direct2 "./save/SupCon/tinyimgnet_models/2/tinyimgnet_resnet18_trail_0_128_0.01/"
#python3 pred_disagreement.py --backbone_model_direct1 "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.005/"  --backbone_model_direct2 "./save/SupCon/tinyimgnet_models/2/tinyimgnet_resnet18_trail_0_128_0.005/"


#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_1.0/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.5/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.1/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.05/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.01/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/2/cifar10_resnet18_trail_0_128_0.005/" --trail 6


python3 pre_model_knn.py --exemplar_features_path "/features6/cifar100_marco_resnet18_1trail_16_128_128_marco_data_16_train" --testing_known_features_path "/features6/cifar100_marco_resnet18_1trail_16_128_128_marco_data_16_test_known" --testing_unknown_features_path "/features6/cifar100_marco_resnet18_1trail_16_128_128_marco_data_23_test_known" --num_classes 20 --K 3
python3 pre_model_knn.py --exemplar_features_path "/features6/cifar100_marco_resnet18_1trail_17_128_128_marco_data_17_train" --testing_known_features_path "/features6/cifar100_marco_resnet18_1trail_17_128_128_marco_data_17_test_known" --testing_unknown_features_path "/features6/cifar100_marco_resnet18_1trail_17_128_128_marco_data_23_test_known" --num_classes 20 --K 3
python3 pre_model_knn.py --exemplar_features_path "/features6/cifar100_marco_resnet18_1trail_18_128_128_marco_data_18_train" --testing_known_features_path "/features6/cifar100_marco_resnet18_1trail_18_128_128_marco_data_18_test_known" --testing_unknown_features_path "/features6/cifar100_marco_resnet18_1trail_18_128_128_marco_data_23_test_known" --num_classes 20 --K 3

