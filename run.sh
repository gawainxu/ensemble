#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_1.0/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.5/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.1/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.05/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.01/" --trail 6
#python3 main_linear.py --backbone_model_direct "/save/SupCon/cifar10_models/1/cifar10_resnet18_trail_0_128_0.005/" --trail 6

#python3 main_linear.py --backbone_model_direct "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_1.0/" --datasets "tinyimgnet" --trail 5
#python3 main_linear.py --backbone_model_direct "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.5/" --datasets "tinyimgnet" --trail 5
#python3 main_linear.py --backbone_model_direct "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.1/" --datasets "tinyimgnet" --trail 5
#python3 main_linear.py --backbone_model_direct "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.05/" --datasets "tinyimgnet" --trail 5
#python3 main_linear.py --backbone_model_direct "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.01/" --datasets "tinyimgnet" --trail 5
#python3 main_linear.py --backbone_model_direct "/save/SupCon/tinyimgnet_models/1/tinyimgnet_resnet18_trail_0_128_0.005/" --datasets "tinyimgnet" --trail 5


#echo "1.0"
#python3 main_testing_multiheads.py --datasets "cifar10" --trail 0 --num_classes 6 --exemplar_features_path "/features/cifar10_resnet_multi_trail_0_128_512_1.0_1.0_1.0_256_train" --testing_known_features_path "/features/cifar10_resnet_multi_trail_0_128_512_1.0_1.0_1.0_256_test_known" --testing_unknown_features_path "/features/cifar10_resnet_multi_trail_0_128_512_1.0_1.0_1.0_256_test_unknown"
#echo "0.5"
#python3 main_testing_multiheads.py --datasets "cifar10" --trail 0 --num_classes 6 --exemplar_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.5_0.5_0.5_256_train" --testing_known_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.5_0.5_0.5_256_test_known" --testing_unknown_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.5_0.5_0.5_256_test_unknown"
#echo "0.1"
#python3 main_testing_multiheads.py --datasets "cifar10" --trail 0 --num_classes 6 --exemplar_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.1_0.1_0.1_256_train" --testing_known_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.1_0.1_0.1_256_test_known" --testing_unknown_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.1_0.1_0.1_256_test_unknown"
#echo "0.05"
#python3 main_testing_multiheads.py --datasets "cifar10" --trail 0 --num_classes 6 --exemplar_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.05_0.05_0.05_256_train" --testing_known_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.05_0.05_0.05_256_test_known" --testing_unknown_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.05_0.05_0.05_256_test_unknown"
#echo "0.01"
#python3 main_testing_multiheads.py --datasets "cifar10" --trail 0 --num_classes 6 --exemplar_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.01_0.01_0.01_256_train" --testing_known_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.01_0.01_0.01_256_test_known" --testing_unknown_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.01_0.01_0.01_256_test_unknown"
#echo "0.005"
#python3 main_testing_multiheads.py --datasets "cifar10" --trail 0 --num_classes 6 --exemplar_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.005_0.005_0.005_256_train" --testing_known_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.005_0.005_0.005_256_test_known" --testing_unknown_features_path "/features/cifar10_resnet_multi_trail_0_128_512_0.005_0.005_0.005_256_test_unknown"

#echo "1.0"
#python3 main_testing_multiheads.py --datasets "tinyimgnet" --trail 0 --num_classes 20 --exemplar_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_1.0_1.0_1.0_256_train" --testing_known_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_1.0_1.0_1.0_256_test_known" --testing_unknown_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_1.0_1.0_1.0_256_test_unknown"
#echo "0.5"
#python3 main_testing_multiheads.py --datasets "tinyimgnet" --trail 0 --num_classes 20 --exemplar_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.5_0.5_0.5_256_train" --testing_known_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.5_0.5_0.5_256_test_known" --testing_unknown_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.5_0.5_0.5_256_test_unknown"
#echo "0.1"
#python3 main_testing_multiheads.py --datasets "tinyimgnet" --trail 0 --num_classes 20 --exemplar_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.1_0.1_0.1_256_train" --testing_known_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.1_0.1_0.1_256_test_known" --testing_unknown_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.1_0.1_0.1_256_test_unknown"
#echo "0.05"
#python3 main_testing_multiheads.py --datasets "tinyimgnet" --trail 0 --num_classes 20 --exemplar_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.05_0.05_0.05_256_train" --testing_known_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.05_0.05_0.05_256_test_known" --testing_unknown_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.05_0.05_0.05_256_test_unknown"
#echo "0.01"
#python3 main_testing_multiheads.py --datasets "tinyimgnet" --trail 0 --num_classes 20 --exemplar_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.01_0.01_0.01_256_train" --testing_known_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.01_0.01_0.01_256_test_known" --testing_unknown_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.01_0.01_0.01_256_test_unknown"
#echo "0.005"
#python3 main_testing_multiheads.py --datasets "tinyimgnet" --trail 0 --num_classes 20 --exemplar_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.005_0.005_0.005_256_train" --testing_known_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.005_0.005_0.005_256_test_known" --testing_unknown_features_path "/features/tinyimgnet_resnet_multi_trail_0_128_512_0.005_0.005_0.005_256_test_unknown"


python3 pre_model_knn.py --exemplar_features_path "/features/cifar100_marco_resnet18_1trail_16_128_128_data_16_train" --testing_known_features_path "/features/cifar100_marco_resnet18_1trail_16_128_128_data_16_test_known" --testing_unknown_features_path "/features/cifar100_marco_resnet18_1trail_16_128_128_data_23_test_known"
python3 pre_model_knn.py --exemplar_features_path "/features/cifar100_marco_resnet18_1trail_17_128_128_data_17_train" --testing_known_features_path "/features/cifar100_marco_resnet18_1trail_17_128_128_data_17_test_known" --testing_unknown_features_path "/features/cifar100_marco_resnet18_1trail_17_128_128_data_23_test_known"
python3 pre_model_knn.py --exemplar_features_path "/features/cifar100_marco_resnet18_1trail_18_128_128_data_18_train" --testing_known_features_path "/features/cifar100_marco_resnet18_1trail_18_128_128_data_18_test_known" --testing_unknown_features_path "/features/cifar100_marco_resnet18_1trail_18_128_128_data_23_test_known"
python3 pre_model_knn.py --exemplar_features_path "/features/cifar100_marco_resnet18_1trail_19_128_128_data_19_train" --testing_known_features_path "/features/cifar100_marco_resnet18_1trail_19_128_128_data_19_test_known" --testing_unknown_features_path "/features/cifar100_marco_resnet18_1trail_19_128_128_data_23_test_known"
python3 pre_model_knn.py --exemplar_features_path "/features/cifar100_marco_resnet18_1trail_20_128_128_data_20_train" --testing_known_features_path "/features/cifar100_marco_resnet18_1trail_20_128_128_data_20_test_known" --testing_unknown_features_path "/features/cifar100_marco_resnet18_1trail_20_128_128_data_23_test_known"
python3 pre_model_knn.py --exemplar_features_path "/features/cifar100_marco_resnet18_1trail_21_128_128_data_21_train" --testing_known_features_path "/features/cifar100_marco_resnet18_1trail_21_128_128_data_21_test_known" --testing_unknown_features_path "/features/cifar100_marco_resnet18_1trail_21_128_128_data_23_test_known"
python3 pre_model_knn.py --exemplar_features_path "/features/cifar100_marco_resnet18_1trail_22_128_128_data_22_train" --testing_known_features_path "/features/cifar100_marco_resnet18_1trail_22_128_128_data_22_test_known" --testing_unknown_features_path "/features/cifar100_marco_resnet18_1trail_22_128_128_data_23_test_known"