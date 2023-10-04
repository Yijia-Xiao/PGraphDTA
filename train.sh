for i in {0..4};
do
    echo CUDA_VISIBLE_DEVICES=0,1,2,3 python dti_train_dist_contact_map.py --epochs 1500 --dataset_choice 1 --prot_lm_model_choice $i --num_gpus 4
done
