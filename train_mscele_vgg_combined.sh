export CUDA_VISIBLE_DEVICES=1
python src/train_combined.py --logs_base_dir /home_wjf/G_face/ \
           --models_base_dir /home_wjf/G_face/ \
           --data_dir /home_wjf/MsCele_VGG_mtcnnpy \
           --image_size 128 \
           --model_def models.GRCNN \
           --lfw_dir /home_wjf/lfw_mtcnnpy_128_nodis_zero \
           --lfw_file_ext jpg \
           --optimizer MOM \
           --learning_rate -1 \
           --learning_rate_schedule_file data/learning_rate_schedule_classifier_msceleb.txt \
           --max_nrof_epochs 200 \
           --filter_min_nrof_images_per_class 30 \
           --keep_probability 1.0 \
           --random_flip \
           --random_crop \
           --weight_decay 1e-4 \
           --nrof_preprocess_threads 4 \
           --batch_size 128 \
           --epoch_size 1500 \
           --embedding_size 512 \
           --l2_constrained_scale_factor 64 \
           --m1 0.3 \
           --m2 0.2 \
#--finetune \
#--pretrained_model train/20171002-083400/model-20171002-083400.ckpt-92000
#--pretrained_model train/20170831-165210/model-20170831-165210.ckpt-94000
#:~/dataset/asian_faces_cropped/chinese:~/dataset/asian_faces_cropped/korean:~/dataset/asian_faces_cropped/japanese
#~/dataset/MsCele_mtcnnpy_5:~/dataset/vggface2_train_mtcnnpy_5:~/dataset/vggface2_test_mtcnnpy_5:~/dataset/CASIA_Image-DB_mtcnnpy_5:~/dataset/children_mtcnnpy_5 
#~/dataset/MultiPie_mtcnnpy_5:~/dataset/CAS-PEAL-MIX_mtcnnpy_5 
#~/dataset/MsCele_VGG_mtcnnpy:~/dataset/CASIA_Image-DB_mtcnnpy_5:~/dataset/children_mtcnnpy_5 
#~/dataset/MsCele_VGG_mtcnnpy:~/dataset/CASIA_Image-DB_mtcnnpy_5:~/dataset/children_mtcnnpy_5:~/dataset/MultiPie_mtcnnpy_5
