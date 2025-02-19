
for i in `seq 0 9`
do
  datainfo=$(printf "./json_files/train_ETRI_%03d.json" ${i})
  log_file=$(printf "Tube_fps_cross_fuse_1216ETRI_%03d.txt" ${i})
  CUDA_VISIBLE_DEVICES=3 python train.py \
   --database='ETRI' \
   --model_name='Tube_fps_cross_fuse' \
   --conv_base_lr=0.0001 \
   --datainfo=${datainfo} \
   --train_batch_size=4 --epochs=100 \
   --videos_dir='../ETRI/frames' \
   --log_file=${log_file} \
   --exp_version=${i} \
   --num_workers=8 \
   --log_path='./output/Tube/' \
   --imgsize=224 \
   --test_videos_dir='../BVI-HFR/frames' \
   --test_datainfo="../json_files/train_BVI_000.json" \
   --features_dir='./features/ETRI/res50_imgnet_last_ms' \
   --test_features_dir='./features/BVI-HFR/res50_imgnet_last_ms' \
   --read_num=16
done
