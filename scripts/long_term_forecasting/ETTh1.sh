export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=CALF

# 遍历不同的预测长度
for pred_len in 96 192 336 720
do

python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 128 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --itr 1 \
    --model $model \
    --patience 10 \
    --vit_path google/vit-base-patch16-224-in21k \
    --gpt_path gpt2

echo '====================================================================================================================='
done