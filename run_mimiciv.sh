export CUDA_VISIBLE_DEVICES=0

# for mixup_level in 'batch_seq_feature'
# do
# /bin/python  -W ignore main_mimiciv.py  --num_train_epochs 8  --modeltype 'TS' \
#                 --kernel_size 1 --train_batch_size 8 --eval_batch_size 8   --seed 0 \
#                 --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 3 \
#                 --ts_learning_rate  0.0004 --txt_learning_rate 0.00002 \
#                 --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
#                 --output_dir "run/TS/" --embed_dim 128 \
#                 --model_name "bioLongformer"\
#                 --task 'ihm-48'\
#                 --file_path 'Data/mimic4-ihm' \
#                 --num_labels 2 \
#                 --num_heads 8\
#                 --embed_time 64\
#                 --tt_max 48\
#                 --TS_mixup\
#                 --mixup_level $mixup_level\
#                 --fp16 \
#                 --irregular_learn_emb_text \
#                 --irregular_learn_emb_ts \
#                 --cross_method "self_cross" \
#                 --reg_ts
# done


python -W ignore main_mimiciv.py  --num_train_epochs 8  --modeltype 'TS_Text' \
                --kernel_size 1 --train_batch_size 2 --eval_batch_size 8   --seed 42 \
                --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 3 \
                --ts_learning_rate  0.0004 --txt_learning_rate 0.00002 \
                --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
                --output_dir "run/TS_Text/" --embed_dim 128 \
                --model_name "bioLongformer"\
                --task 'ihm-48'\
                --file_path 'Data/mimic4-ihm' \
                --num_labels 2 \
                --num_heads 8\
                --embed_time 64\
                --tt_max 48\
                --TS_mixup\
                --mixup_level 'batch'\
                --fp16 \
                --irregular_learn_emb_text \
                --irregular_learn_emb_ts \
                --cross_method "moe" \
                --gating_function "gaussian" \
                --num_of_experts 12 \
                --hidden_size 512 \
                --top_k 4 \
                --reg_ts