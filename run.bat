@ECHO off
if /I %1 == default goto :default
if /I %1 == train_bpe goto :train_bpe
if /I %1 == create_pretraining_data goto :create_pretraining_data
if /I %1 == train goto :train
if /I %1 == clean goto :clean

:default
echo Nothing Selected
goto :eof

:train_bpe
echo %1
python train_bpe_tokenizer.py ^
 --data-files data/shards/* ^
 --vocab-size 64000 ^
 --output-dir data/gpt2 ^
 --output-file-name ""
goto :eof

:create_pretraining_data
echo %1
python create_pretraining_data.py ^
 --input_file="data/shards/shards_assafir_00000" ^
 --output_file="data/pretraining_data/shards_assafir_00000" ^
 --tokenizer_dir="data/gpt2"
goto :eof

:train
echo %1
python run_pretraining.py ^
 --input_file="data/pretraining_data/*" ^
 --output_dir="data/pretraining_model/" ^
 --config_file="data/gpt2/small-hparams.json" ^
 --batch_size=4 ^
 --eval_batch_size=4 ^
 --num_train_steps=50 ^
 --num_warmup_steps=10 ^
 --learning_rate=1e-4 ^
 --save_checkpoints_steps=10 ^
 --max_seq_length=1024 ^
 --max_eval_steps=10 ^
 --optimizer="adafactor" ^
 --iterations_per_loop=50 ^
 --keep_checkpoint_max=5 ^
 --use_tpu=False ^
 --do_train=True ^
 --do_eval=True
goto :eof