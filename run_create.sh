ls ../shards/ | xargs -n 1 -P 6 -I{} python3 create_pretraining_data.py --input_file=../shards/{} --output_file=../pretraining_data/{} --tokenizer_dir=tokenizer/