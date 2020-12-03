cd gpt2 && python train_bpe_tokenizer.py \
 --data-files ../shards/* \
 --vocab-size 64000 \
 --output-dir ../tokenizer/ \
 --output-file-name ""