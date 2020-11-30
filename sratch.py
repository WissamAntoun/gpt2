import tqdm

combine = 50000

token_chunks = []
raw_text = ''
paths = files

for path in tqdm.tqdm(paths):
    with open(path, 'r', encoding='utf8', errors='ignore') as fp:
        raw_text += fp.read()
    if len(raw_text) >= combine:
        tokens = np.stack(enc_malay.encode(raw_text))
        token_chunks.append(tokens)
        raw_text = ''
    else:
        raw_text += '<|endoftext|>'