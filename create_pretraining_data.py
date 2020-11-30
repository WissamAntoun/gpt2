
import json
from gpt_2.src import model, encoder


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", None, "Input raw text file (or comma-separated list of files)."
)

flags.DEFINE_string(
    "output_file", None, "Output TF example file (or comma-separated list of files)."
)

flags.DEFINE_string(
    "vocab_file", None, "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_string(
    "merges_file", None, "The vocabulary file that the BERT model was trained on."
)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    with open(FLAGS.vocab_file, 'r') as f:
        en = json.load(f)
    with open(FLAGS.merg_file, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
        
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

    enc = encoder.Encoder(
        encoder=en,
        bpe_merges=bpe_merges,
    )

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    for input_file in input_files:
        



if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("merges_file")
    tf.app.run()