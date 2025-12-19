from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

def get_tok(mode_name = "answerdotai/ModernBERT-base", bos_token="<BOS>", eos_token="<EOS>", start_token="<START_ID>", end_token="<END_ID>", eot_token="<EOT_ID>"):
    tok = AutoTokenizer.from_pretrained(mode_name)

if __name__ == "__main__":
    get_tok()






