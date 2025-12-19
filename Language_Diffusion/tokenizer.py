from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

def get_tok(mode_name = "answerdotai/ModernBERT-base", bos_token="<BOS>", eos_token="<EOS>", start_token="<START_ID>", end_token="<END_ID>", eot_token="<EOT_ID>"):
    tok = AutoTokenizer.from_pretrained(mode_name)

    special_tokens = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "additional_special_tokens": [
                start_token, end_token, eot_token]
            }
    tok.add_special_tokens(special_tokens)

    tok.pad_token = eos_token
    tok.cls_token = bos_token

    tok._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            special_tokens=[
                    (bos_token, tok.bos_token_id),
                    (eos_token, tok.eos_token_id)
                ]
            )

    return tok


def test_tok():
    tok_pre = get_tok()
    text = "Hi there"
    ids = tok_pre(text, padding=True, return_tensors="pt")["input_ids"][0]
    dec = tok_pre.decode(ids, skip_special_tokens=False)
    print(dec)


if __name__ == "__main__":
    test_tok()






