import torch.nn.functional as F
from transformers import AutoTokenizer


def encode_dext(comment_text, embeddings):

    tokenizer = AutoTokenizer.from_pretrained(embeddings)
    encoding = tokenizer(
            comment_text,
            add_special_tokens=True,
            max_length=60,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

    return encoding


def test_model_text(text, trained_model,
                    embeddings, type, verbose=None):
    encoding = encode_dext(text, embeddings)
    trained_model.to('cpu')
    pred = trained_model(
        encoding['input_ids'], encoding['attention_mask']).logits
    pred = F.softmax(pred, dim=1).argmax().item()

    if verbose:
        if pred == 1:
            print(f'Is {type}')
        else:
            print(f'Not {type}')

    return pred
