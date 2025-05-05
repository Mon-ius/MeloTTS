from .fr_phonemizer import cleaner as fr_cleaner
from .fr_phonemizer import fr_to_ipa
from transformers import AutoTokenizer


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def text_normalize(text):
    text = fr_cleaner.french_cleaners(text)
    return text

model_id = 'dbmdz/bert-base-french-europeana-cased'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(text, pad_start_end=True, tokenized=None):
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones = []
    tones = []
    word2ph = []
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            phone_list = list(filter(lambda p: p != " ", fr_to_ipa.fr2ipa(w)))
        
        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    from text import french_bert
    return french_bert.get_bert_feature(text, word2ph, device=device)