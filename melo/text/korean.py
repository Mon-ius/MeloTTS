import re
from transformers import AutoTokenizer

from . import punctuation
from melo.text.ko_dictionary import english_dictionary, etc_dictionary
from jamo import hangul_to_jamo
from pecab import PeCab

def normalize(text):
    text = text.strip()
    text = re.sub("[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]", "", text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    text = text.lower()
    return text


def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text


def normalize_english(text):
    def fn(m):
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text


# Map Korean jamo characters to the symbols used in the TTS system
KOREAN_SYMBOL_MAP = {
    'ᄀ': 'ᄀ', 'ᄁ': 'ᄁ', 'ᄂ': 'ᄂ', 'ᄃ': 'ᄃ', 'ᄄ': 'ᄄ',
    'ᄅ': 'ᄅ', 'ᄆ': 'ᄆ', 'ᄇ': 'ᄇ', 'ᄈ': 'ᄈ', 'ᄉ': 'ᄉ',
    'ᄊ': 'ᄊ', 'ᄋ': 'ᄋ', 'ᄌ': 'ᄌ', 'ᄍ': 'ᄍ', 'ᄎ': 'ᄎ',
    'ᄏ': 'ᄏ', 'ᄐ': 'ᄐ', 'ᄑ': 'ᄑ', 'ᄒ': 'ᄒ',
    # Medial vowels
    'ᅡ': 'ᅡ', 'ᅢ': 'ᅢ', 'ᅣ': 'ᅣ', 'ᅤ': 'ᅤ', 'ᅥ': 'ᅥ',
    'ᅦ': 'ᅦ', 'ᅧ': 'ᅧ', 'ᅨ': 'ᅨ', 'ᅩ': 'ᅩ', 'ᅪ': 'ᅪ',
    'ᅫ': 'ᅫ', 'ᅬ': 'ᅬ', 'ᅭ': 'ᅭ', 'ᅮ': 'ᅮ', 'ᅯ': 'ᅯ',
    'ᅰ': 'ᅰ', 'ᅱ': 'ᅱ', 'ᅲ': 'ᅲ', 'ᅳ': 'ᅳ', 'ᅴ': 'ᅴ',
    'ᅵ': 'ᅵ',
    # Final consonants - mapping some that appear to be missing
    'ᆨ': 'ᆨ', 'ᆫ': 'ᆫ', 'ᆮ': 'ᆮ', 'ᆯ': 'ᆯ', 'ᆷ': 'ᆷ',
    'ᆸ': 'ᆸ', 'ᆼ': 'ᆼ',
    # Map unexpected characters to similar or fallback characters
    'ᇂ': 'ᆼ',  # Map ᇂ to ᆼ as they are similar final consonants
    # Add other potential mappings as needed
}

# Define valid Korean symbols from the TTS system
VALID_KOREAN_SYMBOLS = set(['ᄌ', 'ᅥ', 'ᆫ', 'ᅦ', 'ᄋ', 'ᅵ', 'ᄅ', 'ᅴ', 'ᄀ', 'ᅡ', 
                            'ᄎ', 'ᅪ', 'ᄑ', 'ᅩ', 'ᄐ', 'ᄃ', 'ᅢ', 'ᅮ', 'ᆼ', 'ᅳ', 
                            'ᄒ', 'ᄆ', 'ᆯ', 'ᆷ', 'ᄂ', 'ᄇ', 'ᄉ', 'ᆮ', 'ᄁ', 'ᅬ', 
                            'ᅣ', 'ᄄ', 'ᆨ', 'ᄍ', 'ᅧ', 'ᄏ', 'ᆸ', 'ᅭ', 'ᄊ', 'ᅲ', 
                            'ᅨ', 'ᄈ', 'ᅱ', 'ᅯ', 'ᅫ', 'ᅰ', 'ᅤ'])

def map_to_valid_symbols(jamo_list):
    """Map jamo characters to valid symbols in the TTS system."""
    mapped_jamo = []
    for jamo in jamo_list:
        if jamo in VALID_KOREAN_SYMBOLS:
            mapped_jamo.append(jamo)
        elif jamo in KOREAN_SYMBOL_MAP:
            mapped_symbol = KOREAN_SYMBOL_MAP[jamo]
            if mapped_symbol in VALID_KOREAN_SYMBOLS:
                mapped_jamo.append(mapped_symbol)
            # Skip if mapped symbol is not valid
        # Skip characters not in the map
    return mapped_jamo


pecab_instance = None
def korean_text_to_phonemes(text, character: str = "hangeul") -> str:
    """
    The input and output values look the same, but they are different in Unicode.

    example :

        input = '하늘' (Unicode : \ud558\ub298), (하 + 늘)
        output = '하늘' (Unicode :\u1112\u1161\u1102\u1173\u11af), (ᄒ + ᅡ + ᄂ + ᅳ + ᆯ)

    """
    global pecab_instance  # pylint: disable=global-statement
    if pecab_instance is None:
        pecab_instance = PeCab()

    if character == "english":
        from anyascii import anyascii
        text = normalize(text)
        # Use PeCab for basic processing
        result = pecab_instance.pos(text)
        text = ' '.join([token[0] for token in result])
        text = anyascii(text)
        return text

    text = normalize(text)
    # Use PeCab for grapheme-to-phoneme conversion
    result = pecab_instance.pos(text)
    # Join the tokens
    processed_text = ''.join([token[0] for token in result])
    # Convert to jamo
    jamo_list = list(hangul_to_jamo(processed_text))  # '하늘' --> ['ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆯ']
    # Map to valid symbols
    mapped_jamo = map_to_valid_symbols(jamo_list)
    return "".join(mapped_jamo)

def text_normalize(text):
    text = normalize(text)
    return text


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

model_id = 'kykim/bert-kor-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    word2ph = []
    for group in ph_groups:
        text = ""
        for ch in group:
            text += ch
        if text == '[UNK]':
            phs += ['_']
            word2ph += [1]
            continue
        elif text in punctuation:
            phs += [text]
            word2ph += [1]
            continue

        phonemes = korean_text_to_phonemes(text)
        phone_len = len(phonemes)
        word_len = len(group)

        aaa = distribute_phone(phone_len, word_len)
        assert len(aaa) == word_len
        word2ph += aaa

        phs += phonemes
    phones = ["_"] + phs + ["_"]
    tones = [0 for i in phones]
    word2ph =  [1] + word2ph + [1]
    assert len(word2ph) == len(tokenized) + 2
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device='cuda'):
    from . import japanese_bert
    return japanese_bert.get_bert_feature(text, word2ph, device=device, model_id=model_id)