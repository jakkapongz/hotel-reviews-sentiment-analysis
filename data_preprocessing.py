import string
from pythainlp import word_tokenize
import pandas as pd
from thai2transformers import preprocess


def text_process(text):
    print(text)

    if type(text) is float:
        return " "

    t = text.lower()
    t = preprocess.fix_html(t)
    t = preprocess.rm_brackets(t)
    t = preprocess.replace_newlines(t)
    t = preprocess.rm_useless_spaces(t)
    t = preprocess.replace_spaces(t)
    t = preprocess.replace_rep_after(t)
    t = t.replace('\u200b', '')

    # final = "".join(u for u in t if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    # final = final.translate(str.maketrans('', '', string.punctuation))
    # tokens = tokens.translate(str.maketrans('','', string.punctuation))

    tokens = "".join(u for u in t if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    tokens = tokens.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(tokens, engine="newmm")

    tokens = preprocess.ungroup_emoji(tokens)
    tokens = preprocess.replace_wrep_post(tokens)

    final = " ".join(word for word in tokens)

    return final


def main():
    all_tsv_df = pd.read_csv(
        '/Users/j4kkapongz/Google Drive/Shared drives/EGIT697_THEMATIC/th_reviews_removed_duplicated_sort_unicode_tokenized.tsv',
        delimiter='\t')

    all_tsv_df['text_tokens'] = all_tsv_df['review_text'].apply(text_process)

    review_ids = []
    text_tokens = []
    token_length = []

    for idx, row in all_tsv_df.iterrows():
        print(row['text_tokens'])
        split_tokens = row['text_tokens'].split(' ')

        while '' in split_tokens:
            split_tokens.remove('')

        print(len(split_tokens))

        token_len = len(split_tokens)

        review_ids.append(row['review_id'])
        text_tokens.append(row['text_tokens'])
        token_length.append(token_len)

    data = {
        "review_id": review_ids,
        "text_tokens": text_tokens,
        "token_length": token_length
    }

    df = pd.DataFrame(data=data)

    df.to_csv('th_reviews_tokens.tsv', sep="\t")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
