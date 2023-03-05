import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pythainlp.corpus.common import thai_stopwords

thai_stopwords = list(thai_stopwords())


def main():
    all_df = pd.read_csv(
        '/Users/j4kkapongz/Google Drive/Shared drives/EGIT697_THEMATIC/10000_good_bad_reviews_no_gap.tsv',
        delimiter='\t')

    # all_df['rating'].value_counts().plot.bar()

    # print(all_df)
    all_words = " ".join(text for text in all_df['review_tokens'])

    reg = r"[ก-๙a-zA-Z']+"
    fp = '/Users/j4kkapongz/Google Drive/Shared drives/EGIT697_THEMATIC/THSarabunNew.ttf'

    wordcloud = WordCloud(stopwords=thai_stopwords, background_color='white', max_words=2000, height=2000, width=4000,
                          font_path=fp, regexp=reg).generate(all_words)
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

    word_counts = WordCloud(stopwords=thai_stopwords, font_path=fp, regexp=reg).process_text(all_words)

    print(dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True)))

    # bbb = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))

    with open("./hotel_reviews_word_cloud_4.tsv", 'w', encoding='utf-8', errors='ignore') as file1:
        file1.writelines("{0}\t{1}\n".format("word", "frequency"))
        # loop over dictionary keys and values
        for key, val in word_counts.items():
            # write every key and value to file
            file1.writelines("{0}\t{1}\n".format(key, val))

    # plt.title("Hotel Reviews Domain")
    # plt.savefig("hotel_review_word_cloud.png")


if __name__ == '__main__':
    main()
