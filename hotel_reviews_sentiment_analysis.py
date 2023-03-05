import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_recall_curve
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def main():
    # /Users/j4kkapongz/ITM64/EGIT697_THEMATIC/10000_good_bad_reviews_no_gap_02012023.pkl

    all_df = pd.read_pickle(
        '/Users/j4kkapongz/Google Drive/Shared drives/EGIT697_THEMATIC/10000_good_bad_reviews_no_gap_02012023_completed.pkl')

    all_df = tfidf_feature_extractions(all_df)
    all_df = bow_feature_extractions(all_df)

    all_df['label'].value_counts().plot.bar()

    plt.show()

    # print(all_df.columns)

    run_by_repeated_kflod(all_df, 5, 1, 400)
    # run_by_cross_validation(all_df)


def run_by_repeated_kflod(df, split_num, repeat_num, random_state):
    rkf = RepeatedKFold(n_splits=split_num, n_repeats=repeat_num, random_state=random_state)

    tasks = {
        "bert": {
            'col': 'content_bert_vector',
            'language_model': 'BERT (multilingual)'
        },
        "xlmr": {
            'col': 'content_xlmr_vector',
            'language_model': 'XLM-RoBERTa (multilingual)'
        },
        "wangchan": {
            'col': 'content_wangchan_vector',
            'language_model': 'WangchanBERTa (monolingual)'
        },
        "tfidf": {
            'col': 'content_tfidf_vector',
            'language_model': 'TF-IDF'
        },
        "bow": {
            'col': 'content_bow_vector',
            'language_model': 'Bag Of Word'
        }
    }

    predict_round = 1

    result_table = pd.DataFrame(columns=['models', 'language_model', 'predict_round', 'predict_prob',
                                         'test_label', 'fpr', 'tpr', 'auc'])

    for train_indexes, test_indexes in rkf.split(df):

        print(train_indexes)
        print(test_indexes)

        content_bert_vector = np.array(df['content_bert_vector'])
        content_roberta_vector = np.array(df['content_wangchan_vector'])
        content_xlmr_vector = np.array(df['content_xlmr_vector'])
        content_tfidf_vector = np.array(df['content_tfidf_vector'])
        content_bow_vector = np.array(df['content_bow_vector'])

        label = np.array((df['label']))
        rating = np.array((df['rating']))
        review_text = np.array((df['review_text']))

        train_label, test_label, test_rating, test_review_text = label[train_indexes].tolist(), \
            label[test_indexes].tolist(), \
            rating[test_indexes].tolist(), \
            review_text[test_indexes].tolist()
        for task in tasks:

            target_train_vector = None
            target_test_vector = None

            vector_col_name = tasks[task]['col']

            if vector_col_name == 'content_bert_vector':
                target_train_vector, target_test_vector = content_bert_vector[train_indexes].tolist(), \
                    content_bert_vector[test_indexes].tolist()

            if vector_col_name == 'content_wangchan_vector':
                target_train_vector, target_test_vector = content_roberta_vector[train_indexes].tolist(), \
                    content_roberta_vector[test_indexes].tolist()
            if vector_col_name == 'content_xlmr_vector':
                target_train_vector, target_test_vector = content_xlmr_vector[train_indexes].tolist(), \
                    content_xlmr_vector[test_indexes].tolist()

            if vector_col_name == 'content_tfidf_vector':
                target_train_vector, target_test_vector = content_tfidf_vector[train_indexes].tolist(), \
                    content_tfidf_vector[test_indexes].tolist()

            if vector_col_name == 'content_bow_vector':
                target_train_vector, target_test_vector = content_bow_vector[train_indexes].tolist(), \
                    content_bow_vector[test_indexes].tolist()

            language_model = tasks[task]['language_model']

            model, predicted_label = predict_by_logistic_regression(language_model, predict_round, target_train_vector,
                                                                    train_label, target_test_vector)

            display_confusion_metrix(language_model, predict_round, model, test_label, predicted_label)

            print(classification_report(predicted_label, test_label, digits=4))

            prob_result = predict_prob_by_logistic_regression(language_model, predict_round, target_train_vector,
                                                              train_label, target_test_vector, test_label)

            result_table = result_table.append(prob_result, ignore_index=True)

            write_wrong_predicted_list(language_model, predict_round, target_test_vector, test_label, test_rating,
                                       test_review_text, predicted_label)

        predict_round += 1

        break

    result_table.set_index('models', inplace=True)

    display_predict_curve(result_table, 'multiple_roc_curve', 'ROC Curve Analysis', 'False Positive Rate',
                          'True Positive Rate')
    display_predict_curve(result_table, 'multiple_precision_recall_curve', 'Precision and Recall Analysis', 'Recall',
                          'Precision')

    # def display_predict_curve(result_table, file_name, graph_title, xlabel, ylabel):
    # display_precision_recall_curve(result_table)


def display_precision_recall_curve(result_table):
    current_round = 1

    fig = plt.figure(figsize=(16, 12))

    for i in result_table.index:

        pr = result_table.loc[i]['predict_round']

        if current_round != pr:
            plt.xticks(np.arange(0.0, 1.1, step=0.1))
            plt.xlabel("Recall", fontsize=15)

            plt.yticks(np.arange(0.5, 1.0, step=0.1))
            plt.ylabel("Precision", fontsize=15)

            plt.title(f'Precision and Recall Analysis {current_round}', fontweight='bold', fontsize=15)
            plt.legend(prop={'size': 13}, loc='lower left')

            plt.show()

            fig.savefig(f'multiple_precision_recall_curve_{current_round}.png')

            current_round = current_round + 1
            fig = plt.figure(figsize=(16, 12))

        precision = result_table.loc[i]['precision']
        recall = result_table.loc[i]['recall']

        plt.plot(recall, precision, label="{}".format(i))

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall", fontsize=15)

    plt.yticks(np.arange(0.5, 1.0, step=0.1))
    plt.ylabel("Precision", fontsize=15)

    plt.title(f'Precision and Recall Analysis {current_round}', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower left')

    plt.show()

    fig.savefig(f'multiple_precision_recall_curve_{current_round}.png')


def display_roc_curve(result_table):
    current_round = 1

    fig = plt.figure(figsize=(16, 12))

    for i in result_table.index:

        pr = result_table.loc[i]['predict_round']

        if current_round != pr:
            plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

            plt.xticks(np.arange(0.0, 1.1, step=0.1))
            plt.xlabel("False Positive Rate", fontsize=15)

            plt.yticks(np.arange(0.0, 1.1, step=0.1))
            plt.ylabel("True Positive Rate", fontsize=15)

            plt.title(f'ROC Curve Analysis {current_round}', fontweight='bold', fontsize=15)
            plt.legend(prop={'size': 13}, loc='lower right')

            plt.show()

            fig.savefig(f'multiple_roc_curve_{current_round}.png')

            current_round = current_round + 1
            fig = plt.figure(figsize=(16, 12))

        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title(f'ROC Curve Analysis {current_round}', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()

    fig.savefig(f'multiple_roc_curve_{current_round}.png')


def display_predict_curve(result_table, file_name, graph_title, xlabel, ylabel):
    current_round = 1

    fig = plt.figure(figsize=(16, 12))

    for i in result_table.index:

        pr = result_table.loc[i]['predict_round']

        if current_round != pr:
            plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

            plt.xticks(np.arange(0.0, 1.1, step=0.1))
            plt.xlabel(xlabel, fontsize=15)

            plt.yticks(np.arange(0.0, 1.1, step=0.1))
            plt.ylabel(ylabel, fontsize=15)

            plt.title(f'{graph_title} {current_round}', fontweight='bold', fontsize=15)
            if graph_title.startswith('ROC'):
                plt.legend(prop={'size': 13}, loc='lower right')
            else:
                plt.legend(prop={'size': 13}, loc='lower left')

            plt.show()

            fig.savefig(f'{file_name}_{current_round}.png')

            current_round = current_round + 1
            fig = plt.figure(figsize=(16, 12))

        if graph_title.startswith('ROC'):
            plt.plot(result_table.loc[i]['fpr'],
                     result_table.loc[i]['tpr'],
                     label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
        else:
            plt.legend(prop={'size': 13}, loc='lower left')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel(xlabel, fontsize=15)

    plt.yticks(np.arange(0.5, 1.0, step=0.1))
    plt.ylabel(ylabel, fontsize=15)

    plt.title(f'{graph_title} {current_round}', fontweight='bold', fontsize=15)
    if graph_title.startswith('ROC'):
        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
        plt.legend(prop={'size': 13}, loc='lower right')
    else:
        plt.legend(prop={'size': 13}, loc='lower left')

    plt.show()

    fig.savefig(f'{file_name}_{current_round}.png')


def run_by_train_test_split(df, test_size, random_state):
    train_df, test_df, train_label, test_label = train_test_split(df,
                                                                  list(df['label']),
                                                                  test_size=test_size,
                                                                  random_state=random_state)

    tasks = {
        "bert": {
            'col': 'content_bert_vector',
            'language_model': 'BERT (multilingual)'
        },
        "xlmr_df": {
            'col': 'content_xlmr_vector',
            'language_model': 'XML-RoBERTa (multilingual)'
        },
        "wangchan": {
            'col': 'content_wangchan_vector',
            'language_model': 'WangchanBERTa (monolingual)'
        },
        "tfidf": {
            'col': 'content_tfidf_vector',
            'language_model': 'TF-IDF'
        },
        "bow": {
            'col': 'content_bow_vector',
            'language_model': 'Bag Of Word'
        },
    }

    for task in tasks:
        vector_col_name = tasks[task]['col']
        target_train_vector = list(train_df[vector_col_name])
        target_train_label = list(train_label)

        target_test_vector = list(test_df[vector_col_name])
        target_test_label = list(test_label)
        target_test_rating = list(test_df['rating'])
        target_test_reviews = list(test_df['review_text'])

        language_model = tasks[task]['language_model']

        model, predicted_label = predict_by_logistic_regression(language_model, 1, target_train_vector,
                                                                target_train_label, target_test_vector)

        display_confusion_metrix(language_model, 1, model, target_test_label, predicted_label)

        print(classification_report(predicted_label, target_test_label, digits=4))

        write_wrong_predicted_list(language_model, 1, target_test_vector, target_test_label, target_test_rating,
                                   target_test_reviews, predicted_label)


def predict_by_logistic_regression(language_model, predicted_round, train_vector, train_label, test_vector):
    print("predict_by_logistic_regression -> language_model : {0}, round : {1}".format(language_model, predicted_round))

    logreg_model = LogisticRegression(max_iter=3000,
                                      random_state=0,
                                      multi_class='auto')

    logreg_model.fit(train_vector, train_label)
    predicted_label = logreg_model.predict(test_vector)

    return logreg_model, predicted_label


def predict_prob_by_logistic_regression(language_model, predicted_round, train_vector, train_label, test_vector,
                                        test_label):
    print("predict_prob_by_logistic_regression -> language_model : {0}, round : {1}".format(language_model,
                                                                                            predicted_round))

    logreg_model = LogisticRegression(max_iter=3000,
                                      random_state=0,
                                      multi_class='auto')

    logreg_model.fit(train_vector, train_label)
    predicted_prob = logreg_model.predict_proba(test_vector)

    predicted_prob = predicted_prob[:, 1]
    fpr, tpr, _ = metrics.roc_curve(test_label, predicted_prob)
    precision, recall, _ = precision_recall_curve(test_label, predicted_prob)

    auc = round(metrics.roc_auc_score(test_label, predicted_prob), 4)

    return {
        'models': f'{language_model}, round : {predicted_round}',
        'language_model': language_model,
        'predict_round': predicted_round,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc,
        'precision': precision,
        'recall': recall
    }


def display_confusion_metrix(language_model, predicted_round, model, test_label, predicted_label):
    cm = confusion_matrix(test_label, predicted_label, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.title("{0} {1}".format(language_model, predicted_round))
    plt.savefig("{0}_{1}.png".format(language_model, predicted_round))

    plt.show()


def write_wrong_predicted_list(language_model, predicted_round, test_vector, test_label, test_rating, test_review_text,
                               predicted_label):
    wrong_predicted_list = []

    for vector, label, rating, review_text, predicted in zip(test_vector, test_label, test_rating,
                                                             test_review_text, predicted_label):
        if predicted != label:
            wrong_predicted = {
                "language_model": language_model,
                "rating": rating,
                "label": label,
                "predicted_label": predicted,
                "review_text": review_text,
                "predicted_round": predicted_round
            }

            wrong_predicted_list.append(wrong_predicted)

    with open("./{0}_wrong_prediction_{1}.tsv".format(language_model, predicted_round), 'w', encoding='UTF-8') as file1:
        file1.writelines("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
            'language_model',
            'rating',
            'label',
            'predict_round',
            'predict_label',
            'review_text',
        ))
        for predicted in wrong_predicted_list:
            file1.writelines("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                predicted['language_model'],
                predicted['rating'],
                predicted['label'],
                predicted['predicted_round'],
                predicted['predicted_label'],
                predicted['review_text'],
            ))


def tfidf_feature_extractions(all_df):
    if 'content_tfidf_vector' in all_df.columns:
        return all_df

    print("extracting features by using tf-idf algorithm")

    tfidf_vectorizer = TfidfVectorizer(analyzer=lambda x: x.split(' '))

    tfidf_vec = tfidf_vectorizer.fit_transform(all_df['review_tokens'])
    tfidf_array = np.array(tfidf_vec.todense())

    content_tfidf = []

    for vec in tfidf_array:
        content_tfidf.append(vec)

    all_df['content_tfidf_vector'] = content_tfidf

    return all_df


def bow_feature_extractions(all_df):
    if 'content_bow_vector' in all_df.columns:
        return all_df

    print("extracting features by using bow algorithm")

    count_vectorizer = CountVectorizer(analyzer=lambda x: x.split(' '))

    bow_vec = count_vectorizer.fit_transform(all_df['review_tokens'])
    bow_array = np.array(bow_vec.todense())

    content_bow = []

    for vec in bow_array:
        content_bow.append(vec)

    all_df['content_bow_vector'] = content_bow

    return all_df


if __name__ == '__main__':
    main()
