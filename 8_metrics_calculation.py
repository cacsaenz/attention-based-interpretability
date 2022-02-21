import pandas as pd
from ast import literal_eval
import nltk
import string
import numpy as np


class WordsMetricsCalculator:

    STOPWORDS = nltk.corpus.stopwords.words('portuguese') + list(string.punctuation)

    METRICS = [
        'attention',
        'captum',
        'tfidf',
    ]

    FOLDERS = [
        'bertimbau-base',
        'bert-base-multilingual-cased',
        'bert-base-multilingual-uncased',
    ]

    def __call__(
        self,
        dataset,
        label_mapper,
        folders=None,
        checkpoints=None,
        stopwords=None
    ):
        self.dataset = dataset
        self.label_mapper = label_mapper

        if folders is not None:
            self.FOLDERS = folders

        if checkpoints is not None:
            self.CHECKPOINTS = checkpoints

        if stopwords is not None:
            self.STOPWORDS = stopwords

        print(f'Dataset {self.dataset}:')

        for folder in self.FOLDERS:
            print(f' {folder}...')
            self.output_folder = f'./outputs/{self.dataset}/{folder}'
            self.words_metrics = {}

            self.__load_counts()
            self.__load_weights_df()
            self.__add_attention_weights()
            self.__add_captum_scores()
            self.__add_tfidf_scores()

            self.__save_metrics()

            print('  🔥 Done.\n')


    def __load_counts(self):
        self.counts = {}
        for _, row in pd.read_csv(
            f'{self.output_folder}/result_counts.csv'
        ).iterrows():
            self.counts[row['name']] = row['count']
        print('  ✅ Counts loaded.')


    def __load_weights_df(self):
        self.attention_df = self.__load_preprocess_weights_df(
            'words_attention.csv'
        )
        self.captum_score_df = self.__load_preprocess_weights_df(
            'words_captum_score.csv'
        )
        self.tfidf = self.__load_tfidf()
        print('  ✅ Weights loaded.')


    def __load_preprocess_weights_df(self, weights_df_name):
        df = pd.read_csv(
            f'{self.output_folder}/{weights_df_name}',
            sep=';'
        )
        df['tokens'] = df.tokens.apply(lambda x: str(x).split(' '))
        # Remove words in stopwords
        return df.loc[
            (df.word.str.len() > 1) &\
            (~df.word.isin(self.STOPWORDS)) &\
            (~df.word.isin(list(string.punctuation)))
        ]


    def __load_tfidf(self):
        self.tfidf_weights = pd.read_csv(
            f'{self.output_folder}/tfidf_weights.csv',
            sep=';',
            index_col=0
        )
        self.tfidf_indices = pd.read_csv(
            f'{self.output_folder}/tfidf_indices.csv',
            sep=';',
            index_col=0
        )
        for label in self.label_mapper:
            self.tfidf_indices[label] = self.tfidf_indices[label].apply(
                lambda x: literal_eval(x)
            )


    def __save_metrics(self):
        words_metrics_df = pd.DataFrame(self.words_metrics).T
        words_metrics_df.to_csv(
            f'{self.output_folder}/words_metrics.csv',
            index=None
        )
        print('  ✅ Metrics saved.')


    def __add_attention_weights(self):
        self.__add_weights(self.attention_df, 'attention')
        print('  ✅ Absolute attention obtained.')


    def __add_captum_scores(self):
        self.__add_weights(self.captum_score_df, 'captum')
        print('  ✅ Captum scores obtained.')


    def __add_weights(self, df, metric):
        for _, row in df.iterrows():
            if row.word not in self.words_metrics:
                self.words_metrics[row.word] = self.__create_word_dict(
                    row.word,
                    row.tokens
                )

            self.words_metrics[row.word][metric] = sum([
                row[f'{label}_correct_w'] + row[f'{label}_incorrect_w'] for label in self.label_mapper
            ]) / self.counts['total'] * 100

            self.words_metrics[row.word][f'{metric}_correct'] = sum([
                row[f'{label}_correct_w'] for label in self.label_mapper
            ]) / self.counts['total'] * 100

            self.words_metrics[row.word][f'{metric}_incorrect'] = sum([
                row[f'{label}_incorrect_w'] for label in self.label_mapper
            ]) / self.counts['total'] * 100

            for label in self.label_mapper:
                self.words_metrics[row.word][f'{label}_{metric}'] = (
                    row[f'{label}_correct_w'] + row[f'{label}_incorrect_w']
                ) / self.counts[label] * 100

                self.words_metrics[row.word][f'{label}_{metric}_correct'] = (
                    row[f'{label}_correct_w']
                ) / self.counts[label] * 100

                self.words_metrics[row.word][f'{label}_{metric}_incorrect'] = (
                    row[f'{label}_incorrect_w']
                ) / self.counts[label] * 100


    def __add_tfidf_scores(self):
        for _, row in self.tfidf_weights.iterrows():
            try:
                if row.name not in self.words_metrics:
                    self.words_metrics[row.name] = self.__create_word_dict(
                        row.name,
                        []
                    )
                self.words_metrics[row.name]['tfidf'] = np.mean(
                    [ row[col] for col in self.tfidf_weights.columns ]
                )
                for label in self.label_mapper:
                    self.words_metrics[row.name][f'{label}_tfidf'] = np.mean(
                        [ row[index] for index in self.tfidf_indices[label][0] ]
                    )
            except:
                print('  - Error with word:', row.name)
        print('  ✅ TFIDF scores obtained.')


    def __create_word_dict(self, word, tokens):
        word_dict = {
            'word': word,
            'tokens': ' '.join(tokens),
        }
        for metric in self.METRICS:
            word_dict[metric] = 0.0
            if metric != 'tfidf':
                word_dict[f'{metric}_correct'] = 0.0
                word_dict[f'{metric}_incorrect'] = 0.0

            for label in self.label_mapper:
                word_dict[f'{label}_{metric}'] = 0.0
                if metric != 'tfidf':
                    word_dict[f'{label}_{metric}_correct'] = 0.0
                    word_dict[f'{label}_{metric}_incorrect'] = 0.0
        return word_dict


wmc = WordsMetricsCalculator()
wmc(
    'V_DS1',
    [
        'antisinovax',
        'antivax',
        'provax',
    ]
)
wmc(
    'V_DS2',
    [
        'antisinovax',
        'provax',
    ]
)
wmc(
    'V_DS3',
    [
        'antivax',
        'provax',
    ]
)
wmc(
    'SI_DS1',
    [
        'chloroquiner',
        'neutral',
        'quarentener',
    ]
)
wmc(
    'SI_DS2',
    [
        'chloroquiner',
        'quarentener',
    ]
)
wmc(
    'H_DS',
    [
        'against',
        'neutral',
        'favor',
    ],
    [
        'bert-base-uncased',
    ],
    [
        'bert-base-uncased',
    ],
    nltk.corpus.stopwords.words('english') + list(string.punctuation)
)
