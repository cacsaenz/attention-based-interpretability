import pandas as pd
import torch
import string
import numpy as np
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


class WordsWeightsCalculator:

    STOPWORDS = nltk.corpus.stopwords.words('portuguese') + list(string.punctuation)

    FOLDERS = [
        'bertimbau-base',
        'bert-base-multilingual-cased',
        'bert-base-multilingual-uncased',
    ]

    CHECKPOINTS = [
        'neuralmind/bert-base-portuguese-cased',
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

        for i in range(len(self.FOLDERS)):
            print(f' {self.FOLDERS[i]}...')
            self.output_folder = f'./outputs/{self.dataset}/{self.FOLDERS[i]}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.CHECKPOINTS[i]
            )
            results_df = self.__load_results()
            attention, captum = self.__load_weights()

            self.__test_weights(results_df, attention, captum)
            self.__save_results_statistics(results_df)
            self.__save_weights(results_df, attention, captum)
            self.__save_tfidf(results_df)

            print('  🔥 Done.\n')


    def __load_weights(self):
        attention_weights = torch.load(
            f'{self.output_folder}/attentions.pt'
        )
        captum_scores = torch.load(f'{self.output_folder}/captum.pt')
        print('  ✅ Weights loaded.')
        return attention_weights, captum_scores


    def __load_results(self):
        results_df = pd.read_csv(
            f'{self.output_folder}/test_results_complete.csv',
            sep=';'
        )
        results_df['words'] = results_df['words'].apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation))
        )
        results_df['tokens'] = results_df['tokens'].apply(lambda x: x.split())

        print('  ✅ Results loaded.')

        return results_df


    def __test_weights(self, results_df, attention, captum):
        for i, row in results_df.sample(10).iterrows():
            tokens = row.tokens
            filtered_tokens = [token for token in tokens if token != "[PAD]"]
            if len(tokens) != len(attention[i]):
                print(
                    f"  Attention error ({i}): `{row.tweet}` ({len(tokens)} vs {len(attention[i])})"
                )
                raise
            if len(filtered_tokens) != len(captum[i]):
                print(
                    f"  Captum error ({i}): `{row.tweet}` ({len(filtered_tokens)} vs {len(captum[i])})"
                )
                raise
        print('  ✅ Random weights test passed.')


    def __save_results_statistics(self, results_df):
        self.__save_counts(results_df)
        print('  ✅ Counts saved.')

        self.__save_statistics(results_df)
        print('  ✅ Statistics saved.')


    def __save_weights(self, results_df, attention, captum):
        attention_weight_map = {}
        captum_score_map = {}
        for i, row in results_df.iterrows():
            try:
                attention_weight_map = self.__get_words_attention_weights(
                    attention_weight_map,
                    row,
                    attention[i]
                )
            except:
                print(f'Attention error found in {i}: {row.tweet}')
                print(len(row.tokens), len(attention[i]))
                print(row.tokens)
                print(attention[i])
                raise

            try:
                captum_score_map = self.__get_words_captum_scores(
                    captum_score_map,
                    row,
                    # Negatives are transformed to zero
                    [0 if score < 0 else score for score in captum[i]]
                )
            except:
                print(f'Captum error found in {i}: {row.tweet}')
                print(row.tokens)
                print(captum[i])
                raise

        pd.DataFrame(
            attention_weight_map
        ).T.sort_index().to_csv(
            f'{self.output_folder}/words_attention.csv',
            index=None,
            sep=';'
        )
        print('  ✅ Attention weights saved.')

        pd.DataFrame(
            captum_score_map
        ).T.sort_index().to_csv(
            f'{self.output_folder}/words_captum_score.csv',
            index=None,
            sep=';'
        )
        print('  ✅ Captum scores saved.')


    def __save_counts(self, results_df):
        counts = {
            'total': 0,
        }
        for i, label in enumerate(self.label_mapper):
            counts[f'{label}_correct'] = len(
                results_df.loc[
                    (
                        results_df.got == i
                    ) & (
                        results_df.expected == results_df.got
                    )
                ]
            )
            counts[f'{label}_incorrect'] = len(
                results_df.loc[
                    (
                        results_df.got == i
                    ) & (
                        results_df.expected != results_df.got
                    )
                ]
            )
            counts[label] = counts[f'{label}_correct'] + counts[f'{label}_incorrect']
            counts['total'] += counts[label]

        pd.DataFrame(
            columns = ['name', 'count'],
            data = [ (key, value) for key, value in counts.items() ]
        ).to_csv(
            f'{self.output_folder}/result_counts.csv',
            index=None
        )


    def __save_statistics(self, results_df):
        counts = {}
        for i, labelA in enumerate(self.label_mapper):
            for j, labelB in enumerate(self.label_mapper):
                counts[f'{labelA}_{labelB}'] = len(
                    results_df.loc[
                        (
                            results_df.expected == i
                        ) & (
                            results_df.got == j
                        )
                    ]
                )
        pd.DataFrame(
            columns = ['name', 'count'],
            data = [ (key, value) for key, value in counts.items() ]
        ).to_csv(
            f'{self.output_folder}/result_statistics.csv',
            index=None
        )


    def __get_words_attention_weights(
        self,
        weights_map,
        tweet_row,
        weights,
    ):
        assert len(tweet_row.tokens) == len(weights)
        return self.__get_words_weights(
            weights_map,
            tweet_row,
            tweet_row.tokens,
            weights
        )


    def __get_words_captum_scores(
        self,
        weights_map,
        tweet_row,
        weights
    ):
        filtered_tokens = [token for token in tweet_row.tokens if token != '[PAD]']
        assert len(filtered_tokens) == len(weights)
        return self.__get_words_weights(
            weights_map,
            tweet_row,
            filtered_tokens,
            weights
        )


    # This is the algorithm to map tokens to words and aggregate their weights
    def __get_words_weights(
        self,
        weights_map,
        tweet_row,
        tokens,
        weights
    ):
        current_sub_text_indices = []
        for i, token in enumerate(tokens):
            try:
                if token in self.tokenizer.all_special_tokens:
                    continue
                if not token.startswith('##') and len(current_sub_text_indices):
                    word = " ".join(
                        [tokens[idx] for idx in current_sub_text_indices]
                    ).replace(" ##", "").strip()
                    average_weight = np.mean([weights[idx] for idx in current_sub_text_indices])
                    weights_map = self.__add_word_weight_to_mapping(
                        weights_map,
                        word,
                        [tokens[idx] for idx in current_sub_text_indices],
                        average_weight,
                        tweet_row.got,
                        tweet_row.expected
                    )
                    current_sub_text_indices = []
                current_sub_text_indices.append(i)
            except:
                print(i, token)
                print(current_sub_text_indices)
                raise

        if len(current_sub_text_indices):
            word = " ".join(
                        [tokens[idx] for idx in current_sub_text_indices]
                    ).replace(" ##", "").strip()
            average_weight = np.mean([weights[idx] for idx in current_sub_text_indices])
            words_map = self.__add_word_weight_to_mapping(
                weights_map,
                word,
                [tokens[idx] for idx in current_sub_text_indices],
                average_weight,
                tweet_row.got,
                tweet_row.expected
            )

        return words_map


    def __add_word_weight_to_mapping(
        self,
        weights_map,
        word,
        tokens,
        weight,
        gotten,
        expected
    ):
        # Considerations:
        # - A word can appear multiple times in the same tweet,
        #   or in many tweets, it is going to be counted each
        #   appearance.
        if word not in weights_map:
            weights_map[word] = {
                'word': word,
                'tokens': ' '.join(tokens),
            }
            for label in self.label_mapper:
                weights_map[word][f'{label}_correct_w'] = 0.0
                weights_map[word][f'{label}_correct_n'] = 0
                weights_map[word][f'{label}_incorrect_w'] = 0.0
                weights_map[word][f'{label}_incorrect_n'] = 0

        prediction_status = 'correct' if gotten == expected else 'incorrect'
        weights_map[word][
            f"{self.label_mapper[gotten]}_{prediction_status}_w"
        ] += weight
        weights_map[word][
            f"{self.label_mapper[gotten]}_{prediction_status}_n"
        ] += 1
        return weights_map


    def __save_tfidf(self, results_df):
        vectorizer = TfidfVectorizer(
            stop_words=self.STOPWORDS,
            use_idf=True
        )
        vectors = vectorizer.fit_transform(results_df.words.values)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()

        # Save TF-IDF weights for each word
        pd.DataFrame(
            denselist,
            columns=feature_names
        ).T.to_csv(
            f'{self.output_folder}/tfidf_weights.csv',
            sep=';'
        )

        # Save the indices of tweets of each group
        indices = {}
        for i, label in enumerate(self.label_mapper):
            subset = results_df[results_df.expected == i]
            indices[label] = {
                'all': list(subset.index),
                'correct': list(subset[subset.got == subset.expected].index),
                'incorrect': list(subset[subset.got != subset.expected].index),
            }
        indices_df = pd.DataFrame(indices)
        indices_df.insert(0, column='group', value=indices_df.index)
        indices_df.to_csv(
            f'{self.output_folder}/tfidf_indices.csv',
            sep=';',
            index=None
        )

        print('  ✅ Words TF-IDF saved.')


wwc = WordsWeightsCalculator()
wwc(
    'V_DS1',
    [
        'antisinovax',
        'antivax',
        'provax',
    ]
)
wwc(
    'V_DS2',
    [
        'antisinovax',
        'provax',
    ]
)
wwc(
    'V_DS3',
    [
        'antivax',
        'provax',
    ]
)
wwc(
    'SI_DS1',
    [
        'chloroquiner',
        'neutral',
        'quarentener',
    ]
)
wwc(
    'SI_DS2',
    [
        'chloroquiner',
        'quarentener',
    ]
)
wwc(
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