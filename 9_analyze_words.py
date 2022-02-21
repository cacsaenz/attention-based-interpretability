from transformers import BertTokenizer
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ndcg_score
import nltk
import warnings

plt.rcParams['figure.figsize'] = [14.4, 10.8]
plt.rcParams['figure.dpi'] = 200


class WordsAnalyzer:

    RANK_TOPS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    TOPS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    COLORS = {
        'blue': '#2d93ad',
        'red': '#ff6978',
    }

    STUDY_CASES = [
        (
            'V_DS1',
            [
                'antisinovax',
                'antivax',
                'provax',
            ],
        ),
        (
            'V_DS2',
            [
                'antisinovax',
                'provax',
            ]
        ),
        (
            'V_DS3',
            [
                'antivax',
                'provax',
            ],
        ),
        (
            'SI_DS1',
            [
                'chloroquiner',
                'neutral',
                'quarentener',
            ]
        ),
        (
            'SI_DS2',
            [
                'chloroquiner',
                'quarentener',
            ]
        ),
        (
            'H_DS',
            [
                'against',
                'neutral',
                'favor',
            ],
        ),
    ]

    CLASSES_NAMES = {
        'antichina': 'Anti-sinovaxxers',
        'antivacina': 'Anti-vaxxers',
        'provacina': 'Pro-vaxxers',
        'against': 'Against',
        'neutral': 'Neutral',
        'favor': 'Favor',
        'chloroquiner': 'Chloroquiner',
        'quarentener': 'Quarentener',
    }

    FOLDERS = [
        'bertimbau-base',
        'bert-base-multilingual-cased',
        'bert-base-multilingual-uncased',
    ]

    FOLDERS_ENGLISH = [
        'bert-base-uncased',
    ]

    CHECKPOINTS = [
        'neuralmind/bert-base-portuguese-cased',
        'bert-base-multilingual-cased',
        'bert-base-multilingual-uncased',
    ]

    CHECKPOINTS_ENGLISH = [
        'bert-base-uncased',
    ]

    METRICS = {
        'attention': 'Attention',
        'captum': 'Captum score',
    }

    def __call__(self):

        self.statistical_tests_results = {
            'general': [],
            'piws': [],
        }
        self.rank_statistical_tests_results = []
        self.global_averages = []
        self.rank_scores = []
        self.bertopic_tfidf = []

        for dataset, label_mapper in self.STUDY_CASES:
            self.dataset = dataset
            self.label_mapper = label_mapper

            self.correlation_tests_results = {
                'general': [],
                'piws': [],
            }
            self.rank_correlation_tests_results = []

            print(f'Dataset {self.dataset}:')

            self.STOPWORDS = nltk.corpus.stopwords.words(
                'portuguese' if not self.dataset.startswith('hydroxychloroquine') else 'english'
            )
            print(f'  ✅ Stopwords loaded {self.STOPWORDS[:3]}.')

            folders = self.FOLDERS if not self.dataset.startswith('hydroxychloroquine') else self.FOLDERS_ENGLISH
            checkpoints = self.CHECKPOINTS if not self.dataset.startswith('hydroxychloroquine') else self.CHECKPOINTS_ENGLISH

            comparisons_saved = False

            self.__load_bertopic_words()

            for i in range(len(folders)):
                print(f'  \n{folders[i]}...')
                self.output_folder = f'./outputs/{self.dataset}/{folders[i]}'
                self.model_name = folders[i]
                self.tops_df = None
                self.piws_analysis_df = None
                self.class_tops_df = None
                self.rank_scores_df = None
                self.class_rank_scores_df = None

                self.__create_subfolder()

                self.__load_models_vocabulary(checkpoints[i])

                self.__load_counts()
                self.__load_metrics_df()
                self.__load_lwo_df()

                self.__save_important_words()

                if self.topic_words:
                    self.__bertopic_vs_tfidf()

                self.__save_correlation_tests_results()
                self.__save_rank_correlation_tests_results()
                self.__add_global_averages()
                self.__add_rank_scores_averages()
                self.__add_statistical_tests_results()

                if not comparisons_saved:
                    self.__save_tfidf_comparisons()
                    comparisons_saved = True

        self.__save_averages()
        self.__save_statistical_tests_results()
        self.__save_bertopic_vs_tfidf()

        print(' 🔥 Done.\n')


    def __create_subfolder(self):
        Path(
            f'{self.output_folder}/words/'
        ).mkdir(parents=True, exist_ok=True)
        print('  ✅ Subfolders created.')


    def __load_models_vocabulary(self, checkpoint):
        tokenizer = BertTokenizer.from_pretrained(
            checkpoint
        )
        self.vocab_words = list(tokenizer.vocab.keys())
        print('  ✅ Model vocabulary loaded.')


    def __load_counts(self):
        self.counts = {}
        for _, row in pd.read_csv(
            f'{self.output_folder}/result_counts.csv'
        ).iterrows():
            self.counts[row['name']] = row['count']
        print('  ✅ Counts loaded.')


    def __load_metrics_df(self):
        self.words_metrics = pd.read_csv(
            f'{self.output_folder}/words_metrics.csv'
        )
        print('  ✅ Metrics loaded.')


    def __load_lwo_df(self):
        self.lwo = pd.read_csv(
            f'{self.output_folder}/lwo.csv'
        )
        print('  ✅ Leave-one-out results loaded.')


    def __load_bertopic_words(self):
        try:
            initial_topic_words_df = pd.read_csv(
                f'./outputs/{self.dataset}/topic_words.csv',
                sep=';'
            )
        except:
            self.topic_words = None
            self.topic_words_sizes = None
            self.clean_topic_words = None
            self.clean_topic_words_sizes = None
            print(f'  ❌ Topic words not found.')
            return

        topic_words_df = self.__filter_bertopic_words(initial_topic_words_df)
        clean_topic_words_df = self.__filter_bertopic_words(initial_topic_words_df, clean=True)

        self.topic_words = set(topic_words_df['Word'].values)
        self.topic_words_sizes = int(len(self.topic_words) / self.TOPS[0])
        self.clean_topic_words = set(clean_topic_words_df['Word'].values)
        self.clean_topic_words_sizes = int(len(self.clean_topic_words) / self.TOPS[0])
        print(f'  ✅ Topic words loaded ({len(self.topic_words)} out of {len(initial_topic_words_df)}).')
        print(f'  ✅ Clean topic words loaded ({len(self.clean_topic_words)}out of {len(initial_topic_words_df)}).')


    def __save_tfidf_comparisons(self):
        try:
            topic_words_df = pd.read_csv(
                f'./outputs/{self.dataset}/topic_words.csv',
                sep=';'
            )
        except:
            print(f'  ❌ Topic words not found.')
            return

        comparisons = []
        for _, row in topic_words_df.iterrows():
            tfidf_row = self.words_metrics[self.words_metrics.word == row['Word']].head(1)
            if tfidf_row.empty:
                continue
            comparisons.append({
                'word': row['Word'],
                'TFIDF': tfidf_row.iloc[0]['tfidf'],
                'C-TFIDF': row['C-TFIDF'],
                'diff': tfidf_row.iloc[0]['tfidf'] - row['C-TFIDF'],
            })
        pd.DataFrame(comparisons).sort_values(
            by='diff', ascending=True
        ).to_csv(
            f'./outputs/{self.dataset}/tf_idf_comparisons.csv',
            index=None
        )
        print('  ✅ TF-IDF comparisons saved.')


    def __filter_bertopic_words(self, df, clean=False):
        if clean:
            df = df[df.Word != '']
            df = df[~df.Word.isin(self.STOPWORDS)]

        return pd.concat([
            group[1].sort_values(
                'C-TFIDF',
                ascending=False
            ).head(10).reset_index(drop=True) for group in df.groupby(['Label', 'Topic'])
        ])


    def __save_graphics(self):
        self.__save_general_graphic(30, 'attention', 'Attention')
        self.__save_general_graphic(30, 'captum', 'Captum score')

        self.__save_per_class_graphic(20, 'attention', 'Attention')
        self.__save_per_class_graphic(20, 'captum', 'Captum score')
        print('  ✅ Graphics saved.')


    def __save_general_graphic(
        self,
        n_words,
        metric,
        metric_name
    ):
        proportion = sum([
            self.counts[f'{label}_correct'] for label in self.label_mapper
        ]) / self.counts['total']

        fig, axes= plt.subplots(1, 1, figsize=(10, 6))

        fig.subplots_adjust(wspace=0.35)
        axes.set_xlabel(metric_name)

        subset = self.words_metrics.sort_values(
            metric,
            ascending=False
        ).head(n_words)
        subset.plot(
            kind='barh',
            ax=axes,
            x='word',
            y=[f'{metric}_correct', f'{metric}_incorrect'],
            color=[self.COLORS['blue'], self.COLORS['red']],
            title=f'Top {n_words} IW based on {metric_name}',
            stacked=True,
            legend=None,
            xlabel='Word'
        )

        self.__add_threshold_lines(
            axes,
            metric,
            proportion,
            subset,
            n_words
        )

        handles, _ = axes.get_legend_handles_labels()
        axes.legend(
            handles[1:] + [handles[0]],
            [
                f'{metric_name} on well-predicted tweets',
                f'{metric_name} on mispredicted tweets',
                'Threshold for misprediction contribution'
            ]
        )

        fig.savefig(
            f'{self.output_folder}/graphics/general_{metric}.svg',
            format='svg'
        )

        plt.close(fig)


    def __save_per_class_graphic(
        self,
        n_words,
        metric,
        metric_name
    ):
        fig, axarr = plt.subplots(
            1,
            len(self.label_mapper),
            figsize=(14.4, 6.5)
        )

        for i, label in enumerate(self.label_mapper):
            axarr[i].set_xlabel(metric_name)

            subset = self.words_metrics.sort_values(
                f'{label}_{metric}',
                ascending=False
            ).head(n_words)

            proportion = (
                self.counts[f'{label}_correct'] / self.counts[label]
            )

            self.__add_threshold_lines(
                axarr[i],
                metric,
                proportion,
                subset,
                n_words,
                label
            )

            subset.plot(
                kind='barh',
                ax=axarr[i],
                x='word',
                y=[
                    f'{label}_{metric}_correct',
                    f'{label}_{metric}_incorrect'
                ],
                title=self.CLASSES_NAMES[label],
                color=[self.COLORS['blue'], self.COLORS['red']],
                stacked=True,
                legend=None,
                xlabel='Word'
            )

        fig.subplots_adjust(wspace=(len(self.label_mapper) * 0.3))
        fig.savefig(
            f'{self.output_folder}/graphics/groups_{metric}.svg', format='svg'
        )

        plt.close(fig)


    def __add_threshold_lines(
        self,
        axes,
        metric,
        proportion,
        subset,
        n_words,
        label=''
    ):
        bar = 0
        thresholds = 0
        for _, row in subset.iterrows():
            prefix = f'{label}_' if label != '' else ''
            if row[
                f'{prefix}{metric}_correct'
            ] < proportion * row[f'{prefix}{metric}']:
                if thresholds == 0:
                    axes.axvline(
                        proportion * row[f'{prefix}{metric}'],
                        bar / n_words,
                        (bar + 1) / n_words,
                        color='black',
                        linestyle='--',
                        label='aa'
                    )
                    thresholds += 1
                else:
                    axes.axvline(
                        proportion * row[f'{prefix}{metric}'],
                        bar / n_words,
                        (bar + 1) / n_words,
                        color='black',
                        linestyle='--'
                    )
            bar += 1


    def __save_important_words(self):
        proportion = sum([
            self.counts[f'{label}_correct'] for label in self.label_mapper
        ]) / self.counts['total']

        important_words = []
        for top in self.TOPS:
            top_dict = {
                'top': top,
                'average_attention': 0.0,
                'relevant': 0.0,
                'piws': 0.0,
                'in_vocabulary': 0.0,
                'w_captum': 0.0,
                'ploo': 0.0,
                'w_bertopic': 0.0,
            }

            iws = self.words_metrics.sort_values(
                'attention',
                ascending=False
            ).head(top)

            if top == 500:
                iws.to_csv(
                    f'./{self.output_folder}/words/top_{top}_words.csv',
                    sep=';',
                    index=None
                )

            top_dict['average_attention'] = np.mean(
                iws['attention'].values
            )

            top_dict['relevant'] = self.__get_intersection_size(
                iws,
                self.words_metrics.sort_values(
                    'tfidf',
                    ascending=False
                ).head(top),
                f'relevant_{top}'
            ) / top

            top_dict['piws'] = self.__get_intersection_size(
                iws,
                iws[
                    iws['attention_correct'] >= proportion * iws['attention']
                ],
                f'correct_{top}'
            ) / top

            words_present_in_vocab = 0
            for word in list(iws.word):
                if word in self.vocab_words:
                    words_present_in_vocab += 1
            top_dict['in_vocabulary'] = words_present_in_vocab / top

            captum = self.words_metrics.sort_values(
                'captum',
                ascending=False
            ).head(top)
            top_dict['w_captum'] = self.__get_intersection_size(
                iws,
                captum
            ) / top

            lwo_subset = self.lwo[self.lwo.word.isin(iws.word.values)]

            top_dict['ploo'] = len(lwo_subset[lwo_subset.f1 < 0]) / top

            if self.topic_words != None:
                top_dict['w_bertopic'] = len([
                    word for word in iws.word.values if word in self.clean_topic_words
                ]) / top

            important_words.append(top_dict)
        self.tops_df = pd.DataFrame(important_words)
        self.tops_df.to_csv(
            f'{self.output_folder}/words/general_percentages.csv',
            index=None
        )

        piw_df = []
        for top in self.TOPS:
            piw_dict = {
                'top': top,
                'average_attention': 0.0,
                'relevant': 0.0,
                'w_captum': 0.0,
                'ploo': 0.0,
                'ploo_size': 0.0,
                'w_bertopic': 0.0,
            }

            iws = self.words_metrics.sort_values(
                'attention',
                ascending=False
            )

            piws = iws[
                iws['attention_correct'] >= proportion * iws['attention']
            ].head(top)
            if len(piws) < top:
                warnings.warn('  ⚠⚠⚠⚠ The number of PIWs is lower than the cutoff:', len(piws))

            piw_dict['average_attention'] = np.mean(
                piws['attention'].values
            )

            piw_dict['relevant'] = self.__get_intersection_size(
                piws,
                self.words_metrics.sort_values(
                    'tfidf',
                    ascending=False
                ).head(top)
            ) / top

            piw_dict['w_captum'] = self.__get_intersection_size(
                piws,
                self.words_metrics.sort_values(
                    'captum',
                    ascending=False
                ).head(top)
            ) / top

            lwo_subset = self.lwo[self.lwo.word.isin(piws.word.values)]
            piw_dict['ploo_size'] = len(lwo_subset)
            piw_dict['ploo'] = len(lwo_subset[lwo_subset.f1 < 0]) / top

            if self.topic_words != None:
                piw_dict['w_bertopic'] = len([
                    word for word in piws.word.values if word in self.clean_topic_words
                ]) / top

            piw_df.append(piw_dict)
        self.piws_analysis_df = pd.DataFrame(piw_df)
        self.piws_analysis_df.to_csv(
            f'{self.output_folder}/words/piws_percentages.csv',
            index=None
        )

        ranking_scores = []
        for top in self.RANK_TOPS:
            ranking_dict = {
                'top': top,
                'average_attention': 0.0,
                'relevant': 0.0,
                'w_captum': 0.0,
                'piw_captum': 0.0,
                'piw_relevant': 0.0,
            }

            iws = self.words_metrics.sort_values(
                'attention',
                ascending=False
            ).head(top)

            ranking_dict['average_attention'] = np.mean(
                iws['attention'].values
            )

            ranking_dict['relevant'] = self.__get_ranking_score(
                iws,
                self.words_metrics.sort_values(
                    'tfidf',
                    ascending=False
                ).head(top),
                'tfidf'
            )

            ranking_dict['w_captum'] = self.__get_ranking_score(
                iws,
                self.words_metrics.sort_values(
                    'captum',
                    ascending=False
                ).head(top),
                'captum',
            )

            piws = self.words_metrics.sort_values(
                'attention',
                ascending=False
            )
            piws = piws[
                piws['attention_correct'] >= proportion * piws['attention']
            ].head(top)

            ranking_dict['piw_captum'] = self.__get_ranking_score(
                piws,
                self.words_metrics.sort_values(
                    'captum',
                    ascending=False
                ).head(top),
                'captum',
            )

            ranking_dict['piw_relevant'] = self.__get_ranking_score(
                piws,
                self.words_metrics.sort_values(
                    'tfidf',
                    ascending=False
                ).head(top),
                'tfidf',
            )

            ranking_scores.append(ranking_dict)

        self.rank_scores_df = pd.DataFrame(ranking_scores)
        self.rank_scores_df.to_csv(
            f'{self.output_folder}/words/rank_scores.csv',
            index=None
        )

        print(f' ✅ Attention important words.')


    def __bertopic_vs_tfidf(self):
        all_topic_words_len = len(self.topic_words)
        clean_topic_words_len = len(self.clean_topic_words)
        vs_dict = {
            'dataset': self.dataset,
            'model': self.model_name,
            'top': clean_topic_words_len,
            'relevant': 0.0,
            'w_bertopic': 0.0,
            'piws_bertopic': 0.0,
        }

        iws = self.words_metrics.sort_values(
            'attention',
            ascending=False
        ).head(all_topic_words_len)

        clean_iws = self.words_metrics.sort_values(
            'attention',
            ascending=False
        ).head(clean_topic_words_len)

        vs_dict['relevant'] = self.__get_intersection_size(
            clean_iws,
            self.words_metrics.sort_values(
                'tfidf',
                ascending=False
            ).head(clean_topic_words_len),
        ) / clean_topic_words_len

        if self.topic_words != None:
            vs_dict['w_bertopic'] = len([
                word for word in clean_iws.word.values if word in self.clean_topic_words
            ]) / clean_topic_words_len

            piws = self.words_metrics.sort_values(
                'attention',
                ascending=False
            )
            proportion = sum([
                self.counts[f'{label}_correct'] for label in self.label_mapper
            ]) / self.counts['total']
            piws = piws[
                piws['attention_correct'] >= proportion * piws['attention']
            ].head(clean_topic_words_len)
            piws = set(piws.word.values)
            vs_dict['piws_bertopic'] = len([
                word for word in self.clean_topic_words if word in piws
            ]) / clean_topic_words_len

        self.bertopic_tfidf.append(vs_dict)

        print(f' ✅ TF-IDF vs BERTopic results saved.')


    def __get_intersection_size(self, dfA, dfB, savename=None):
        intersection = set(dfA.word).intersection(set(dfB.word))
        if savename is not None:
            pd.DataFrame(
                {'word': sorted(list(intersection))}
            ).to_csv(
                f'{self.output_folder}/words/{savename}.csv',
                index=None
            )
        return len(intersection)


    def __get_ranking_score(self, got_df, expected_df, metric):
        got_mapper = {}
        for _, row in got_df.iterrows():
            got_mapper[row['word']] = row['attention']

        expected = []
        got = []
        for _, row in expected_df.iterrows():
            expected.append(row[metric])
            got.append(
                got_mapper[row['word']] if row['word'] in got_mapper else 0
            )
        return ndcg_score([expected], [got])


    def __save_correlation_tests_results(self):
        print('  Correlation tests:')
        self.__save_correlation_tests_results_per_df(
            self.tops_df,
            'general',
            'Attention'
        )
        self.__save_correlation_tests_results_per_df(
            self.piws_analysis_df,
            'piws',
            'PIWs'
        )


    def __save_correlation_tests_results_per_df(self, df, correlation_key, name):
        relevant_metrics = [
            'relevant',
            'w_captum',
        ]
        if correlation_key == 'general':
            relevant_metrics = [
                'relevant',
                'piws',
                'in_vocabulary',
                'w_captum',
                'ploo',
            ]
        if correlation_key == 'piws':
            relevant_metrics = [
                'relevant',
                'w_captum',
                'ploo',
            ]

        tests_results = []
        for metric in relevant_metrics:
            if correlation_key == 'piws' and metric == 'ploo':
                att = []
                metric_values = []
                for _, row in df.iterrows():
                    if row.top != row.ploo_size:
                        print('  Stopped at', row.top, row.ploo_size)
                        break
                    att.append(row.average_attention)
                    metric_values.append(row.ploo)
                tests_results.append({
                    'metricA': 'average_attention',
                    'metricB': metric,
                    'spearman': '%.4f (p=%.4f)' % (
                        stats.spearmanr(
                            att, metric_values
                        )
                    )
                })
                continue
            tests_results.append({
                'metricA': 'average_attention',
                'metricB': metric,
                'spearman': '%.4f (p=%.4f)' % (
                    stats.spearmanr(
                        df['average_attention'].values, df[metric].values
                    )
                )
            })
        if correlation_key == 'general':
            tests_results.append({
                'metricA': 'piws',
                'metricB': 'in_vocabulary',
                'spearman': '%.4f (p=%.4f)' % (
                    stats.spearmanr(
                        df['piws'].values, df['in_vocabulary'].values
                    )
                )
            })

        self.correlation_tests_results[correlation_key] = tests_results
        print(f'   ✅ {name}.')


    def __save_rank_correlation_tests_results(self):
        print('  Rank correlation tests:')
        self.__save_rank_correlation_tests_results_per_df(
            self.rank_scores_df,
            'general',
            'Attention'
        )


    def __save_rank_correlation_tests_results_per_df(self, df, correlation_key, name):
        relevant_metrics = [
            'relevant',
            'w_captum',
        ]

        tests_results = []
        for metric in relevant_metrics:
            tests_results.append({
                'metricA': 'average_attention',
                'metricB': metric,
                'spearman': '%.4f (p=%.4f)' % (
                    stats.spearmanr(
                        df['average_attention'].values, df[metric].values
                    )
                )
            })

        self.rank_correlation_tests_results = tests_results
        print(f'   ✅ {name}.')


    def __add_global_averages(self):
        print(' Global averages:')
        print(f"  Global BERTopic range: (raw: {self.TOPS[self.topic_words_sizes]}, clean: {self.TOPS[self.clean_topic_words_sizes]})")
        self.global_averages.append({
            'dataset': self.dataset,
            'model': self.model_name,
            'AvgAtt': np.mean(self.tops_df['average_attention'].values),
            'Relevant': np.mean(self.tops_df['relevant'].values),
            'PIW': np.mean(self.tops_df['piws'].values),
            'InVocab': np.mean(self.tops_df['in_vocabulary'].values),
            'W-Captum': np.mean(self.tops_df['w_captum'].values),
            'PLOO': np.mean(self.tops_df['ploo'].values),
            'W-BERTopic': np.mean(
                self.tops_df['w_bertopic'].values[:self.clean_topic_words_sizes]
            ),
        })


    def __add_rank_scores_averages(self):
        self.rank_scores.append({
            'dataset': self.dataset,
            'model': self.model_name,
            'AvgAtt': np.mean(self.rank_scores_df['average_attention'].values),
            'Relevant': np.mean(self.rank_scores_df['relevant'].values),
            'W-Captum': np.mean(self.rank_scores_df['w_captum'].values),
            'PIW-Captum': np.mean(self.rank_scores_df['piw_captum'].values),
            'PIW-Relevant': np.mean(self.rank_scores_df['piw_relevant'].values),
        })


    def __add_statistical_tests_results(self):
        for key in ['general', 'piws']:
            row = {
                'model': self.model_name,
                'dataset': self.dataset,
            }
            for c_test in self.correlation_tests_results[key]:
                row[f"{c_test['metricA']}_vs_{c_test['metricB']}"] = c_test['spearman']
            self.statistical_tests_results[key].append(row)

            if key == 'piws':
                continue
            row = {
                'model': self.model_name,
                'dataset': self.dataset,
            }
            for c_test in self.rank_correlation_tests_results:
                row[f"{c_test['metricA']}_vs_{c_test['metricB']}"] = c_test['spearman']
            self.rank_statistical_tests_results.append(row)


    def __save_averages(self):
        pd.DataFrame(
            self.global_averages
        ).to_csv(
            f'./outputs/global_averages.csv',
            index=None
        )
        pd.DataFrame(
            self.rank_scores
        ).to_csv(
            f'./outputs/rank_scores_averages.csv',
            index=None
        )
        print('Averages saved 📊')


    def __save_statistical_tests_results(self):
        pd.DataFrame(
            self.statistical_tests_results['general']
        ).to_csv(
            f'./outputs/attention_statistical_tests.csv',
            index=None
        )
        pd.DataFrame(
            self.statistical_tests_results['piws']
        ).to_csv(
            f'./outputs/piws_correlation_tests.csv',
            index=None
        )
        pd.DataFrame(
            self.rank_statistical_tests_results
        ).to_csv(
            f'./outputs/rank_correlation_tests.csv',
            index=None
        )
        print('Statistical results saved 📊')


    def __save_bertopic_vs_tfidf(self):
        pd.DataFrame(
            self.bertopic_tfidf
        ).to_csv(
            f'./outputs/bertopic_vs_tfidf.csv',
            index=None
        )
        print('BERTopic vs TF-IDF results saved 📊')


wa = WordsAnalyzer()
wa()
