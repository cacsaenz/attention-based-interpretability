import emoji
import pandas as pd
import re
import regex
import unidecode
import json
from sklearn.model_selection import train_test_split

class DatasetPreprocessor:
    HASHTAGS = {
        'SI_DS': [
            '#OBrasilTemQuePararBolsonaro',
            '#OBrasilNãoPodeParar',
            '#OBrasilNaoPodeParar', # without ã
            '#ficaemcasa',
            '#fiqueemcasa',
        ],
        'H_DS': [],
    }

    LOWERCASE = True

    def __removeNumbers(self, tweet):
        return " ".join(
            re.sub(r'[0-9]', '', tweet).split()
        )


    def __removeEmojis(self, tweet):
        return " ".join(
            emoji.get_emoji_regexp().sub(r'', tweet).split()
        )


    def __removeURLs(self, tweet):
        return " ".join(
            re.sub(r"http\S+|youtu.be\S+|\S+.com.br\S+|bit.ly\S+|\S+.com/\S+|\S+.co/\S+|\S+.org\S+|\S+.br/\S+|\S+.es/S+", "", tweet).split()
        )


    def __removeHashtags(self, tweet, dataset):
        for hashtag in self.HASHTAGS[dataset]:
            # Here, it could be two variations:
            # the hashtag symbol (#) + the HT
            # the HT only
            ht = re.compile(
                re.escape(
                    hashtag
                ),
                re.IGNORECASE
            )
            tweet = ht.sub('', tweet)

            ht = re.compile(
                re.escape(
                    hashtag[1:]
                ),
                re.IGNORECASE
            )
            tweet = ht.sub('', tweet)

        # Finally, replace all empty hashtags symbols (#)
        return " ".join(
            re.sub(r'# ', '', tweet).split()
        )


    def __removeMentions(self, tweet):
        return " ".join(
            re.sub(r"@\S+", "", tweet).split()
        )


    def __isValidTweet(self, tweet):
        without_mentions = " ".join(re.sub(r"@", "", tweet).split())
        if (
            len(without_mentions.split()) < 3 or
            not without_mentions or
            re.search("^\s*$", without_mentions)
        ):
            return False
        return True


    def __call__(
        self,
        tweet,
        hashtagsKey
    ):
        if self.LOWERCASE:
            tweet = tweet.lower()

        preprocessed = self.__removeNumbers(
            self.__removeEmojis(
                self.__removeMentions(
                    self.__removeURLs(
                        self.__removeHashtags(
                            tweet,
                            hashtagsKey
                        )
                    )
                )
            )
        )
        if self.__isValidTweet(preprocessed) and preprocessed:
            return preprocessed
        return None


class DatasetBuilder:
    PREFIX = './datasets'
    RANDOM_SEED = 42
    DATASET_SIZE = 6000

    DP = DatasetPreprocessor()

    def __build_V_DS_dataset(self):
        print('  💉 V_DS dataset...')
        labels = []
        tweets = []
        for label in self.config['V_DS']['mapper'].keys():
            group_df = pd.read_csv(
                f"{self.PREFIX}/{self.config['V_DS']['folder']}/raw/{label}.csv",
                names=['tweet']
            )
            for tweet in group_df['tweet'].values:
                try:
                    preprocessed_tweet = self.DP(
                        tweet,
                        hashtagsKey='V_DS'
                    )
                except:
                    print(f'      Failed with {tweet}')
                    continue
                if preprocessed_tweet:
                    tweets.append(preprocessed_tweet)
                    labels.append(
                        self.config['V_DS']['mapper'][label]
                    )
            print(f'     ✔ {label}')
        self.datasets['V_DS'] = pd.DataFrame({
            'tweet': tweets,
            'label': labels,
        })
        print('')


    def __build_SI_DS_dataset(self):
        print('  🏠 SI_DS dataset...')
        labels = []
        tweets = []
        ids = []
        for label in self.config['SI_DS']['mapper'].keys():
            group_df = pd.read_csv(
                f"{self.PREFIX}/{self.config['SI_DS']['folder']}/raw/{label}.csv",
                sep=';'
            )[['text', 'id']]
            for _, row in group_df.iterrows():
                try:
                    preprocessed_tweet = self.DP(
                        row['text'],
                        hashtagsKey='SI_DS'
                    )
                except:
                    print(f"       Failed with {row['text']}")
                    continue
                if preprocessed_tweet:
                    tweets.append(preprocessed_tweet)
                    labels.append(
                        self.config['SI_DS']['mapper'][label]
                    )
                    ids.append(row['id'])
            print(f'     ✔ {label}')
        self.datasets['SI_DS'] = pd.DataFrame({
            'id': ids,
            'tweet': tweets,
            'label': labels,
        })


    def __build_SI_DS_2_dataset(self):
        print('  🏠 Social isolation dataset 2...')
        labels = []
        tweets = []
        ids = []
        for label in self.config['SI_DS2']['mapper'].keys():
            group_df = pd.read_csv(
                f"{self.PREFIX}/{self.config['SI_DS2']['folder']}/raw/{label}.csv",
                sep=';'
            )[['text', 'id']]
            for _, row in group_df.iterrows():
                try:
                    preprocessed_tweet = self.DP(
                        row['text'],
                        hashtagsKey='SI_DS'
                    )
                except:
                    print(f"       Failed with {row['text']}")
                    continue
                if preprocessed_tweet:
                    tweets.append(preprocessed_tweet)
                    labels.append(
                        self.config['SI_DS2']['mapper'][label]
                    )
                    ids.append(row['id'])
            print(f'     ✔ {label}')
        self.datasets['SI_DS2'] = pd.DataFrame({
            'id': ids,
            'tweet': tweets,
            'label': labels,
        })


    def __build_H_DS_dataset(self):
        print('  💊 H_DS dataset...')
        labels = []
        tweets = []
        ids = []
        for label in self.config['H_DS']['mapper'].keys():
            group_df = pd.read_csv(
                f"{self.PREFIX}/{self.config['H_DS']['folder']}/raw/{label}.csv"
            )[['text', 'id']]
            for _, row in group_df.iterrows():
                try:
                    preprocessed_tweet = self.DP(
                        row['text'],
                        hashtagsKey='H_DS'
                    )
                except:
                    print(f"       Failed with {row['text']}")
                    continue
                if preprocessed_tweet:
                    tweets.append(preprocessed_tweet)
                    labels.append(
                        self.config['H_DS']['mapper'][label]
                    )
                    ids.append(row['id'])
            print(f'     ✔ {label}')
        self.datasets['H_DS'] = pd.DataFrame({
            'id': ids,
            'tweet': tweets,
            'label': labels,
        })


    def __build_datasets(self):
        print('⌛ Building datasets...')
        # self.__build_V_DS_dataset()
        self.__build_SI_DS_dataset()
        self.__build_SI_DS_2_dataset()
        self.__build_H_DS_dataset()
        print(' ✅ Done.')


    def __load_config(self):
        self.datasets = {}
        self.samples = {}
        self.config = json.load(
            open(f'{self.PREFIX}/config.json')
        )
        print('📄 Config loaded.')


    def __save_datasets(self):
        print('💾 Saving complete datasets...')
        metrics = {}
        for key, value in self.config.items():
            if key not in self.datasets:
                print(f'  - {key} not found.')
                continue

            samples = []
            dataset = self.datasets[key]
            dataset.to_csv(
                f'{self.PREFIX}/{key}/complete_dataset.csv',
                sep=';',
                index=False
            )

            metrics[key] = {
                'complete': {
                    'total': len(dataset),
                },
                'sample': {},
            }
            for label in value['mapper'].keys():
                subset = dataset[dataset['label'] == value['mapper'][label]]
                metrics[key]['complete'][label] = len(subset)

                size = self.DATASET_SIZE
                print(key, size)
                samples.append(
                    subset.sample(
                        random_state=self.RANDOM_SEED,
                        n=int(size/len(value['mapper'].keys()))
                    ).reset_index(drop=True)
                )
            sample = pd.concat(samples).reset_index(drop=True)
            self.samples[key] = sample

            metrics[key]['sample']['total'] = len(sample)
            for label in value['mapper'].keys():
                subset = sample[sample['label'] == value['mapper'][label]]
                metrics[key]['sample'][label] = len(subset)

        with open(f'{self.PREFIX}/metrics.json', 'w+') as fp:
            json.dump(metrics, fp)

        print(' ✅ Done.')


    def __save_samples(self):
        print('💾 Saving samples...')

        for key in self.config.keys():
            if key not in self.samples:
                print(f'  - {key} not found.')
                continue

            print(f' - {key}:')
            sample = self.samples[key]

            train_df, test_df = train_test_split(
                sample,
                test_size=0.2,
                random_state=self.RANDOM_SEED,
                stratify=sample[['label']]
            )

            train_df, val_df = train_test_split(
                train_df,
                test_size=0.1,
                random_state=self.RANDOM_SEED,
                stratify=train_df[['label']]
            )

            print('   Training sentences: {:,}'.format(train_df.shape[0]))
            print('     # 0:', len(train_df[train_df['label'] == 0]))
            print('     # 1:', len(train_df[train_df['label'] == 1]))
            print('     # 2:', len(train_df[train_df['label'] == 2]))
            print('')

            print('   Validation sentences: {:,}'.format(val_df.shape[0]))
            print('     # 0:', len(val_df[val_df['label'] == 0]))
            print('     # 1:', len(val_df[val_df['label'] == 1]))
            print('     # 2:', len(val_df[val_df['label'] == 2]))
            print('')

            print('   Testing sentences: {:,}'.format(test_df.shape[0]))
            print('     # 0:', len(test_df[test_df['label'] == 0]))
            print('     # 1:', len(test_df[test_df['label'] == 1]))
            print('     # 2:', len(test_df[test_df['label'] == 2]))

            print('\n')

            # Save complete sample
            sample.to_csv(
                f'{self.PREFIX}/{key}/sample.csv',
                index=None
            )

            # Save train/val/test dataframes
            train_df.to_csv(
                f'{self.PREFIX}/{key}/train_dataset.csv',
                index=None
            )
            val_df.to_csv(
                f'{self.PREFIX}/{key}/val_dataset.csv',
                index=None
            )
            test_df.to_csv(
                f'{self.PREFIX}/{key}/test_dataset.csv',
                index=None
            )
        print(' ✅ Done.')


    def __call__(self):
        self.__load_config()
        self.__build_datasets()
        self.__save_datasets()
        self.__save_samples()


db = DatasetBuilder()
db()