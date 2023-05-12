import pandas as pd
import gzip
import pickle
import os
import json


# This class is designed to extract the data out of Julian McAuley dataset
class DataProvider:
    def __init__(self, category):
        self.category = category

    def get_all_data_points(self):
        df = DataProvider.get_df(self.data_path)
        return df

    @staticmethod
    def parse_from_gzip(path):
        # for memory issues
        max_extract = 10000000
        count = 0
        print(path)
        g = gzip.open(path, 'rb')
        for l in g:
            if count < max_extract:
                yield json.loads(l)
            else:
                return
            count += 1

    @staticmethod
    def get_df(path):
        i = 0
        df = {}
        for d in DataProvider.parse_from_gzip(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')



class ReviewDataProvider(DataProvider):
    def __init__(self, data_dir, category):
        super().__init__(category)
        self.data_dir = data_dir
        self.data_path = os.path.join(data_dir, "{}_5.json.gz".format(self.category))
        self.reviews_path = os.path.join(data_dir, "X_{}_5.pkl".format(self.category))
        self.labels_path = os.path.join(data_dir, "y_{}_5.pkl".format(self.category))


    def balance_dataset(self, df, max_length):
        neutral = 3
        df_pos = df[df["overall"] > neutral]
        df_neg = df[df["overall"] < neutral]
        pos_reviews = df_pos["reviewText"].tolist()
        neg_reviews = df_neg["reviewText"].tolist()
        min_length = min(len(pos_reviews), len(neg_reviews))
        if min_length < max_length:
            print("Warning, you asked for {} examples, but there are only {} for {}".format(max_length,
                                                                                            min_length, self.category))
        if min_length > max_length:
            min_length = max_length
        # keeps the dataset balanced
        pos_reviews = pos_reviews[:min_length]
        neg_reviews = neg_reviews[:min_length]
        reviews = neg_reviews + pos_reviews
        labels = [0] * min_length + [1] * min_length
        return reviews, labels
    
    def create_json(self, reviews, labels, json_filename):
        json_path = os.path.join(self.data_dir, json_filename)
        assert len(reviews) == len(labels)
        with open(json_path, 'w', encoding='utf-8') as f:
            for i in range(len(reviews)):
                data = {'text':str(reviews[i]).replace('\n', ' '), 'label':labels[i]}
                f.write(json.dumps(data) + '\n')
    
    def create_txt(self, reviews, labels, review_filename, label_filename):
        txt_path = os.path.join(self.data_dir, review_filename)
        label_path = os.path.join(self.data_dir, label_filename)
        assert len(reviews) == len(labels)
        with open(txt_path, 'w', encoding='utf-8') as f,\
            open(label_path, 'w', encoding='utf-8') as f_label:
            for i in range(len(reviews)):
                f.write(str(reviews[i]).replace('\n', ' ') + '\n')
                f_label.write(str(labels[i]) + '\n')


    def construct_dataset(self, sample_size):
        reviews_df = self.get_all_data_points()
        reviews, labels = self.balance_dataset(reviews_df, sample_size)
        with open(self.reviews_path, 'wb') as f:
            pickle.dump(reviews, f)
        with open(self.labels_path, 'wb') as f:
            pickle.dump(labels, f)
        return reviews, labels