# All imports
import pandas as pd
from surprise import Dataset, Reader
from itertools import chain
import numpy as np
from surprise import SVD
from surprise.model_selection import KFold
from tqdm.auto import tqdm

def load_data():
    global products_data, transactions_data, customer_data
    # Load item data
    products_data = pd.read_csv('/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/articles.csv')
    # Load transaction data
    transactions_data = pd.read_csv('/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
    # Load customer data
    customer_data = pd.read_csv('/Users/miguel/Documents/h-and-m-personalized-fashion-recommendations/customers.csv')

def preprocess_data():
    global merge_data, outfit_data, unrolled_data, transaction_freq, data
    # Preprocess data
    merge_data = pd.merge(transactions_data, products_data, on='article_id', how='left')
    merge_data = pd.merge(merge_data, customer_data, on='customer_id', how='left')
    outfit_data = transactions_data.groupby(['customer_id', 't_dat'])['article_id'].apply(list).reset_index()
    unrolled_data = pd.DataFrame({
        'customer_id': np.repeat(outfit_data['customer_id'], outfit_data['article_id'].apply(len)),
        'article_id': list(chain.from_iterable(outfit_data['article_id']))
    })
    # Frequency of transactions for each item for each user
    transaction_freq = unrolled_data.groupby(['customer_id', 'article_id']).size().reset_index(name='freq')
    # Load dataframe into surprise database
    reader = Reader(rating_scale=(transaction_freq.freq.min(), transaction_freq.freq.max()))
    data = Dataset.load_from_df(transaction_freq[['customer_id', 'article_id', 'freq']], reader)
def train_model():
    global model, splits
    model = SVD()
    #split dataset into 5 fold and select the first for learning rest for testing
    kf = KFold(n_splits=5)
    splits = list(kf.split(data))

    for trainset, testset in tqdm(splits, desc='Training', leave=False):
        # Train and test algorithm
        model.fit(trainset)
        model.test(testset)

if __name__ == '__main__':
    major_steps = [('Load data', load_data), ('Preprocess data', preprocess_data), ('Train model', train_model)]

    with tqdm(major_steps, desc='Overall Progress') as major_steps_bar:
        for desc, step in major_steps_bar:
            major_steps_bar.set_description(desc)
            step()
            major_steps_bar.refresh()


def recommend(merge_data, product_type, product_group):
    product_transactions = merge_data[(merge_data['product_type_name'] == product_type) & (merge_data['product_group_name'] == product_group)]
    associated_transactions = merge_data[merge_data['t_dat'].isin(product_transactions['t_dat'])]
    associated_products = associated_transactions[associated_transactions['product_group_name'] != product_group]
    recommended_product = associated_products['product_type_name'].value_counts().idxmax()
    return recommended_product
    print(recommended_product)

