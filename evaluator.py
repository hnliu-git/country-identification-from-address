"""
Evaluation on the test set and latency
"""

import pandas as pd
from preset import code2id
from tqdm import tqdm


class Evaluator:

    def __init__(self, countries):
        df_test = pd.DataFrame()
        for country in countries:
            df_test = pd.concat([df_test, pd.read_csv(f'data/test/{country}.csv', sep='\t')])

        self.df_test = df_test

    def evaluate_metrics(self, model):
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix

        import seaborn as sns
        import matplotlib.pyplot as plt

        y_preds = []
        y_trues = []
        for i in tqdm(range(len(self.df_test))):
            address = self.df_test.iloc[i]['address']
            country = self.df_test.iloc[i]['country']

            pred_dict = model(address)
            if len(pred_dict) == 0:
                # no prediction
                y_pred = 'UN'
            else:
                y_pred = max(pred_dict, key=pred_dict.get)

            y_preds.append(code2id[y_pred])
            y_trues.append(code2id[country])

        print(f'Accuracy: {accuracy_score(y_trues, y_preds):2f}')
        labels = list(code2id.keys())
        sns.heatmap(confusion_matrix(y_trues, y_preds),
                    annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.show()

    def evaluate_latency(self, model, n=100):
        # warmup
        for i in range(100):
            model(self.df_test.iloc[i]['address'])

        # evaluate
        import time
        sample = self.df_test.sample(n=n)
        st = time.time()
        for i in range(len(sample)):
            model(self.df_test.iloc[i]['address'])
        print(f'Average latency per item: {((time.time() - st) / n)*1000:2f}ms')

