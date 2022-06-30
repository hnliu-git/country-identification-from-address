"""
Offline evaluation of the model.
"""

import pandas as pd
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
        from preset import code2id

        import seaborn as sns
        import matplotlib.pyplot as plt

        y_preds = []
        y_trues = []
        code2id['UN'] = len(code2id)

        for i in tqdm(range(len(self.df_test))):
            address = self.df_test.iloc[i]['address']
            country = self.df_test.iloc[i]['country']

            pred_dict = model(address)
            if sum(pred_dict.values()) == 0:
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


if __name__ == '__main__':

    from models.rule_model import *
    from models.stat_model import *
    from preset import code2country

    class CombModel:
        def __init__(self, model1, model2):
            self.model1 = model1
            self.model2 = model2

        def __call__(self, address):
            pred_dict1 = self.model1(address)
            pred_dict2 = self.model2(address)
            pred_dict = {}
            for key in pred_dict1:
                pred_dict[key] = pred_dict1[key] + pred_dict2[key]
            return pred_dict

    s_model_path = 'ckpts/epoch04-val_loss0.92-coun14/'
    r_model = RuleModel(code2country.keys())
    s_model = StatModel(s_model_path, code2country.keys())
    comb_model = CombModel(r_model, s_model)

    evaluator = Evaluator(code2country.keys())

    # Evaluate metrics
    evaluator.evaluate_metrics(r_model)

    # Evaluate latency
    evaluator.evaluate_latency(r_model, n=100)

    # Evaluate metrics
    evaluator.evaluate_metrics(s_model)

    # Evaluate latency
    evaluator.evaluate_latency(s_model, n=100)

    # Evaluate metrics
    evaluator.evaluate_metrics(comb_model)

    # Evaluate latency
    evaluator.evaluate_latency(comb_model, n=100)

