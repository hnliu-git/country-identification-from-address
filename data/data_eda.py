"""
Some basic EDA
"""


import json
import pandas as pd
import matplotlib.pyplot as plt


def jsonl2df(jsonl_path):
    """
    Convert a jsonl file to a pandas dataframe.
    """
    with open(jsonl_path) as rf:
        data = [json.loads(line) for line in rf]

    return pd.DataFrame(data)


if __name__ == '__main__':
    df = jsonl2df('dataset/cities.jsonl')

    label_dict = {}
    adds_len = {}
    # Length of addressses and labels distribution
    with open('dataset/addresses.jsonl') as rf:
        for line in rf:
            country = json.loads(line)['country']
            address = len(json.loads(line)['address'])
            if address not in adds_len:
                adds_len[address] = 1
            else:
                adds_len[address] += 1
            if country not in label_dict:
                label_dict[country] = 0
            else:
                label_dict[country] += 1

    label_dict = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)
    plt.bar([x[0] for x in label_dict], [x[1] for x in label_dict])
    plt.show()

    plt.bar(list(adds_len.keys()), list(adds_len.values()))
    plt.xlim(0, 80)
    plt.show()

    label_dict = {k: 0 for k, _ in label_dict}

    for code in label_dict.keys():
        label_dict[code] += len(pd.read_csv(f'train/{code}.csv', sep='\t'))
        label_dict[code] += len(pd.read_csv(f'val/{code}.csv', sep='\t'))
        label_dict[code] += len(pd.read_csv(f'test/{code}.csv', sep='\t'))

    label_dict = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)
    plt.bar([x[0] for x in label_dict], [x[1] for x in label_dict])
    plt.show()
