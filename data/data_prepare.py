"""
Split a balanced validation set and test set for each country.
Generate data to make a balanced training set.
Update and save the city list.
"""
import os
import json
import random
import pandas as pd

from preset import code2country


def jsonl2df(jsonl_path):
    """
    Convert a jsonl file to a pandas dataframe.
    """
    with open(jsonl_path) as rf:
        data = [json.loads(line) for line in rf]

    return pd.DataFrame(data)


def generate_data(country_code, nums):
    """
    Generate addresses from openaddr-collected-europe
    """
    country = code2country[country_code.upper()]

    # Load data
    dfs = []
    for root, dirs, files in os.walk(f'openaddr-collected-europe/{country_code.lower()}'):
        for file in files:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(root, file), usecols=['NUMBER', 'STREET', 'CITY', 'DISTRICT', 'REGION', 'POSTCODE'], dtype=str)
                # City must be in the address
                dfs.append(df.dropna(subset=['CITY']))
    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs)

    # Sample data
    print(f"Genrating {min(len(df),nums)} addresses from {country}")
    df_sample = df.sample(n=min(len(df),nums)).reset_index(drop=True)

    # Generate data
    data = []
    for i in range(min(len(df), nums)):
        item = df_sample.iloc[i]
        item = item.dropna()
        indices = item.index
        address = []
        # NUMBERT STREET or STREET NUMBER
        if 'NUMBER' in indices and 'STREET' in indices:
            if random.random() > 0.5:
                address.append(item['NUMBER']+' '+item['STREET'])
            else:
                address.append(item['STREET']+' '+item['NUMBER'])
        elif 'STREET' in indices:
            address.append(item['STREET'])

        # CITY in the middle
        address.append(item['CITY'])

        # Randomly add DISTRICT, REGION
        if 'DISTRICT' in indices and random.random() > 0.5:
            address.append(item['DISTRICT'])
        if 'REGION' in indices and random.random() > 0.5:
            address.append(item['REGION'])
        # POSTCODE can be at the end or followed by the street
        if 'POSTCODE' in indices and random.random() > 0.5:
            if random.random() > 0.5 and 'STREET' in indices:
                address.insert(1, item['POSTCODE'])
            else:
                address.append(item['POSTCODE'])
        # Randomly add country
        if random.random() > 0.8:
            address.append(country)
        # Join address
        address = ', '.join(address)
        # Replace commas with spaces
        if random.random() > 0.8:
            address = address.replace(', ', ' ')
        # Add to data
        data.append({
            'country': country_code,
            'address': address
        })

    return pd.DataFrame(data), df_sample[['CITY']]


if __name__ == '__main__':
    # Load data
    df_addr = jsonl2df('dataset/addresses.jsonl')
    df_cities = jsonl2df('dataset/cities.jsonl').rename(columns={'city': 'CITY'})
    df_addr_group = df_addr.groupby('country')
    df_cities_group = df_cities.groupby('country')
    # Split a balanced dataset for each country
    num_train, num_eval = 50000, 2000
    for country, group in df_addr_group:
        print(f"{country} has {len(group)} addresses")
        df_city = df_cities_group.get_group(country)[['CITY']]
        if len(group) > num_train:
            # For those countries with more than 'num_train' addresses, sample part of it
            df_train = group.sample(n=num_train)
            df_eval = group.drop(df_train.index).sample(n=num_eval)
        else:
            # For those countries with less than 'num_train' addresses
            if len(group) < num_eval:
                # If the number of addresses is less than 'num_eval', use all for evaluation,
                # and generate 'num_train' more addresses for training
                df_eval = group
                df_train, df_city_ex = generate_data(country, num_train)
            else:
                # If the number of addresses is greater than 'num_eval', sample 'num_eval' for evaluation,
                # and generate more addresses to fill the rest of the training set
                df_eval = group.sample(n=num_eval)
                df_train = group.drop(df_eval.index)
                df_gen, df_city_ex = generate_data(country, num_train - len(df_train))
                df_train = pd.concat([df_train, df_gen])

            # Merge city list
            df_city = pd.concat([df_city, df_city_ex])

        # Drop duplicates and save
        df_city = df_city.drop_duplicates()
        df_city.to_csv(f'cities/{country}.csv', index=False, sep='\t')

        # Split into validation and test set
        df_val = df_eval.sample(n=int(len(df_eval) / 2))
        df_test = df_eval.drop(df_val.index)

        # Save the train, validation and test set
        df_train.to_csv(f'train/{country}.csv', index=False, sep='\t')
        df_val.to_csv(f'val/{country}.csv', index=False, sep='\t')
        df_test.to_csv(f'test/{country}.csv', index=False, sep='\t')




