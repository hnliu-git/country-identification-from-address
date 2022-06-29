# ISO 3166 country codes

code2country = {
    'FR': 'France',
    'DE': 'Germany',
    'NL': 'Netherlands',
    'BE': 'Belgium',
    'PL': 'Poland',
    'PT': 'Portugal',
    'IT': 'Italy',
    'CZ': 'Czechia',
    'EE': 'Estonia',
    'SK': 'Slovakia',
    'SI': 'Slovenia',
    'SE': 'Sweden',
    'AT': 'Austria',
    'LU': 'Luxembourg',
    'RO': 'Romania'
}

country2code = {v: k for k, v in code2country.items()}

code2id = {code: i for i, code in enumerate(sorted(code2country.keys()))}
code2id['UN'] = len(code2country)