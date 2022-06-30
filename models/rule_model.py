"""
A rule-based model for country identification given address.
"""


import re
from preset import code2country, country2code


class RuleModel:
    """
        The rule-based model for country identification
    """

    def __init__(self, country_codes):
        '''
        :param country_codes: list[str] - list of country codes that are expected to be supported
        '''
        city2codes = {}
        # A city can be in multiple countries
        for country_code in country_codes:
            country_code = country_code.upper()
            with open(f'data/cities/{country_code}.csv') as f:
                for line in f:
                    line = line.strip()
                    if line not in city2codes:
                        city2codes[line.lower()] = {country_code}
                    else:
                        city2codes[line.lower()].add(country_code)
            city2codes.pop('city')

        self.country_codes = country_codes
        self.city2codes = city2codes
        self.country_set = set(code2country[code].lower() for code in country_codes)

    def __call__(self, address):
        """
        Rule 1: If the last block is a country, return the country code
        Rule 2: Iterate through the address blocks and find a complete match with part of the city name,
                there can be multiple matches, so calculate confidence score for each match

        :param address: str - the address to be identified
        :return: dict(code, score) - dict of country code with confidence
        """
        # Preprocess address and
        address = address.lower()
        address = address.replace(' - ', ',')

        # Split into blocks
        if ',' in address:
            blocks = address.split(',')
        else:
            blocks = address.split()

        # Rule 1, a shortcut
        potential_country = blocks[-1].strip()
        if potential_country in self.country_set:
            country = potential_country[0].upper() + potential_country[1:]
            return {country2code[country]: 1.0}

        # Rule 2
        code_cot = {}
        for block in blocks:
            block = block.strip()
            # Skip if the block is empty
            if block == "": continue
            # Skip if only contains numbers
            if re.match(r'^\d+$', block): continue
            # Iterate through all cities
            for city in self.city2codes.keys():
                if block in city:
                    # Name of the city can have multiple words
                    for city_block in city.split():
                        if block == city_block:
                            codes = self.city2codes[city]
                            for code in codes:
                                code_cot[code] = code_cot.get(code, 0) + 1

        if len(code_cot) == 0:
            # No match found
            return {code.upper(): 0.0 for code in self.country_codes}
        else:
            # Calculate confidence score
            score_sum = sum(code_cot.values())
            res = {}
            for code in self.country_codes:
                code = code.upper()
                if code in code_cot:
                    res[code] = code_cot[code] / score_sum
                else:
                    res[code] = 0.0

            return res
