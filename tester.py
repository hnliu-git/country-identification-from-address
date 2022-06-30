"""
A simple tester testing the functionality of the service.
"""


import time
import unittest
import requests


class ServiceTest(unittest.TestCase):
    url = 'http://127.0.0.1:8080'
    address = 'KRISTJAN KÄRBERI TÄNAV, TALLINN, LINNAOSA'

    def test_empty_request(self):
        res = requests.post(f'{self.url}/comb/', json={'address': ''}).json()
        self.assertEqual(res, {'error': 'address is empty'})
        res = requests.post(f'{self.url}/rule/', json={'address': ''}).json()
        self.assertEqual(res, {'error': 'address is empty'})
        res = requests.post(f'{self.url}/stat/', json={'address': ''}).json()
        self.assertEqual(res, {'error': 'address is empty'})

    def test_wrong_param(self):
        code = requests.post(f'{self.url}/comb/', json={'addr': self.address}).status_code
        assert code == 422
        code = requests.post(f'{self.url}/rule/', json={'addr': self.address}).status_code
        assert code == 422
        code = requests.post(f'{self.url}/stat/', json={'addr': self.address}).status_code
        assert code == 422

    def test_rule_function(self):
        res = requests.post(f'{self.url}/rule/', json={'address': self.address}).json()
        assert res['country'] == 'Estonia', 'rule model failed to predict the right country'

    def test_stat_function(self):
        res = requests.post(f'{self.url}/stat/', json={'address': self.address}).json()
        assert res['country'] == 'Estonia', 'stat model failed to predict the right country'

    def test_comb_function(self):
        res = requests.post(f'{self.url}/comb/', json={'address': self.address}).json()
        assert res['country'] == 'Estonia', 'comb model failed to predict the right country'

    def test_rule_latency(self):
        st = time.time()
        for _ in range(100):
            requests.post(f'{self.url}/rule/', json={'address': self.address})
        print(f'Rule: {((time.time() - st) / 100) * 1000:2f}ms per item')

    def test_stat_latency(self):
        st = time.time()
        for _ in range(100):
            requests.post(f'{self.url}/stat/', json={'address': self.address})
        print(f'Stat: {((time.time() - st) / 100) * 1000:2f}ms per item')

    def test_comb_latency(self):
        st = time.time()
        for _ in range(100):
            requests.post(f'{self.url}/comb/', json={'address': self.address})
        print(f'Comb: {((time.time() - st) / 100) * 1000:2f}ms per item')


if __name__ == '__main__':
    unittest.main()