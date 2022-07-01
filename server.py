"""
Web service for country identification
"""

import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from models.rule_model import RuleModel
from models.stat_model import StatModel
from preset import code2country


# Request model
class Request(BaseModel):
    address: str


# Load config file
config = yaml.load(open('configs/service.yaml'), Loader=yaml.FullLoader)
# Load models
s_model = StatModel(config['s_model'], config['countries'])
r_model = RuleModel(config['countries'])
# Create app
app = FastAPI()


# Endpoint for statistical-based identification
@app.post("/stat/")
def stat_model(request: Request):
    if request.address == "": return {'error': 'address is empty'}
    pred_dict = s_model(request.address)
    code = max(pred_dict, key=pred_dict.get)
    return {'country': code2country[code], 'score': pred_dict[code]}


# Endpoint for rule-based identification
@app.post("/rule/")
def rule_model(request: Request):
    if request.address == "": return {'error': 'address is empty'}
    pred_dict = r_model(request.address)
    code = max(pred_dict, key=pred_dict.get)
    return {'country': code2country[code], 'score': pred_dict[code]}


# Endpoint for combined method identification
@app.post("/comb/")
def comb_model(request: Request):
    if request.address == "": return {'error': 'address is empty'}
    s_dict = s_model(request.address)
    r_dict = r_model(request.address)
    for k, v in s_dict.items():
        r_dict[k] += v
    code = max(r_dict, key=r_dict.get)
    return {'country': code2country[code], 'score': r_dict[code]}








