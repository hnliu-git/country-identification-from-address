
from evaluator import Evaluator
from models.rule_model import *
from models.stat_model import *
from preset import code2country


if __name__ == '__main__':
    # r_model = RuleModel(code2country.keys())
    # evaluator = Evaluator(code2country.keys())
    #
    # # Evaluate metrics
    # evaluator.evaluate_metrics(r_model)
    #
    # # Evaluate latency
    # evaluator.evaluate_latency(r_model, n=100)

    model_path = 'ckpts/last.ckpt/'
    s_model = StatModel(model_path, code2country.keys())
    evaluator = Evaluator(code2country.keys())

    # Evaluate metrics
    # evaluator.evaluate_metrics(s_model)

    # Evaluate latency
    evaluator.evaluate_latency(s_model, n=100)