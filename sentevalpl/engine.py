from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from sentevalpl.pairs_classification import RelatednessEval, EntailmentEval
from sentevalpl.classifier import SentEvalClassifier
from sentevalpl.tasks import get_task_names, get_task_by_name


class SE(object):
    def __init__(self, params, batcher, prepare=None):
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed
        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None
        self.list_tasks = get_task_names()

    def eval(self, name):
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        task = get_task_by_name(name)
        task_dir = task["dir"]
        task_path = f"{tpath}/downstream/{task_dir}"
        task_type = task["type"]
        classes = {"classification": SentEvalClassifier, "entailment": EntailmentEval, "relatedness": RelatednessEval}
        eval_class = classes[task_type]
        if task_type == "classification":
            self.evaluation = eval_class(task_path, name, task["num_classes"], seed=self.params.seed)
        else:
            self.evaluation = eval_class(task_path, task_dir, seed=self.params.seed)
        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)
        self.results = self.evaluation.run(self.params, self.batcher)
        return self.results
