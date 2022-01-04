from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from sentevalpl.pairs_classification import RelatednessEval, EntailmentEval
from sentevalpl.classifier import SentEvalClassifier


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
        self.list_tasks = [
            "WCCRS_HOTELS", "WCCRS_MEDICINE", "CDSEntailment", "CDSRelatedness",
            "SICKEntailment", "SICKRelatedness", "8TAGS"
        ]

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        if name == 'WCCRS_HOTELS':
            self.evaluation = SentEvalClassifier(tpath + '/downstream/WCCRS_HOTELS', name, 4, seed=self.params.seed)
        elif name == 'WCCRS_MEDICINE':
            self.evaluation = SentEvalClassifier(tpath + '/downstream/WCCRS_MEDICINE', name, 4, seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = RelatednessEval(tpath + '/downstream/SICK', task_name='SICK', seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = EntailmentEval(tpath + '/downstream/SICK', task_name='SICK', seed=self.params.seed)
        elif name == 'CDSRelatedness':
            self.evaluation = RelatednessEval(tpath + '/downstream/CDS', task_name='CDS', seed=self.params.seed)
        elif name == 'CDSEntailment':
            self.evaluation = EntailmentEval(tpath + '/downstream/CDS', task_name='CDS', seed=self.params.seed)
        elif name == '8TAGS':
            self.evaluation = SentEvalClassifier(tpath + '/downstream/8TAGS', name, 8, seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)
        self.results = self.evaluation.run(self.params, self.batcher)
        return self.results
