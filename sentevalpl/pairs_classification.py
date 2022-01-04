# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Relatedness and Entailment tasks
'''
from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

from senteval.tools.relatedness import RelatednessPytorch
from senteval.tools.validation import SplitClassifier


class RelatednessEval(object):
    def __init__(self, task_path, task_name = 'SICK', seed=1111):
        logging.debug(f'***** Transfer task : {task_name}-Relatedness*****\n\n')
        self.task_name = task_name
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, f'{task_name}_train.txt'))
        dev = self.loadFile(os.path.join(task_path, f'{task_name}_trial.txt'))
        test = self.loadFile(os.path.join(task_path, f'{task_name}_test_annotated.txt'))
        self.data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.data['train']['X_A'] + \
                  self.data['train']['X_B'] + \
                  self.data['dev']['X_A'] + \
                  self.data['dev']['X_B'] + \
                  self.data['test']['X_A'] + self.data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        skipFirstLine = True
        data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    data['X_A'].append(text[1].split())
                    data['X_B'].append(text[2].split())
                    data['y'].append(text[3])

        data['y'] = [float(s) for s in data['y']]
        return data

    def run(self, params, batcher):
        embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.data[key]['X_A'],
                                       self.data[key]['X_B'],
                                       self.data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                params.batcher_dataset = f"{key}_{txt_type}"
                embed[key][txt_type] = []
                for ii in range(0, len(self.data[key]['y']), bsize):
                    params.batcher_offset = str(ii)
                    batch = self.data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    embed[key][txt_type].append(embeddings)
                embed[key][txt_type] = np.vstack(embed[key][txt_type])
            embed[key]['y'] = np.array(self.data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = embed['train']['X_A']
        trainB = embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = self.encode_labels(self.data['train']['y'])

        # Dev
        devA = embed['dev']['X_A']
        devB = embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = self.encode_labels(self.data['dev']['y'])

        # Test
        testA = embed['test']['X_A']
        testB = embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = self.encode_labels(self.data['test']['y'])

        config = {'seed': self.seed, 'nclasses': 5}
        clf = RelatednessPytorch(train={'X': trainF, 'y': trainY},
                                 valid={'X': devF, 'y': devY},
                                 test={'X': testF, 'y': testY},
                                 devscores=self.data['dev']['y'],
                                 config=config)

        devpr, yhat = clf.run()

        pr = pearsonr(yhat, self.data['test']['y'])[0]
        sr = spearmanr(yhat, self.data['test']['y'])[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(yhat, self.data['test']['y'])
        logging.debug('Dev : Pearson {0}'.format(devpr))
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                       for {3} Relatedness\n'.format(pr, sr, se, self.task_name))

        return {'devpearson': devpr, 'pearson': pr, 'spearman': sr, 'mse': se,
                'yhat': yhat, 'ndev': len(devA), 'ntest': len(testA)}

    def encode_labels(self, labels, nclass=5):
        """
        Label encoding from Tree LSTM paper (Tai, Socher, Manning)
        """
        Y = np.zeros((len(labels), nclass)).astype('float32')
        for j, y in enumerate(labels):
            for i in range(nclass):
                if i+1 == np.floor(y) + 1:
                    Y[j, i] = y - np.floor(y)
                if i+1 == np.floor(y):
                    Y[j, i] = np.floor(y) - y + 1
        return Y


class EntailmentEval(RelatednessEval):
    def __init__(self, task_path, task_name='SICK', seed=1111):
        super().__init__(task_path, task_name, seed)
        logging.debug(f'***** Transfer task : {task_name}-Entailment*****\n\n')
        self.task_name = task_name
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, f'{task_name}_train.txt'))
        dev = self.loadFile(os.path.join(task_path, f'{task_name}_trial.txt'))
        test = self.loadFile(os.path.join(task_path, f'{task_name}_test_annotated.txt'))
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        label2id = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        skipFirstLine = True
        data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    data['X_A'].append(text[1].split())
                    data['X_B'].append(text[2].split())
                    data['y'].append(text[4])
        data['y'] = [label2id[s] for s in data['y']]
        return data

    def run(self, params, batcher):
        embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.data[key]['X_A'],
                                       self.data[key]['X_B'],
                                       self.data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                params.batcher_dataset = f"{key}_{txt_type}"
                embed[key][txt_type] = []
                for ii in range(0, len(self.data[key]['y']), bsize):
                    params.batcher_offset = str(ii)
                    batch = self.data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    embed[key][txt_type].append(embeddings)
                embed[key][txt_type] = np.vstack(embed[key][txt_type])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = embed['train']['X_A']
        trainB = embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = np.array(self.data['train']['y'])

        # Dev
        devA = embed['dev']['X_A']
        devB = embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = np.array(self.data['dev']['y'])

        # Test
        testA = embed['test']['X_A']
        testB = embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = np.array(self.data['test']['y'])

        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid}
        clf = SplitClassifier(X={'train': trainF, 'valid': devF, 'test': testF},
                              y={'train': trainY, 'valid': devY, 'test': testY},
                              config=config)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
                       {2} entailment\n'.format(devacc, testacc, self.task_name))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(devA), 'ntest': len(testA)}
