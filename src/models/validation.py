import scipy.stats
import numpy as np
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "."))
import itertools
import statsmodels.stats.multitest
import sklearn.metrics
import imblearn.metrics
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class StatisticalAnalysis:
    def __init__(self,results_location):
        self.scores = {}
        for filename in os.listdir(results_location):
            if filename.endswith(".json"):
                estimator_name = filename.split('.')[0]
                self.scores[estimator_name] = {}
                with open(results_location + filename, "r") as read_file:
                    data = json.load(read_file)
                    self.scores[estimator_name] = data['test_f2_True']
    def analyze(self):
        self.combinations = list(itertools.combinations(self.scores.items(),2))
        self.summary = {}
        for model in self.scores.keys():
            self.summary[model] = {
                'scores':self.scores[model],
                'normally_distributed':scipy.stats.shapiro(self.scores[model])[1] > 0.05,
                'mean_score':np.mean(self.scores[model]),
                'standard_deviation_scores': np.std(self.scores[model]),
                'confidence_interval_scores':scipy.stats.norm.interval(0.95,loc=np.mean(self.scores[model]),scale=np.std(self.scores[model]))
            }
        self.p_values = []
        for combination in self.combinations:
            normal_distribution = self.summary[combination[0][0]]['normally_distributed'] & self.summary[combination[1][0]]['normally_distributed']
            if normal_distribution:
                p = scipy.stats.ttest_rel(combination[0][1],combination[1][1]).pvalue
            else:
                p = scipy.stats.wilcoxon(combination[0][1],combination[1][1]).pvalue
            self.p_values.append(p)
        self.p_values_corrected = statsmodels.stats.multitest.multipletests(self.p_values,method='bonferroni',returnsorted=False)[1]
    def generate_report(self,report_location):
        self.report = {
            'summary':self.summary,
            'statistical_report': {
                'combinations' : self.combinations,
                'p_values': self.p_values,
                'p_values_corrected': self.p_values_corrected
            }
        }
        with open(report_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.report, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)

class Performance:
    def __init__(self,results_file,threshold=None):
        results = np.loadtxt(results_file)
        self.threshold = threshold
        self.true = np.array(results[:,0],dtype=int)
        self.predicted_proba = results[:,3]
        if threshold == None:
            self.predicted_class = np.array(results[:,1],dtype=int)
        else:
            self.predicted_class = self.predicted_proba >= threshold
    def analyze(self):
        self.classification_report = sklearn.metrics.classification_report(self.true,self.predicted_class,output_dict=True)
        self.f2_score = sklearn.metrics.fbeta_score(self.true,self.predicted_class,beta=2, pos_label=1)
        self.confusion_matrix = sklearn.metrics.confusion_matrix(self.true,self.predicted_class)
        self.roc_curve = sklearn.metrics.roc_curve(self.true,self.predicted_proba)
        self.roc_auc_score = sklearn.metrics.roc_auc_score(self.true,self.predicted_proba)
        self.geometric_mean = imblearn.metrics.geometric_mean_score(self.true,self.predicted_class)
        FP = self.confusion_matrix[0,1]
        FN = self.confusion_matrix[1,0]
        TP = self.confusion_matrix[1,1]
        TN = self.confusion_matrix[0,0]

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        self.TPR = TP/(TP+FN)
        # Specificity or true negative rate
        self.TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        self.PPV = TP/(TP+FP)
        # Negative predictive value
        self.NPV = TN/(TN+FN)
        # Fall out or false positive rate
        self.FPR = FP/(FP+TN)
        # False negative rate
        self.FNR = FN/(TP+FN)
        # False discovery rate
        self.FDR = FP/(TP+FP)

        self.total = FP + FN + TP + TN
        self.calls = FP + TP
        self.nsp_i = FN + TP

        self.nsp_i_p=self.nsp_i / self.total
        self.calls_p=self.calls / self.total
        self.nsp_f_p=FN/self.total
        self.reduction_p=1-self.nsp_f_p/self.nsp_i_p
        self.cost_effectiveness=self.reduction_p * (1 - self.calls_p)
        self.cost_effectiveness_2 = self.reduction_p / self.calls_p

        self.report = {
            'threshold':self.threshold,
            'classification_report': self.classification_report,
            'table': {
                'precision':self.classification_report["1"]['precision'],
                'recall':self.classification_report["1"]['recall'],
                'f1-score':self.classification_report["1"]['f1-score'],
                'f2-score':self.f2_score,
                'geometric_mean': self.geometric_mean,
                'nsp_i_p':self.nsp_i_p,
                'calls_p':self.calls_p,
                'nsp_f_p':self.nsp_f_p,
                'reduction_p':self.reduction_p,
                'cost_effectiveness':self.cost_effectiveness,
                'cost_effectiveness_2':self.cost_effectiveness_2,
                'support':self.classification_report["1"]['support']
            },
            'f2_score_1':self.f2_score,
            'roc_auc_score': self.roc_auc_score,
            'confusion_matrix': self.confusion_matrix,
            'TPR': self.TPR,
            'TNR': self.TNR,
            'PPV': self.PPV,
            'NPV': self.NPV,
            'FPR': self.FPR,
            'FNR': self.FNR,
            'FDR': self.FDR,
            'roc_curve': self.roc_curve
        }
    def generate_report(self,report_location):
        with open(report_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.report, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)

class ThresholdTuner:
    def __init__(self,results_file):
        results = np.loadtxt(results_file)
        self.true = np.array(results[:,0],dtype=bool)
        self.predicted_proba = results[:,3]
    def tune(self,ratio=1,min_calls_p=0.1,min_threshold=0.5):
        thresholds = np.arange(0,1,0.01)
        costs = np.array([ratio,1]) / (ratio + 1)
        logger.info("costs: " + str(costs))
        type_1_2_errors_sums = []
        calls_ps = []

        for threshold in thresholds:
            predicted = self.predicted_proba >= threshold
            confusion_matrix = sklearn.metrics.confusion_matrix(self.true,predicted)
            FP = confusion_matrix[0,1]
            FN = confusion_matrix[1,0]
            TP = confusion_matrix[1,1]
            TN = confusion_matrix[0,0]

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)

            total = FP + FN + TP + TN
            nsp_i = FN + TP
            N_show = total - nsp_i
            calls = FP + TP
            calls_p=calls / total

            type_1_error = FP/N_show
            type_2_error = FN/nsp_i
            type_1_2_errors_sum = (costs[0] * type_1_error + costs[1] * type_2_error)
            type_1_2_errors_sums.append(type_1_2_errors_sum)
            calls_ps.append(calls_p)
        performances = np.column_stack([thresholds, calls_ps, type_1_2_errors_sums])
        performances = performances[(performances[:,1]>min_calls_p) & (performances[:,0]>min_threshold),:]
        if performances.shape[0] > 0:
            min_idx = np.argmin(performances[:,2])
            min_threshold_calculated = performances[min_idx,0]
        else:
            min_threshold_calculated = 0.5
        logger.info("threshold: " + str(min_threshold_calculated))
        return min_threshold_calculated