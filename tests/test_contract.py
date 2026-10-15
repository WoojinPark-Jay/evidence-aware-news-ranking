import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pec_fuse import build_feature_frame, query_metrics, rank_within_query, paired_user_cluster_bootstrap

class ContractTests(unittest.TestCase):
    def fixture(self):
        return pd.read_csv(Path(__file__).resolve().parents[1]/'examples/synthetic/candidate_inputs.csv')
    def test_infinite_features_rejected(self):
        frame=self.fixture(); frame.loc[0,'P_semantic']=np.inf
        with self.assertRaises(ValueError): build_feature_frame(frame)
    def test_inconsistent_trace_rejected(self):
        frame=self.fixture(); frame.loc[0,'same_day_recommended']=1; frame.loc[0,'retained_rank']=0
        with self.assertRaises(ValueError): build_feature_frame(frame)
    def test_known_ranking_and_stable_ties(self):
        frame=pd.DataFrame({'query_id':['q']*3,'user_id':['u']*3,'label':[0,1,0]})
        ranked=rank_within_query(frame,[1,1,0])
        metrics=query_metrics(ranked,k=1).iloc[0]
        self.assertEqual(metrics['recall@1'],0)
        self.assertEqual(metrics['mrr'],0.5)
    def test_identical_models_have_zero_delta(self):
        frame=pd.DataFrame({'query_id':['a','b','c'],'user_id':['u','u','v'],'ndcg@10':[0.3,0.7,1.]})
        result=paired_user_cluster_bootstrap(frame,frame,iterations=50)
        self.assertEqual((result['mean_delta'],result['ci_low'],result['ci_high']),(0,0,0))
if __name__=='__main__': unittest.main()
