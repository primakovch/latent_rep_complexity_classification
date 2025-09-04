import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union, Optional, Dict
from scipy.stats import mode
from tqdm import tqdm

class DecisionMaker():
    def __init__(self, feature_embedding: Dict):

        self.feature_embedding = dict()

        for complexity in feature_embedding.keys():
            layer_embedding = []
            for layers in feature_embedding[complexity]:
                layer_embedding.append(feature_embedding[complexity][layers])
            self.feature_embedding[complexity] = np.array(layer_embedding).squeeze()
    
    def make_decision(self, test_embedding):
        """_summary_

        Args:
            test_embedding (numpy array): embedding of test claims of shape (num_claims, num_layers, embedding_size)
        """


        final_decision_list = []

        for idx in tqdm(range(test_embedding.shape[0])):
            simple_dist = np.diagonal(cosine_similarity(test_embedding[idx], self.feature_embedding["simple_claims"]))
            intermediate_dist = np.diagonal(cosine_similarity(test_embedding[idx], self.feature_embedding["intermediate_claims"]))
            complex_dist = np.diagonal(cosine_similarity(test_embedding[idx], self.feature_embedding["complex_claims"]))

            distance_metric = np.transpose(np.vstack([simple_dist, intermediate_dist, complex_dist]), (1,0))
            # distance_metric = np.transpose(np.vstack([simple_dist, complex_dist]), (1,0))
            decision = np.argmax(distance_metric, axis = 1)
            final_complexity, count = mode(decision)
            
            # if final_complexity == 1: 
            #     final_complexity = 2

            final_decision_list.append(final_complexity)
        return final_decision_list

        
            

        

        