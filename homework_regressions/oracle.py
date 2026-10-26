from sklearn.datasets import make_spd_matrix
import numpy as np
import torch 
import torch.distributions as d


class BiasOracle():
    def __init__(self, lambdas, distribution, sample):
        self.lambdas = lambdas
        self.distribution = distribution
        self.sample = sample

    def predict(self, observation):
        observation = observation.unsqueeze(1)
        ratios = torch.log(self.distribution.mixture_distribution.probs)
        likehood = self.distribution.component_distribution.log_prob(observation)
        aposterial = torch.log(self.lambdas) + likehood + ratios
        return torch.argmax(aposterial, dim = -1)

    def empirical_risk(self, predictions, labels):
        N = predictions.shape[0]
        misses = np.bitwise_xor(predictions, labels).sum()
        return misses / N
        

#if __name__ == "__main__":
