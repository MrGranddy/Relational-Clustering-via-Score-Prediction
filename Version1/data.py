import torch

from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal


class DataGenerator:

    def __init__(self, num_dim, num_modes):

        if num_dim % num_modes != 0:
            raise ValueError('Number of modes must divide number of dimensions.')

        self.num_dim = num_dim
        self.num_modes = num_modes
        self.len_weight_partitions = num_dim // num_modes

        # Create mode means
        self.means = torch.randn(num_modes, num_dim) * 5

        # Create mode covariances
        self.covs = torch.randn(num_modes, num_dim, num_dim)
        self.covs = torch.bmm(self.covs, self.covs.transpose(1, 2))

        # Normalize covariances
        self.covs /= torch.max(torch.abs(self.covs))

        # Categorical distribution probabilities
        self.cat_probs = torch.ones(num_modes) / num_modes

        # Create score weights
        self.weights = torch.randn(num_dim, 1)
        self.quadratic_weights = torch.randn(num_dim, num_dim)

    def sample(self, sample_size):

        # Sample modes
        mode_idx = torch.multinomial(self.cat_probs, sample_size, replacement=True)

        # Sample from modes
        samples = torch.zeros(sample_size, self.num_dim)
        for i in range(self.num_modes):
            idx = mode_idx == i
            samples[idx] = MultivariateNormal(self.means[i], self.covs[i]).sample((idx.sum(),))

        # Compute scores
        scores = torch.zeros(sample_size, 1)

        # Zero out irrelevant parts of the weights for each mode (each equal lenght partition is for one mode)
        weights = self.weights.clone()
        for i in range(self.num_modes):
            idx = mode_idx == i

            start = i * self.len_weight_partitions
            end = (i + 1) * self.len_weight_partitions

            scores[idx] = samples[idx] @ weights

        # Add quadratic terms
        for i in range(self.num_modes):
            idx = mode_idx == i

            start = i * self.len_weight_partitions
            end = (i + 1) * self.len_weight_partitions

            scores[idx] += torch.mm(torch.mm(samples[idx, start:end], self.quadratic_weights[start:end, start:end]), samples[idx, start:end].t()).diag().unsqueeze(1)

        # Add noise
        scores += torch.randn_like(scores) * 0.1

        scores = scores / 100

        return samples, scores, mode_idx
    
if __name__ == '__main__':

    generator = DataGenerator(10, 5)
    samples, scores, mode_idx = generator.sample(1000)

    import matplotlib.pyplot as plt

    plt.scatter(samples[:, 0], samples[:, 1], c=mode_idx)
    plt.show()