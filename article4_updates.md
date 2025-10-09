# Sequential Bootstrapping: Generating "Purer" Ensembles

Sequential Bootstrapping is an advanced resampling technique designed to create bootstrap samples that maximize the average uniqueness of the selected labels. It is used in the bagging process of ensemble methods like Random Forests.

## Why Standard Bootstrapping Fails in Finance

Standard bootstrapping randomly draws samples with replacement from the entire dataset, under the assumption that all samples are independent. In finance, this results in bootstrap samples that are often full of concurrent, overlapping labels. Training trees on these highly impure samples leads to poorly diversified and overfit ensembles.

## How Sequential Bootstrapping Works

The objective of sequential bootstrapping is to still draw random samples, but in a way that the average uniqueness (purity) of the selected subsample is maximized. The process can be broken down as follows:

1. **Build an Indicator Matrix**: This matrix tracks which time periods (e.g., price returns) influence each label. A value of 1 indicates that a specific return was used in generating a specific label.
2. **Sequential Sample Selection**: Instead of drawing all samples at once, they are drawn one by one. After each draw, the probability of selecting a new sample is updated based on its uniqueness relative to the samples already chosen for the current bootstrap sample. Samples that contain information (returns) not already over-represented in the current bootstrap set have a higher probability of being selected next.
3. **Build the Ensemble**: This process is repeated to create multiple bootstrap samples. Each sample is, on average, "purer" than one created by standard bootstrapping. A model (like a decision tree) is then trained on each of these sequentially bootstrapped samples.

The table below contrasts the two approaches:

| Feature | Standard Bootstrapping | Sequential Bootstrapping |
| :--- | :--- | :--- |
| **Core Assumption** | Data is IID | Data contains temporal dependencies |
| **Sampling Method** | Pure random sampling with replacement | Sequential sampling that maximizes uniqueness |
| **Sample "Purity"** | Low; samples have high concurrency | High; samples are more diverse and unique |
| **Resulting Ensemble** | Trees can be correlated due to overlapping information | Trees are more diverse and less prone to overfitting |

## Key Advantages

Integrating Sequential Bootstrapping into your ensemble models provides two key benefits:

- **More Robust Models**: By building trees on less correlated, more unique samples, the ensemble becomes more robust and better at capturing diverse patterns in the data.
- **Better Performance Estimates**: The out-of-bag (OOB) score from a sequentially bootstrapped model is less inflated and provides a more reliable estimate of the model's out-of-sample performance.