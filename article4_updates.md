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

## A Numerical Example of Sequential Bootstrap

We have three observations (labels) with the following time spans:

- **Observation 1**: Spans time periods 1 to 3 (r₁,₃)
- **Observation 2**: Spans time periods 3 to 4 (r₃,₄)  
- **Observation 3**: Spans time periods 5 to 6 (r₅,₆)

The **indicator matrix** {1ₜ,ᵢ} shows when each observation is active:

| Time | Obs 1 | Obs 2 | Obs 3 |
|------|-------|-------|-------|
| 1    | 1     | 0     | 0     |
| 2    | 1     | 0     | 0     |
| 3    | 1     | 1     | 0     |
| 4    | 0     | 1     | 0     |
| 5    | 0     | 0     | 1     |
| 6    | 0     | 0     | 1     |

---

## Sequential Bootstrap Process

### Step 1: First Draw
- Initial probabilities: δ₁ = 1/3, δ₂ = 1/3, δ₃ = 1/3
- **Observation 2** is selected (as given in the book)
- Current sample: φ¹ = {2}

### Step 2: Update Probabilities for Second Draw

**Calculate average uniqueness for each observation:**

- **Observation 1**:
  - Active at times: 1,2,3
  - Overlaps with Obs 2 only at time 3
  - Uniqueness calculation:
    - Time 1: 1/(1+0) = 1
    - Time 2: 1/(1+0) = 1  
    - Time 3: 1/(1+1) = 0.5
  - Average uniqueness: (1 + 1 + 0.5)/3 = 2.5/3 = 5/6

- **Observation 2**:
  - Already selected → high self-overlap
  - Active at times: 3,4
  - Uniqueness calculation:
    - Time 3: 1/(1+1) = 0.5
    - Time 4: 1/(1+0) = 1
  - Average uniqueness: (0.5 + 1)/2 = 1.5/2 = 3/6 = 1/2

- **Observation 3**:
  - Active at times: 5,6
  - No overlap with Obs 2
  - Uniqueness calculation:
    - Time 5: 1/(1+0) = 1
    - Time 6: 1/(1+0) = 1
  - Average uniqueness: (1 + 1)/2 = 1 = 6/6

**Calculate draw probabilities:**
- Total uniqueness sum = 5/6 + 3/6 + 6/6 = 14/6
- δ¹ = (5/6) ÷ (14/6) = 5/14
- δ² = (3/6) ÷ (14/6) = 3/14  
- δ³ = (6/6) ÷ (14/6) = 6/14

**Second draw probabilities: δ² = [5/14, 3/14, 6/14]**

### Step 3: Second Draw
- Probabilities: [5/14, 3/14, 6/14]
- **Observation 3** is selected (as given in the book)
- Current sample: φ² = {2, 3}

### Step 4: Update Probabilities for Third Draw

**Calculate average uniqueness for each observation:**

- **Observation 1**:
  - Active at times: 1,2,3
  - Overlaps:
    - With Obs 2 at time 3
    - With Obs 3 at no times
  - Uniqueness calculation:
    - Time 1: 1/(1+0) = 1
    - Time 2: 1/(1+0) = 1
    - Time 3: 1/(1+1) = 0.5
  - Average uniqueness: (1 + 1 + 0.5)/3 = 2.5/3 = 5/6

- **Observation 2**:
  - Active at times: 3,4
  - Overlaps:
    - With Obs 1 at time 3
    - With Obs 3 at no times
  - Uniqueness calculation:
    - Time 3: 1/(1+1) = 0.5
    - Time 4: 1/(1+0) = 1
  - Average uniqueness: (0.5 + 1)/2 = 1.5/2 = 3/6 = 1/2

- **Observation 3**:
  - Active at times: 5,6
  - Overlaps with no other observations in sample
  - Uniqueness calculation:
    - Time 5: 1/(1+0) = 1
    - Time 6: 1/(1+0) = 1
  - Average uniqueness: (1 + 1)/2 = 1 = 6/6

**Calculate final draw probabilities:**
- Total uniqueness sum = 5/6 + 3/6 + 6/6 = 14/6
- δ¹ = (5/6) ÷ (14/6) = 5/14
- δ² = (3/6) ÷ (14/6) = 3/14
- δ³ = (6/6) ÷ (14/6) = 6/14

**Third draw probabilities: δ³ = [5/14, 3/14, 6/14]**

### Step 5: Third Draw
- Probabilities: [5/14, 3/14, 6/14]
- Final sample depends on random draw from these probabilities

---

## Complete Probability Summary

| Draw | Observation 1 | Observation 2 | Observation 3 | Selected |
|------|---------------|---------------|---------------|----------|
| 1    | 1/3 (33.3%)   | 1/3 (33.3%)   | 1/3 (33.3%)   | 2        |
| 2    | 5/14 (35.7%)  | 3/14 (21.4%)  | 6/14 (42.9%)  | 3        |
| 3    | 5/14 (35.7%)  | 3/14 (21.4%)  | 6/14 (42.9%)  | ?        |

---

## Key Observations from This Example

1. **Lowest probability** goes to the previously selected observation (Observation 2)
2. **Highest probability** goes to the observation with no overlap (Observation 3)
3. **Observation 1** has intermediate probability due to partial overlap
4. **Probabilities remain the same** for draw 2 and draw 3 because the overlap pattern doesn't change

---

## Why This Pattern Occurs

- **Observation 2** (already selected) gets penalized for self-overlap
- **Observation 3** gets rewarded for having zero overlap with current sample  
- **Observation 1** gets moderate probability due to partial overlap at only one time period
- The method successfully **discourages redundancy** while allowing for sample diversity

---
## Visualizing the Superior Uniqueness of Sequential Bootstrap

**Figure 1** provides crucial visual evidence from Monte Carlo experiments comparing the effectiveness of the sequential bootstrap method against the standard bootstrap method.

**(Insert Figure 4.2 here)**
*Caption: Monte Carlo experiment comparing the distribution of average uniqueness for standard bootstrap (left) and sequential bootstrap (right). The sequential method produces samples with significantly higher average uniqueness.*

The chart presents a histogram that visually compares the `average uniqueness` of samples generated through two methods:
- **Standard Bootstrap** (left distribution): The conventional method of random sampling with replacement.
- **Sequential Bootstrap** (right distribution): The advanced method introduced in Section 4.5.1 that dynamically adjusts draw probabilities to reduce overlap.

The key finding is that the distribution for the sequential bootstrap is shifted noticeably to the right, indicating a higher average uniqueness value across many experimental trials[citation:1].

### Methodology Behind the Data

The data for this figure was generated through extensive **Monte Carlo simulations**:

1. **Random Series Generation**: Multiple random `t1` series were created, where each observation has a random start time and duration[citation:1].
2. **Indicator Matrix Formation**: For each series, the `getIndMatrix` function was used to create the corresponding indicator matrix, mapping which observations were active at each time point.
3. **Bootstrap Sampling**: For each generated indicator matrix:
   - Samples were drawn using the **standard bootstrap** method.
   - Samples were drawn using the **sequential bootstrap** method (`seqBootstrap`).
4. **Uniqueness Calculation**: For each generated sample, the average uniqueness was calculated using the `getAvgUniqueness` function.
5. **Distribution Plotting**: The uniqueness values from thousands of iterations were compiled into the two histograms shown in the figure.


Two key quantitative findings are demonstrated:

- The **median average uniqueness** for the standard bootstrap method is approximately **0.6**.
- The **median average uniqueness** for the sequential bootstrap method is approximately **0.7**.

This difference is not just visually apparent but also **statistically significant**. An ANOVA test on the difference of means reportedly returns a "vanishingly small probability," meaning we can be highly confident that the sequential bootstrap method genuinely produces more unique samples.

To recreate this analysis, implement the Monte Carlo experiment using the functions provided in the attachment *mc_experiment.py*.

