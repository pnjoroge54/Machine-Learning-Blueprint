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

# Section 4.5.3: A Numerical Example of Sequential Bootstrap

## 🎯 Objective
To understand how the **sequential bootstrap** method works by walking through the exact example from the book with three observations and their overlapping time spans.

---

## 📌 Scenario Setup (From the Book)

We have three observations (labels) with the following time spans:

- **Observation 1**: Spans time periods 0 to 2 (r₀,₃)
- **Observation 2**: Spans time periods 2 to 3 (r₂,₄)  
- **Observation 3**: Spans time periods 4 to 5 (r₄,₆)

The **indicator matrix** {1ₜ,ᵢ} shows when each observation is active:

| Time | Obs 1 | Obs 2 | Obs 3 |
|------|-------|-------|-------|
| 0    | 1     | 0     | 0     |
| 1    | 1     | 0     | 0     |
| 2    | 1     | 1     | 0     |
| 3    | 0     | 1     | 0     |
| 4    | 0     | 0     | 1     |
| 5    | 0     | 0     | 1     |

---

## 🔁 Sequential Bootstrap Process

### Step 1: First Draw
- Initial probabilities: \( \delta^{(1)} = \left[ \frac{1}{3}, \frac{1}{3}, \frac{1}{3} \right] \)
- **Observation 2** is selected (as given in the book)
- Current sample: \( \phi^{(1)} = \{2\} \)

### Step 2: Update Probabilities for Second Draw

**Calculate average uniqueness for each observation:**

- **Observation 1**:
  - Active at times: 0,1,2
  - Overlaps with Obs 2 only at time 2
  - Uniqueness calculation:
    - Time 0: 1/(1+0) = 1
    - Time 1: 1/(1+0) = 1  
    - Time 2: 1/(1+1) = 0.5
  - Average uniqueness: \( \bar{u}_1^{(2)} = (1 + 1 + 0.5)/3 = \frac{2.5}{3} = \frac{5}{6} \)

- **Observation 2**:
  - Already selected → high self-overlap
  - Active at times: 2,3
  - Uniqueness calculation:
    - Time 2: 1/(1+1) = 0.5
    - Time 3: 1/(1+0) = 1
  - Average uniqueness: \( \bar{u}_2^{(2)} = (0.5 + 1)/2 = \frac{1.5}{2} = \frac{3}{6} = \frac{1}{2} \)

- **Observation 3**:
  - Active at times: 4,5
  - No overlap with Obs 2
  - Uniqueness calculation:
    - Time 4: 1/(1+0) = 1
    - Time 5: 1/(1+0) = 1
  - Average uniqueness: \( \bar{u}_3^{(2)} = (1 + 1)/2 = 1 = \frac{6}{6} \)

**Calculate draw probabilities:**
- Total uniqueness sum = \( \frac{5}{6} + \frac{3}{6} + \frac{6}{6} = \frac{14}{6} \)
- \( \delta^{(2)} = \left[ \frac{5}{14}, \frac{3}{14}, \frac{6}{14} \right] \)

### Step 3: Second Draw
- Probabilities: \( \left[ \frac{5}{14}, \frac{3}{14}, \frac{6}{14} \right] \)
- **Observation 3** is selected (as given in the book)
- Current sample: \( \phi^{(2)} = \{2, 3\} \)

### Step 4: Update Probabilities for Third Draw

**Calculate average uniqueness for each observation:**

- **Observation 1**:
  - Active at times: 0,1,2
  - Overlaps:
    - With Obs 2 at time 2
    - With Obs 3 at no times
  - Uniqueness calculation:
    - Time 0: 1/(1+0) = 1
    - Time 1: 1/(1+0) = 1
    - Time 2: 1/(1+1) = 0.5
  - Average uniqueness: \( \bar{u}_1^{(3)} = (1 + 1 + 0.5)/3 = \frac{2.5}{3} = \frac{5}{6} \)

- **Observation 2**:
  - Active at times: 2,3
  - Overlaps:
    - With Obs 1 at time 2
    - With Obs 3 at no times
  - Uniqueness calculation:
    - Time 2: 1/(1+1) = 0.5
    - Time 3: 1/(1+0) = 1
  - Average uniqueness: \( \bar{u}_2^{(3)} = (0.5 + 1)/2 = \frac{1.5}{2} = \frac{3}{6} = \frac{1}{2} \)

- **Observation 3**:
  - Active at times: 4,5
  - Overlaps with no other observations in sample
  - Uniqueness calculation:
    - Time 4: 1/(1+0) = 1
    - Time 5: 1/(1+0) = 1
  - Average uniqueness: \( \bar{u}_3^{(3)} = (1 + 1)/2 = 1 = \frac{6}{6} \)

**Calculate final draw probabilities:**
- Total uniqueness sum = \( \frac{5}{6} + \frac{3}{6} + \frac{6}{6} = \frac{14}{6} \)
- \( \delta^{(3)} = \left[ \frac{5}{14}, \frac{3}{14}, \frac{6}{14} \right] \)

### Step 5: Third Draw
- Probabilities: \( \left[ \frac{5}{14}, \frac{3}{14}, \frac{6}{14} \right] \)
- Final sample depends on random draw from these probabilities

---

## 📊 Complete Probability Summary

| Draw | Observation 1 | Observation 2 | Observation 3 | Selected |
|------|---------------|---------------|---------------|----------|
| 1    | 1/3           | 1/3           | 1/3           | 2        |
| 2    | 5/14          | 3/14          | 6/14          | 3        |
| 3    | 5/14          | 3/14          | 6/14          | ?        |

---

## 🔍 Key Observations from This Example

1. **Lowest probability** goes to the previously selected observation (Observation 2)
2. **Highest probability** goes to the observation with no overlap (Observation 3)
3. **Observation 1** has intermediate probability due to partial overlap
4. **Probabilities remain the same** for draw 2 and draw 3 because the overlap pattern doesn't change

---

## 🧠 Why This Pattern Occurs

- **Observation 2** (already selected) gets penalized for self-overlap
- **Observation 3** gets rewarded for having zero overlap with current sample  
- **Observation 1** gets moderate probability due to partial overlap at only one time period
- The method successfully **discourages redundancy** while allowing for sample diversity

---

## ✅ Summary

This numerical example demonstrates how sequential bootstrap:
- **Dynamically adjusts probabilities** based on overlap
- **Penalizes redundant sampling** of highly overlapping observations
- **Rewards unique observations** that add diversity to the sample
- Creates **more representative samples** for financial ML applications