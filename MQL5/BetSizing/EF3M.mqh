//+------------------------------------------------------------------+
//|                                                       EF3M.mqh   |
//|  Mixture-of-Gaussians parameter estimation via multi-start      |
//|  analytic moment-matching (EF3M-3 variant).                     |
//|                                                                  |
//|  Each random starting point is specified by (mu1, p1). The      |
//|  remaining three parameters (mu2, sigma1, sigma2) are derived   |
//|  analytically and exactly from the first three raw moments of   |
//|  the data, giving distinct component standard deviations with   |
//|  no iterative refinement required. Among all starting points    |
//|  that produce positive variances, the candidate with the        |
//|  highest log-likelihood on the data is returned.               |
//|                                                                  |
//|  Why multi-start is needed                                       |
//|  For a given (mu1, p1), the analytic solve is unique. But the   |
//|  log-likelihood surface over (mu1, p1) is non-convex and often  |
//|  multi-modal. Running n_runs random starts and selecting by     |
//|  log-likelihood is the standard strategy for global maximum     |
//|  search at modest computational cost.                           |
//|                                                                  |
//|  Reference:                                                      |
//|    Lopez de Prado, M. & Foreman, M. (2014). A mixture of two    |
//|    Gaussians approach to mathematical portfolio oversight:        |
//|    The EF3M algorithm. Quantitative Finance 14(5), 913-930.     |
//+------------------------------------------------------------------+
#property strict
#include "BetSizingUtils.mqh"

//+------------------------------------------------------------------+
//| Parameters of a two-Gaussian mixture.                           |
//| PDF: p1*N(mu1, s1) + (1-p1)*N(mu2, s2)                        |
//+------------------------------------------------------------------+
struct M2NParams
  {
   double mu1;             // Mean of component 1
   double mu2;             // Mean of component 2
   double s1;              // Std dev of component 1
   double s2;              // Std dev of component 2
   double p1;              // Mixing weight of component 1, in (0, 1)
   double log_likelihood;  // Log-likelihood of this fit on the data
  };

//+------------------------------------------------------------------+
//| Evaluate the mixture PDF at x.                                  |
//+------------------------------------------------------------------+
double MixturePDF(double x, const M2NParams &p)
  {
   return p.p1        * NormPDF(x, p.mu1, p.s1)
        + (1.0 - p.p1) * NormPDF(x, p.mu2, p.s2);
  }

//+------------------------------------------------------------------+
//| Evaluate the mixture CDF at x.                                  |
//+------------------------------------------------------------------+
double MixtureCDF(double x, const M2NParams &p)
  {
   return p.p1        * NormCDF((x - p.mu1) / p.s1)
        + (1.0 - p.p1) * NormCDF((x - p.mu2) / p.s2);
  }

//+------------------------------------------------------------------+
//| Log-likelihood of params on data[].                             |
//| Floored at log(1e-300) to avoid -Inf from zero-density points. |
//+------------------------------------------------------------------+
double LogLikelihood(const double &data[], const M2NParams &params)
  {
   int    n  = ArraySize(data);
   double ll = 0.0;
   for(int i = 0; i < n; i++)
     {
      double px = MixturePDF(data[i], params);
      ll += (px > 1e-300) ? MathLog(px) : MathLog(1e-300);
     }
   return ll;
  }

//+------------------------------------------------------------------+
//| Derive (mu2, sigma1, sigma2) from (mu1, p1) and the first      |
//| three raw moments using an exact closed-form linear solve.      |
//|                                                                  |
//| Raw moments of the mixture X = p1*N(mu1,s1) + (1-p1)*N(mu2,s2):|
//|   m1 = p1*mu1 + (1-p1)*mu2                                     |
//|   m2 = p1*(mu1^2+s1^2) + (1-p1)*(mu2^2+s2^2)                  |
//|   m3 = p1*(mu1^3+3*mu1*s1^2) + (1-p1)*(mu2^3+3*mu2*s2^2)      |
//|                                                                  |
//| Step 1 — mu2 from m1 (unique given mu1, p1).                   |
//| Step 2 — rearrange m2 and m3 into a 2x2 linear system          |
//|   for the unknowns (s1^2, s2^2):                               |
//|                                                                  |
//|   A = p1*s1^2 + (1-p1)*s2^2          (from m2)                |
//|   B = p1*mu1*s1^2 + (1-p1)*mu2*s2^2  (from m3)                |
//|                                                                  |
//|   Matrix form:                                                   |
//|   [ p1       (1-p1)   ] [ s1^2 ]   [ A ]                       |
//|   [ p1*mu1   (1-p1)*mu2] [ s2^2 ] = [ B ]                       |
//|                                                                  |
//|   det = p1*(1-p1)*(mu2-mu1)    (non-zero iff mu1 != mu2)       |
//|   s1^2 = (A*mu2 - B) / (p1*(mu2-mu1))                         |
//|   s2^2 = (B - A*mu1) / ((1-p1)*(mu2-mu1))                     |
//|                                                                  |
//| Returns false when:                                             |
//|   - p1 is outside (0, 1)                                        |
//|   - mu1 == mu2 (degenerate single-component case)               |
//|   - implied s1^2 or s2^2 is non-positive (the starting point   |
//|     (mu1, p1) is incompatible with the sample moments)          |
//+------------------------------------------------------------------+
bool DeriveComponentParams(double mu1, double p1,
                            const double &m[],
                            double &mu2, double &s1, double &s2)
  {
   if(p1 <= 0.0 || p1 >= 1.0) return false;

   // --- Step 1: mu2 from the first raw moment ---
   mu2 = (m[0] - p1 * mu1) / (1.0 - p1);

   // Degenerate: both components at the same location.
   // The system is rank-deficient when mu1 == mu2.
   double gap = mu2 - mu1;
   if(MathAbs(gap) < 1e-12) return false;

   // --- Step 2: s1^2, s2^2 from moments 2 and 3 ---
   //
   // A = p1*s1^2 + (1-p1)*s2^2
   //   = m2 - p1*mu1^2 - (1-p1)*mu2^2
   double A = m[1] - p1*(mu1*mu1) - (1.0-p1)*(mu2*mu2);

   // B = p1*mu1*s1^2 + (1-p1)*mu2*s2^2
   //   (derived from the raw third moment of N(mu,sigma):
   //    E[X^3] = mu^3 + 3*mu*sigma^2)
   double B = (m[2] - p1*(mu1*mu1*mu1) - (1.0-p1)*(mu2*mu2*mu2)) / 3.0;

   // Cramer's rule: det = p1*(1-p1)*(mu2-mu1) = p1*(1-p1)*gap
   double var1 = (A*mu2 - B) / (p1 * gap);
   double var2 = (B - A*mu1) / ((1.0-p1) * gap);

   if(var1 <= 0.0 || var2 <= 0.0) return false;

   s1 = MathSqrt(var1);
   s2 = MathSqrt(var2);
   return true;
  }

//+------------------------------------------------------------------+
//| Multi-start analytic EF3M-3 fit.                               |
//|                                                                  |
//| Algorithm:                                                       |
//|   1. Compute the first three raw moments of data[].             |
//|   2. For each of n_runs random (mu1, p1) starting points:      |
//|      a. Derive (mu2, s1, s2) analytically via DeriveComponentParams.|
//|      b. If the derived variances are positive, evaluate the      |
//|         log-likelihood of the resulting mixture on data[].       |
//|   3. Return the candidate with the highest log-likelihood.      |
//|                                                                  |
//| Parameters                                                       |
//|   data[]  : Empirical sample of c_t imbalance values            |
//|   n_runs  : Number of random starting points (100 recommended)  |
//|                                                                  |
//| Data requirements                                               |
//|   Minimum 30 observations for stable moment estimates.          |
//|   500+ observations recommended for a reliable mixture fit.     |
//|   Data must span both positive and negative values for the      |
//|   two-component structure to be identifiable.                   |
//|                                                                  |
//| Returns a zeroed M2NParams with log_likelihood = -1e18 if no   |
//| starting point produces valid positive variances.              |
//+------------------------------------------------------------------+
M2NParams FitM2N(const double &data[], int n_runs = 100)
  {
   M2NParams best = {};
   best.log_likelihood = -1e18;

   int n = ArraySize(data);
   if(n < 30)
     {
      Print("FitM2N: need at least 30 observations. Got: ", n);
      return best;
     }

   // Compute the first three raw moments: m[0]=E[X], m[1]=E[X^2], m[2]=E[X^3]
   double m[];
   RawMoments(data, m, 3);

   // Estimate the standard deviation for scaling the mu1 search range.
   // Var[X] = E[X^2] - (E[X])^2 = m[1] - m[0]^2.
   double var_est  = MathMax(1e-12, m[1] - m[0]*m[0]);
   double sigma_est = MathSqrt(var_est);

   MathSrand((int)TimeCurrent());

   for(int r = 0; r < n_runs; r++)
     {
      // mu1: uniform in (m1 - 2*sigma, m1 + 2*sigma)
      // p1:  uniform in (0.1, 0.9)
      double noise    = 2.0 * sigma_est *
                        ((double)(MathRand() - 16383) / 16383.0);
      double mu1_init = m[0] + noise;
      double p1_init  = 0.1 + 0.8 * ((double)MathRand() / 32767.0);

      double mu2, s1, s2;
      if(!DeriveComponentParams(mu1_init, p1_init, m, mu2, s1, s2))
         continue; // starting point incompatible with sample moments

      M2NParams candidate;
      candidate.mu1 = mu1_init;
      candidate.mu2 = mu2;
      candidate.s1  = s1;
      candidate.s2  = s2;
      candidate.p1  = p1_init;
      candidate.log_likelihood = LogLikelihood(data, candidate);

      if(candidate.log_likelihood > best.log_likelihood)
         best = candidate;
     }

   if(best.log_likelihood <= -1e17)
      Print("FitM2N: no valid candidate found. "
            "Check for extreme outliers or consider increasing n_runs.");

   return best;
  }

//+------------------------------------------------------------------+
//| Convert a concurrent-imbalance value c_t to a bet size in      |
//| [-1, 1] using the fitted mixture CDF.                           |
//|                                                                  |
//| Formula (AFML Ch. 10):                                          |
//|   c_t >= 0: bet = (F(c_t) - F(0)) / (1 - F(0))                |
//|   c_t <  0: bet = (F(c_t) - F(0)) / F(0)                      |
//|                                                                  |
//| The result is zero when c_t = 0 and approaches +/-1 as c_t    |
//| becomes extreme relative to the fitted distribution.            |
//|                                                                  |
//| Note: c_t is the raw integer imbalance (active_long -          |
//| active_short), not the normalized fraction used by BetSizeBudget.|
//| The EF3M model must be fitted to the same raw c_t series.      |
//+------------------------------------------------------------------+
double ReserveBetSize(double c_t, const M2NParams &p)
  {
   double F0 = MixtureCDF(0.0, p);
   double Fx = MixtureCDF(c_t, p);

   if(c_t >= 0.0)
     {
      double denom = 1.0 - F0;
      if(denom < 1e-10) return 1.0;
      return Clamp((Fx - F0) / denom, 0.0, 1.0);
     }
   else
     {
      if(F0 < 1e-10) return -1.0;
      return Clamp((Fx - F0) / F0, -1.0, 0.0);
     }
  }
