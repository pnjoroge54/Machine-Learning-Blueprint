//+------------------------------------------------------------------+
//|                                                    BetSizing.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

/*
This MQL5 library provides functions for bet sizing, translated from
Marcos López de Prado's "Advances in Financial Machine Learning" Python implementations.

It includes:
1. BetSizeProbability: Calculates bet size from a model's probability (Snippet 10.1).
2. BetSizeBudget: Calculates bet size based on the ratio of concurrent active bets (Section 10.2).

As requested, this file uses the standard MQL5 "Stat" library (CStatistics) 
to replace Scipy.stats.norm.cdf, simplifying the implementation.
*/

#include <Math\Stat\Stat.mqh>

//+------------------------------------------------------------------+
//| 1. Bet Size from Probability (Snippet 10.1)                      |
//+------------------------------------------------------------------+
/*
Calculates the bet size based on the predicted probability (confidence) of a side.
This is a translation of `get_signal` from ch10_snippets.py.

Parameters:
   prob         - double: The model's probability for the predicted side (e.g., 0.7 for "long").
                        Must be between 0.0 and 1.0.
   num_classes  - int:    The number of possible outcomes (e.g., 2 for long/short, 3 for long/short/neutral).
   pred         - int:    The predicted side: 1 for long, -1 for short. If 0, will return the 
                        unsigned bet size (confidence only).
   step_size    - double: If > 0, discretizes the final bet size. E.g., 0.1 rounds to 0.0, 0.1, 0.2...
                        This implements `discrete_signal` from ch10_snippets.py.

Returns:
   double: The calculated bet size, from -1.0 to 1.0.
*/
double BetSizeProbability(double prob, int num_classes, int pred, double step_size = 0.0)
{
   // 1. Initialize CStatistics object to access statistical functions
   CStatistics stat;

   // 2. Clip probability to avoid log(0) or division by zero.
   // Same as `prob.clip(lower=1e-6, upper=1 - 1e-6)`
   double p = prob;
   if(p < 1e-6)
      p = 1e-6;
   if(p > 1.0 - 1e-6)
      p = 1.0 - 1e-6;

   // 3. Calculate z-score (from snippet 10.1)
   // bet_sizes = (prob - 1 / num_classes) / (prob * (1 - prob)) ** 0.5
   if(num_classes <= 0)
     {
      Print("BetSizeProbability: num_classes must be > 0");
      return 0.0;
     }
   double z = (p - 1.0 / num_classes) / MathSqrt(p * (1.0 - p));

   // 4. Calculate bet size using Normal CDF
   // bet_size = 2 * norm.cdf(z) - 1
   // We use the standard library's StatNormalCdf (mean=0, variance=1)
   double bet_size = 2.0 * stat.NormalCdf(z, 0.0, 1.0) - 1.0;

   // 5. Apply predicted side (if provided)
   // signal = side * size
   if(pred != 0)
     {
      bet_size = (double)pred * bet_size;
     }

   // 6. Discretize signal if step_size is provided (from snippet 10.3)
   if(step_size > 0.0)
     {
      // signal1 = (signal0 / step_size).round() * step_size
      bet_size = MathRound(bet_size / step_size) * step_size;

      // Cap/Floor
      // signal1[signal1 > 1] = 1
      // signal1[signal1 < -1] = -1
      bet_size = MathMin(1.0, bet_size);
      bet_size = MathMax(-1.0, bet_size);
     }

   return bet_size;
}

//+------------------------------------------------------------------+
//| 2. Bet Sizing from a Budget (Section 10.2)                       |
//+------------------------------------------------------------------+
/*
Calculates a bet size based on the number of concurrent active bets.
This implements the final calculation from `bet_size_budget`.

NOTE: The Python version calculates these counts from a historical DataFrame.
In MQL5, your Expert Advisor (EA) is responsible for tracking these
four values in real-time (e.g., using global variables or a class).
See the User Guide (BetSizing_UserGuide.md) for details.

Parameters:
   active_long        - int: The number of *currently* active long signals/positions.
   active_short       - int: The number of *currently* active short signals/positions.
   max_active_long    - int: The *historical maximum* number of concurrent long signals.
   max_active_short   - int: The *historical maximum* number of concurrent short signals.

Returns:
   double: The calculated bet size, from -1.0 to 1.0.
*/
double BetSizeBudget(int active_long, int active_short, int max_active_long, int max_active_short)
{
   // frac_active_long = events_1["active_long"] / active_long_max
   double frac_long = 0.0;
   if(max_active_long > 0)
     {
      frac_long = (double)active_long / (double)max_active_long;
     }

   // frac_active_short = events_1["active_short"] / active_short_max
   double frac_short = 0.0;
   if(max_active_short > 0)
     {
      frac_short = (double)active_short / (double)max_active_short;
     }

   // events_1["bet_size"] = frac_active_long - frac_active_short
   return frac_long - frac_short;
}

//+------------------------------------------------------------------+
//| 3. Dynamic Position Size (Snippet 10.4)                          |
//|    Sigmoid-based functions                                       |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Calculates 'w', the coefficient for the sigmoid function.        |
//| This is a calibration step, typically done once.                 |
//| from `get_w_sigmoid`                                             |
//+------------------------------------------------------------------+
double GetWSigmoid(double price_div, double m_bet_size)
{
   // w = (price_div**2) * ((m_bet_size ** (-2)) - 1)
   if(MathAbs(m_bet_size) >= 1.0)
     {
      Print("GetWSigmoid: m_bet_size must be between -1 and 1");
      return 0.0;
     }
   if(MathAbs(m_bet_size) < 1e-9) // Avoid division by zero
     {
       Print("GetWSigmoid: m_bet_size is too close to zero");
       return 0.0; // Or a very large number, depending on desired behavior
     }
     
   double m_inv_sq = 1.0 / (m_bet_size * m_bet_size);
   return (price_div * price_div) * (m_inv_sq - 1.0);
}

//+------------------------------------------------------------------+
//| Calculates the bet size based on price divergence.               |
//| from `bet_size_sigmoid`                                          |
//+------------------------------------------------------------------+
double BetSizeSigmoid(double w_param, double price_div)
{
   // return price_div * ((w_param + price_div**2) ** (-0.5))
   double den = w_param + (price_div * price_div);
   if(den <= 0)
     {
      Print("BetSizeSigmoid: Invalid denominator (w_param + price_div^2) <= 0");
      return 0.0;
     }
   return price_div / MathSqrt(den);
}

//+------------------------------------------------------------------+
//| Calculates the target position (integer).                        |
//| from `get_target_pos_sigmoid`                                    |
//+------------------------------------------------------------------+
double GetTargetPosSigmoid(double w_param, double forecast_price, double market_price, double max_pos)
{
   // return int(bet_size_sigmoid(w_param, forecast_price - market_price) * max_pos)
   double price_div = forecast_price - market_price;
   double bet_size = BetSizeSigmoid(w_param, price_div);
   return (double)((int)(bet_size * max_pos)); // Return as double for MQL5 compatibility, but value is floored
}

//+------------------------------------------------------------------+
//| Calculates the inverse price for a given bet size.               |
//| from `inv_price_sigmoid`                                         |
//+------------------------------------------------------------------+
double InvPriceSigmoid(double forecast_price, double w_param, double m_bet_size)
{
   // return forecast_price - m_bet_size * (w_param / (1 - m_bet_size**2)) ** 0.5
   double m_sq = m_bet_size * m_bet_size;
   if(m_sq >= 1.0)
     {
      Print("InvPriceSigmoid: bet size must be < 1.0");
      // Return forecast price as a neutral value
      return forecast_price;
     }
   double den = 1.0 - m_sq;
   return forecast_price - m_bet_size * MathSqrt(w_param / den);
}

//+------------------------------------------------------------------+
//| Calculates the limit price for a trade.                          |
//| from `limit_price_sigmoid`                                       |
//+------------------------------------------------------------------+
double LimitPriceSigmoid(double target_pos, double pos, double forecast_price, double w_param, double max_pos)
{
   if(target_pos == pos)
     {
      // No trade, return NaN or a marker. 
      // We return an impossible price (0) as a marker.
      // An EA should check `if(target_pos != pos)` before calling this.
      return 0.0; 
     }

   // sgn = np.sign(target_pos - pos)
   int sgn = (target_pos > pos) ? 1 : -1;
   
   double l_p = 0;
   
   // for j in range(abs(pos + sgn), abs(target_pos + 1)):
   //    l_p += inv_price_sigmoid(forecast_price, w_param, j / float(max_pos))
   
   int start = (int)MathAbs(pos + sgn);
   int end = (int)MathAbs(target_pos);
   
   for(int j = start; j <= end; j++)
     {
      l_p += InvPriceSigmoid(forecast_price, w_param, (double)j / max_pos);
     }

   // l_p = l_p / abs(target_pos - pos)
   if(MathAbs(target_pos - pos) == 0) return 0.0; // Should not happen
   
   return l_p / MathAbs(target_pos - pos);
}

//+------------------------------------------------------------------+

