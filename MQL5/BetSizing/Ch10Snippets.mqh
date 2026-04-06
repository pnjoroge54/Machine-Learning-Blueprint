//+------------------------------------------------------------------+
//|                                               Ch10Snippets.mqh   |
//|  Low-level implementations corresponding to AFML Chapter 10    |
//|  Snippets 10.1 – 10.4.                                          |
//|                                                                  |
//|  Provides: GetSignal, AvgActiveSignals, DiscreteSignal,         |
//|            SigmoidBetSize, PowerBetSize, GetW, LimitPrice.      |
//+------------------------------------------------------------------+
#property strict
#include "BetSizingUtils.mqh"

//+------------------------------------------------------------------+
//| SNIPPET 10.1                                                     |
//| Transform a classifier's predicted probability to a signed bet  |
//| size via a z-score through the standard normal CDF.             |
//|                                                                  |
//| Parameters                                                       |
//|   prob        : Predicted probability for the positive class     |
//|   num_classes : Number of outcome classes (2 = binary)          |
//|   pred        : +1 (long), -1 (short), 0 = return magnitude    |
//|                                                                  |
//| Returns a value in [-1, 1].                                     |
//| Zero when prob == 1/num_classes (no edge).                      |
//+------------------------------------------------------------------+
double GetSignal(double prob, int num_classes, int pred)
  {
   if(num_classes < 2)
     {
      Print("GetSignal: num_classes must be >= 2. Got: ", num_classes);
      return 0.0;
     }
   prob = Clamp(prob, 1e-10, 1.0 - 1e-10);

   double base_rate = 1.0 / num_classes;
   double denom     = MathSqrt(prob * (1.0 - prob));

   // Edge case: near-certain prediction => denom collapses
   if(denom < 1e-10)
      return (prob > base_rate) ? 1.0 : -1.0;

   double z      = (prob - base_rate) / denom;
   double signal = 2.0 * NormCDF(z) - 1.0;

   if(pred != 0)
      signal *= (pred > 0) ? 1.0 : -1.0;

   return Clamp(signal, -1.0, 1.0);
  }

//+------------------------------------------------------------------+
//| SNIPPET 10.2                                                     |
//| Compute the time-averaged bet size across all concurrently      |
//| active signals at each query timestamp.                          |
//|                                                                  |
//| At each query time, the function computes the arithmetic mean   |
//| of all signal values whose holding period [open_t, close_t)    |
//| covers that timestamp. This is the concurrency correction that  |
//| prevents exposure from growing proportionally to signal density.|
//|                                                                  |
//| Parameters                                                       |
//|   open_t[]  : Signal start times (one per observation)          |
//|   close_t[] : Signal end times / t1 (one per observation)       |
//|   signals[] : Per-signal bet sizes from GetSignal               |
//|   query_t[] : Bar timestamps to evaluate the average at         |
//|   avg_out[] : Output — averaged signal at each query time       |
//+------------------------------------------------------------------+
void AvgActiveSignals(
   const datetime &open_t[],
   const datetime &close_t[],
   const double   &signals[],
   const datetime &query_t[],
   double         &avg_out[]
   )
  {
   int n_sig   = ArraySize(open_t);
   int n_query = ArraySize(query_t);

   ArrayResize(avg_out, n_query);
   ArrayInitialize(avg_out, 0.0);

   for(int q = 0; q < n_query; q++)
     {
      double sum   = 0.0;
      int    count = 0;
      for(int i = 0; i < n_sig; i++)
        {
         // Include signal i if query_t[q] falls in [open_t[i], close_t[i])
         if(open_t[i] <= query_t[q] && query_t[q] < close_t[i])
           {
            sum += signals[i];
            count++;
           }
        }
      avg_out[q] = (count > 0) ? sum / count : 0.0;
     }
  }

//+------------------------------------------------------------------+
//| SNIPPET 10.3                                                     |
//| Discretize a continuous signal to multiples of step_size.      |
//|                                                                  |
//| Prevents micro-adjustments whose transaction cost exceeds their |
//| expected P&L contribution. A step_size of 0.05 means positions |
//| only change in 5% increments of the maximum.                    |
//|                                                                  |
//| Parameters                                                       |
//|   signal    : Continuous signal in [-1, 1]                      |
//|   step_size : Grid spacing (0 = no discretization)              |
//|                                                                  |
//| Returns the nearest grid point, capped to [-1, 1].             |
//+------------------------------------------------------------------+
double DiscreteSignal(double signal, double step_size)
  {
   if(step_size <= 0.0)
      return Clamp(signal, -1.0, 1.0);

   double s = MathRound(signal / step_size) * step_size;
   return Clamp(s, -1.0, 1.0);
  }

//+------------------------------------------------------------------+
//| SNIPPET 10.4a                                                    |
//| Sigmoid bet-size function.                                      |
//|                                                                  |
//|   bet_size = price_div / sqrt(w + price_div^2)                 |
//|                                                                  |
//| Properties:                                                      |
//|   - Zero at price_div = 0                                       |
//|   - Bounded in (-1, +1)                                         |
//|   - Odd-symmetric: f(-x) = -f(x)                               |
//|   - Smaller w => steeper function (more aggressive sizing)      |
//|   - Always produces an S-shaped curve regardless of w           |
//|                                                                  |
//| Parameters                                                       |
//|   price_div : forecast_price - market_price                     |
//|   w         : calibration parameter > 0                         |
//+------------------------------------------------------------------+
double SigmoidBetSize(double price_div, double w)
  {
   if(w < 0.0)
     {
      Print("SigmoidBetSize: w must be non-negative. Got: ",
            DoubleToString(w, 6));
      return 0.0;
     }
   double denom = MathSqrt(w + price_div * price_div);
   if(denom < 1e-15) return MathSign(price_div);
   return price_div / denom;
  }

//+------------------------------------------------------------------+
//| SNIPPET 10.4b                                                    |
//| Power bet-size function.                                        |
//|                                                                  |
//|   bet_size = sign(price_div) * |price_div|^w                   |
//|                                                                  |
//| price_div MUST be pre-normalized to [-1, 1].                   |
//|                                                                  |
//| Properties:                                                      |
//|   - w < 1: concave — rises steeply near zero (momentum fit)    |
//|   - w = 1: exactly linear                                       |
//|   - w > 1: convex — suppressed near zero (mean-reversion fit)  |
//|                                                                  |
//| Parameters                                                       |
//|   price_div : Normalized divergence in [-1, 1]                 |
//|   w         : Shape exponent > 0                                |
//+------------------------------------------------------------------+
double PowerBetSize(double price_div, double w)
  {
   if(w <= 0.0)
     {
      Print("PowerBetSize: w must be positive. Got: ",
            DoubleToString(w, 6));
      return 0.0;
     }
   if(MathAbs(price_div) > 1.0)
     {
      Print("PowerBetSize: price_div must be in [-1, 1]. Got: ",
            DoubleToString(price_div, 6),
            ". Clamping.");
      price_div = Clamp(price_div, -1.0, 1.0);
     }
   if(price_div == 0.0) return 0.0;
   return MathSign(price_div) * MathPow(MathAbs(price_div), w);
  }

//+------------------------------------------------------------------+
//| Calibrate w from a (divergence, target_bet_size) pair.         |
//|                                                                  |
//| Answers the question: "I want a bet size of cal_bet_size when  |
//| the price divergence is cal_divergence. What is w?"            |
//|                                                                  |
//| Sigmoid inversion (analytic):                                   |
//|   m = x/sqrt(w+x^2)  =>  w = x^2*(1-m^2)/m^2                 |
//|                                                                  |
//| Power inversion (analytic):                                     |
//|   m = |x|^w  =>  w = log(|m|) / log(|x|)                     |
//|                                                                  |
//| Parameters                                                       |
//|   cal_divergence : Target price divergence (pips, points, etc.) |
//|   cal_bet_size   : Desired bet size at cal_divergence (0, 1)   |
//|   func           : "sigmoid" or "power"                         |
//+------------------------------------------------------------------+
double GetW(double cal_divergence, double cal_bet_size,
            string func = "sigmoid")
  {
   cal_bet_size = MathAbs(cal_bet_size);
   if(cal_bet_size <= 0.0 || cal_bet_size >= 1.0)
     {
      Print("GetW: cal_bet_size must be strictly in (0, 1). Got: ",
            DoubleToString(cal_bet_size, 6));
      return 1.0;
     }

   if(func == "sigmoid")
     {
      double m2 = cal_bet_size * cal_bet_size;
      double x2 = cal_divergence * cal_divergence;
      return x2 * (1.0 - m2) / m2;
     }
   else // power
     {
      double ax = MathAbs(cal_divergence);
      if(ax <= 0.0 || ax >= 1.0)
        {
         Print("GetW power: cal_divergence must be in (0, 1) for the "
               "power form (price_div is normalized). Got: ",
               DoubleToString(ax, 6));
         return 1.0;
        }
      return MathLog(cal_bet_size) / MathLog(ax);
     }
  }

//+------------------------------------------------------------------+
//| Compute the limit price for moving from pos_curr to pos_target. |
//|                                                                  |
//| For each discrete position unit from pos_curr+1 to pos_target,  |
//| inverts the bet-size function to find the divergence that        |
//| justifies that unit, then returns the average implied price.    |
//|                                                                  |
//| This is the average break-even execution price: placing limit   |
//| orders at or better than l_p is economically rational given     |
//| the forecast.                                                    |
//|                                                                  |
//| Parameters                                                       |
//|   market_price : Current mid price                               |
//|   pos_curr     : Current signed position                         |
//|   pos_target   : Target signed position                          |
//|   max_pos      : Maximum allowed position size                   |
//|   w            : Calibrated w parameter                          |
//|   func         : "sigmoid" or "power"                            |
//+------------------------------------------------------------------+
double LimitPrice(double market_price, double pos_curr,
                  double pos_target,   double max_pos,
                  double w,            string func)
  {
   if(max_pos <= 0.0)
     {
      Print("LimitPrice: max_pos must be positive.");
      return market_price;
     }

   double sgn   = (pos_target > pos_curr) ? 1.0 : -1.0;
   int    steps = (int)MathRound(MathAbs(pos_target - pos_curr));
   if(steps == 0) return market_price;

   double sum = 0.0;
   for(int k = 1; k <= steps; k++)
     {
      // Normalized bet size for this unit
      double m     = sgn * (MathAbs(pos_curr) + k) / max_pos;
      m            = Clamp(m, -1.0 + 1e-10, 1.0 - 1e-10);
      double div_k;

      if(func == "sigmoid")
        {
         // Invert m = x/sqrt(w+x^2) => x = sign(m)*sqrt(w)*|m|/sqrt(1-m^2)
         double m2 = m * m;
         if(1.0 - m2 < 1e-15)
            div_k = sgn * 1e6; // effectively infinite divergence
         else
            div_k = MathSign(m) * MathSqrt(w) * MathAbs(m) /
                    MathSqrt(1.0 - m2);
        }
      else // power: m = sign(x)*|x|^w => x = sign(m)*|m|^(1/w)
        {
         div_k = MathSign(m) * MathPow(MathAbs(m), 1.0 / w);
        }

      sum += div_k;
     }

   return market_price + sum / steps;
  }
