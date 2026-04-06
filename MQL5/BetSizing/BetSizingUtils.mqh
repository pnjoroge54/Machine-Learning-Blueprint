//+------------------------------------------------------------------+
//|                                             BetSizingUtils.mqh   |
//|  Shared statistical utilities for the AFML bet-sizing module.    |
//|  Provides: NormCDF, NormICDF, NormPDF, RawMoments,              |
//|            SweepLineActiveCounts, BetSizeResult struct.          |
//+------------------------------------------------------------------+
#property strict

//+------------------------------------------------------------------+
//| Result struct returned by all four user-level sizing functions.  |
//+------------------------------------------------------------------+
struct BetSizeResult
  {
   double   bet_size;      // Signed position size in [-1, 1]
   double   t_pos;         // Target integer position (dynamic method)
   double   l_p;           // Limit price (dynamic method)
   double   raw_signal;    // Pre-averaging, pre-discretization signal
   double   avg_signal;    // Post-averaging, pre-discretization signal
   int      active_long;   // Concurrent active long bets at this bar
   int      active_short;  // Concurrent active short bets at this bar
   double   c_t;           // Long-short imbalance (budget/reserve)
   datetime bar_time;      // Bar open time this result corresponds to
  };

//+------------------------------------------------------------------+
//| Standard normal PDF                                              |
//+------------------------------------------------------------------+
double NormPDF(double x, double mu = 0.0, double sigma = 1.0)
  {
   if(sigma <= 0.0)
     {
      Print("NormPDF: sigma must be positive. Got: ", DoubleToString(sigma, 6));
      return 0.0;
     }
   double z = (x - mu) / sigma;
   return (1.0 / (sigma * MathSqrt(2.0 * M_PI))) * MathExp(-0.5 * z * z);
  }

//+------------------------------------------------------------------+
//| Standard normal CDF                                              |
//| Hart (1968) minimax rational approximation.                      |
//| |error| < 7.5e-8 across the full real line.                     |
//+------------------------------------------------------------------+
double NormCDF(double x)
  {
   double t    = 1.0 / (1.0 + 0.2316419 * MathAbs(x));
   double poly = t * ( 0.319381530
                + t * (-0.356563782
                + t * ( 1.781477937
                + t * (-1.821255978
                + t *   1.330274429))));
   double pdf  = (1.0 / MathSqrt(2.0 * M_PI)) * MathExp(-0.5 * x * x);
   double cdf  = 1.0 - pdf * poly;
   return (x >= 0.0) ? cdf : 1.0 - cdf;
  }

//+------------------------------------------------------------------+
//| Inverse standard normal CDF (quantile function)                 |
//| Beasley-Springer-Moro algorithm. Accurate to ~1e-7.             |
//+------------------------------------------------------------------+
double NormICDF(double p)
  {
   if(p <= 0.0) return -1e300;
   if(p >= 1.0) return  1e300;

   static const double a[4] = { 2.50662823884,
                                -18.61500062529,
                                 41.39119773534,
                                -25.44106049637 };
   static const double b[4] = { -8.47351093090,
                                 23.08336743743,
                                -21.06224101826,
                                  3.13082909833 };
   static const double c[9] = { 0.3374754822726147,
                                 0.9761690190917186,
                                 0.1607979714918209,
                                 0.0276438810333863,
                                 0.0038405729373609,
                                 0.0003951896511349,
                                 0.0000321767881768,
                                 0.0000002888167364,
                                 0.0000003960315187 };
   double y = p - 0.5;
   if(MathAbs(y) < 0.42)
     {
      double r = y * y;
      return y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) /
                 ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0);
     }
   double r = (y > 0.0) ? 1.0 - p : p;
   r = MathLog(-MathLog(r));
   double q = c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] +
              r*(c[5] + r*(c[6] + r*(c[7] + r*c[8])))))));
   return (y > 0.0) ? q : -q;
  }

//+------------------------------------------------------------------+
//| Compute the first n_moments raw moments of a data array.        |
//| m[k] = E[X^(k+1)], so m[0]=mean, m[1]=E[X^2], etc.            |
//+------------------------------------------------------------------+
void RawMoments(const double &data[], double &m[], int n_moments = 5)
  {
   ArrayResize(m, n_moments);
   ArrayInitialize(m, 0.0);
   int n = ArraySize(data);
   if(n == 0) return;
   for(int k = 0; k < n_moments; k++)
     {
      double sum = 0.0;
      for(int i = 0; i < n; i++)
         sum += MathPow(data[i], k + 1);
      m[k] = sum / n;
     }
  }

//+------------------------------------------------------------------+
//| Internal event struct for the sweep-line algorithm.             |
//+------------------------------------------------------------------+
struct SLEvent
  {
   datetime t;      // Event time
   int      delta;  // +1 = interval opens, -1 = interval closes
   int      side;   // +1 = long bet, -1 = short bet
  };

//+------------------------------------------------------------------+
//| Insertion sort for SLEvent arrays (adequate for N < 20000).     |
//+------------------------------------------------------------------+
void SortSLEvents(SLEvent &events[])
  {
   int n = ArraySize(events);
   for(int i = 1; i < n; i++)
     {
      SLEvent key = events[i];
      int j = i - 1;
      while(j >= 0 && events[j].t > key.t)
        {
         events[j+1] = events[j];
         j--;
        }
      events[j+1] = key;
     }
  }

//+------------------------------------------------------------------+
//| Sweep-line active-count for parallel arrays of open/close times.|
//|                                                                  |
//| Computes, at each query timestamp, the number of concurrently   |
//| active long and short bets whose intervals [open_t, close_t)    |
//| cover that timestamp. Runs in O(N log N) via a sorted event     |
//| list rather than the O(N^2) naive nested loop.                  |
//|                                                                  |
//| Parameters                                                       |
//|   open_t[]      : Bet open timestamps (one per bet)             |
//|   close_t[]     : Bet close timestamps / t1 (one per bet)       |
//|   sides[]       : +1 for long, -1 for short (one per bet)       |
//|   query_t[]     : Bar timestamps to evaluate counts at           |
//|   active_long[] : Output — active long count at each query time  |
//|   active_short[]: Output — active short count at each query time |
//+------------------------------------------------------------------+
void SweepLineActiveCounts(
   const datetime &open_t[],
   const datetime &close_t[],
   const int      &sides[],
   const datetime &query_t[],
   int            &active_long[],
   int            &active_short[]
   )
  {
   int n_bets  = ArraySize(open_t);
   int n_query = ArraySize(query_t);

   ArrayResize(active_long,  n_query);
   ArrayResize(active_short, n_query);
   ArrayInitialize(active_long,  0);
   ArrayInitialize(active_short, 0);

   if(n_bets == 0 || n_query == 0)
      return;

   // Build the event list: one open event and one close event per bet
   SLEvent events[];
   ArrayResize(events, 2 * n_bets);
   for(int i = 0; i < n_bets; i++)
     {
      events[2*i].t     = open_t[i];
      events[2*i].delta = 1;
      events[2*i].side  = sides[i];

      events[2*i+1].t     = close_t[i];
      events[2*i+1].delta = -1;
      events[2*i+1].side  = sides[i];
     }

   SortSLEvents(events);

   int n_events   = ArraySize(events);
   int long_cnt   = 0;
   int short_cnt  = 0;
   int ev_idx     = 0;

   for(int q = 0; q < n_query; q++)
     {
      // Consume all events with t <= query_t[q]
      while(ev_idx < n_events && events[ev_idx].t <= query_t[q])
        {
         if(events[ev_idx].side == 1)
            long_cnt  += events[ev_idx].delta;
         else
            short_cnt += events[ev_idx].delta;
         ev_idx++;
        }
      active_long[q]  = MathMax(0, long_cnt);
      active_short[q] = MathMax(0, short_cnt);
     }
  }

//+------------------------------------------------------------------+
//| Clamp a value to [lo, hi].                                      |
//+------------------------------------------------------------------+
double Clamp(double x, double lo, double hi)
  {
   return MathMax(lo, MathMin(hi, x));
  }

//+------------------------------------------------------------------+
//| Return the sign of x: +1, 0, or -1.                            |
//+------------------------------------------------------------------+
double MathSign(double x)
  {
   if(x > 0.0) return  1.0;
   if(x < 0.0) return -1.0;
   return 0.0;
  }
