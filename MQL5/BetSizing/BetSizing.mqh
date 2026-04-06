//+------------------------------------------------------------------+
//|                                                  BetSizing.mqh   |
//|  User-level orchestration functions for the AFML bet-sizing     |
//|  module. Mirrors the Python afml.bet_sizing.bet_sizing module.  |
//|                                                                  |
//|  Provides:                                                       |
//|    BetSizeProbability  — classifier probability + concurrency    |
//|    BetSizeDynamic      — forecast price divergence               |
//|    BetSizeBudget       — directional signals + capacity rule     |
//|    BetSizeReserve      — EF3M mixture CDF sizing                 |
//+------------------------------------------------------------------+
#property strict
#include "BetSizingUtils.mqh"
#include "EF3M.mqh"
#include "Ch10Snippets.mqh"

//+------------------------------------------------------------------+
//| BetSizeProbability                                               |
//|                                                                  |
//| Transforms classifier output into a confidence-weighted,        |
//| concurrency-corrected, discretized position size.               |
//|                                                                  |
//| Pipeline:                                                        |
//|   1. GetSignal     — z-score through normal CDF per observation  |
//|   2. AvgActiveSignals — average over concurrently active bets   |
//|      (only when average_active = true)                           |
//|   3. DiscreteSignal  — round to nearest step_size multiple      |
//|                                                                  |
//| Parameters                                                       |
//|   open_t[]      : Label open timestamps                          |
//|   close_t[]     : Label close timestamps (t1)                   |
//|   prob[]        : Classifier predicted probabilities             |
//|   pred[]        : Predicted sides (+1 long / -1 short)          |
//|   num_classes   : Number of outcome classes (2 = binary)        |
//|   step_size     : Discretization grid (0 = off)                 |
//|   average_active: true = apply concurrency correction           |
//|   query_time    : Current bar timestamp                          |
//|                                                                  |
//| Returns BetSizeResult with bet_size in [-1, 1].                 |
//+------------------------------------------------------------------+
BetSizeResult BetSizeProbability(
   const datetime &open_t[],
   const datetime &close_t[],
   const double   &prob[],
   const int      &pred[],
   int             num_classes,
   double          step_size,
   bool            average_active,
   datetime        query_time
   )
  {
   BetSizeResult result = {};
   result.bar_time = query_time;

   int n = ArraySize(prob);
   if(n == 0)
     {
      Print("BetSizeProbability: prob[] is empty.");
      return result;
     }
   if(ArraySize(close_t) != n || ArraySize(pred) != n ||
      ArraySize(open_t)  != n)
     {
      Print("BetSizeProbability: all input arrays must have the "
            "same length.");
      return result;
     }

   // --- Stage 1: per-signal z-score transformation ---
   double raw_signals[];
   ArrayResize(raw_signals, n);
   for(int i = 0; i < n; i++)
      raw_signals[i] = GetSignal(prob[i], num_classes, pred[i]);

   // Store the last observation's raw signal for diagnostics
   result.raw_signal = raw_signals[n - 1];

   // --- Stage 2: concurrency correction (optional) ---
   if(average_active)
     {
      datetime query_arr[1];
      query_arr[0] = query_time;
      double avg_arr[];
      AvgActiveSignals(open_t, close_t, raw_signals, query_arr, avg_arr);
      result.avg_signal = avg_arr[0];
     }
   else
     {
      // No averaging: use the most recent signal directly
      result.avg_signal = raw_signals[n - 1];
     }

   // --- Stage 3: discretization ---
   result.bet_size = DiscreteSignal(result.avg_signal, step_size);

   // Populate active counts via the sweep-line for diagnostics
   datetime query_arr2[1];
   query_arr2[0] = query_time;
   int al[], as_arr[];
   SweepLineActiveCounts(open_t, close_t, pred, query_arr2, al, as_arr);
   result.active_long  = al[0];
   result.active_short = as_arr[0];

   return result;
  }

//+------------------------------------------------------------------+
//| BetSizeDynamic                                                   |
//|                                                                  |
//| Sizes a position from the divergence between a forecast price   |
//| and the current market price, using a sigmoid or power          |
//| functional form calibrated to a target (divergence, bet_size)  |
//| pair. Also computes a target integer position and a limit price.|
//|                                                                  |
//| Parameters                                                       |
//|   current_pos    : Current open position (signed, e.g. lots*100)|
//|   max_pos        : Maximum allowed position size (same units)   |
//|   market_price   : Current mid price                             |
//|   forecast_price : Model forecast price                          |
//|   cal_divergence : Calibration divergence (pips, points, etc.)  |
//|   cal_bet_size   : Desired bet size at cal_divergence (0, 1)   |
//|   func           : "sigmoid" (default) or "power"               |
//|                                                                  |
//| Returns BetSizeResult with bet_size, t_pos, l_p populated.     |
//+------------------------------------------------------------------+
BetSizeResult BetSizeDynamic(
   double  current_pos,
   double  max_pos,
   double  market_price,
   double  forecast_price,
   double  cal_divergence,
   double  cal_bet_size,
   string  func = "sigmoid"
   )
  {
   BetSizeResult result = {};
   result.bar_time = TimeCurrent();

   if(max_pos <= 0.0)
     {
      Print("BetSizeDynamic: max_pos must be positive.");
      return result;
     }

   // Calibrate w from the (divergence, bet_size) target
   double w = GetW(cal_divergence, cal_bet_size, func);
   result.raw_signal = div;    // price divergence for diagnostics

   // Compute price divergence and bet size
   double div = forecast_price - market_price;
   double bsz;

   if(func == "sigmoid")
     {
      bsz = SigmoidBetSize(div, w);
     }
   else // power form requires normalized divergence
     {
      double norm_div = div / cal_divergence;
      norm_div = Clamp(norm_div, -1.0, 1.0);
      bsz = PowerBetSize(norm_div, w);
     }

   result.bet_size   = Clamp(bsz, -1.0, 1.0);
   result.avg_signal = result.bet_size; // no averaging stage here
   result.t_pos      = MathRound(result.bet_size * max_pos);

   // Compute the limit price
   result.l_p = LimitPrice(market_price, current_pos,
                            result.t_pos, max_pos, w, func);

   return result;
  }

//+------------------------------------------------------------------+
//| BetSizeDynamic — running-maximum state for BetSizeBudget.       |
//| Declared at file scope so they persist across OnTick calls.     |
//+------------------------------------------------------------------+
static int g_budget_max_long  = 1;
static int g_budget_max_short = 1;

//+------------------------------------------------------------------+
//| SeedBudgetMaxima                                                 |
//|                                                                  |
//| Call once from OnInit() to initialise the running maxima from   |
//| the full warm-up history rather than letting them grow from 1.  |
//|                                                                  |
//| Without seeding, the earliest bars of live trading will produce |
//| oversized positions until the true historical maxima are seen.  |
//+------------------------------------------------------------------+
void SeedBudgetMaxima(
   const datetime &open_t[],
   const datetime &close_t[],
   const int      &sides[]
   )
  {
   int n = ArraySize(open_t);
   if(n == 0) return;

   int al[], as_arr[];
   // Evaluate at every open time to find the true historical maxima
   SweepLineActiveCounts(open_t, close_t, sides, open_t, al, as_arr);
   int ml = 1, ms = 1;
   for(int i = 0; i < n; i++)
     {
      if(al[i]    > ml) ml = al[i];
      if(as_arr[i] > ms) ms = as_arr[i];
     }
   g_budget_max_long  = ml;
   g_budget_max_short = ms;
  }

//+------------------------------------------------------------------+
//| BetSizeBudget                                                    |
//|                                                                  |
//| Sizes positions from the normalized long-short imbalance of     |
//| concurrently active directional signals. No probability input   |
//| is required.                                                     |
//|                                                                  |
//|   bet_size = (active_long/max_long) - (active_short/max_short)  |
//|                                                                  |
//| The running maxima g_budget_max_long / g_budget_max_short are   |
//| updated on every call. Seed them first with SeedBudgetMaxima()  |
//| if historical data is available.                                 |
//|                                                                  |
//| Parameters                                                       |
//|   open_t[]   : Bet open timestamps                               |
//|   close_t[]  : Bet close timestamps (t1)                        |
//|   sides[]    : +1 long / -1 short                               |
//|   query_time : Current bar timestamp                             |
//+------------------------------------------------------------------+
BetSizeResult BetSizeBudget(
   const datetime &open_t[],
   const datetime &close_t[],
   const int      &sides[],
   datetime        query_time
   )
  {
   BetSizeResult result = {};
   result.bar_time = query_time;

   datetime query_arr[1];
   query_arr[0] = query_time;
   int al[], as_arr[];
   SweepLineActiveCounts(open_t, close_t, sides, query_arr, al, as_arr);

   result.active_long  = al[0];
   result.active_short = as_arr[0];

   // Update running maxima
   if(result.active_long  > g_budget_max_long)
      g_budget_max_long  = result.active_long;
   if(result.active_short > g_budget_max_short)
      g_budget_max_short = result.active_short;

   double frac_long  = (double)result.active_long  / g_budget_max_long;
   double frac_short = (double)result.active_short / g_budget_max_short;

   result.c_t        = frac_long - frac_short;
   result.bet_size   = Clamp(result.c_t, -1.0, 1.0);
   result.avg_signal = result.bet_size;

   return result;
  }

//+------------------------------------------------------------------+
//| BetSizeReserve                                                   |
//|                                                                  |
//| Sizes positions using the CDF of a mixture of two Gaussians     |
//| fitted to the empirical distribution of concurrent position     |
//| imbalance. The sizing curve shape is entirely data-driven.      |
//|                                                                  |
//| Workflow:                                                        |
//|   1. Call FitM2N() once in OnInit() from historical c_t data.  |
//|   2. Pass the resulting M2NParams here on each bar.             |
//|   3. The function computes c_t, evaluates MixtureCDF(c_t),     |
//|      and maps it to a bet size via the conditional CDF formula. |
//|                                                                  |
//| Minimum recommended history: ~500 bets for stable EF3M fit.    |
//|                                                                  |
//| Parameters                                                       |
//|   open_t[]       : Bet open timestamps                           |
//|   close_t[]      : Bet close timestamps (t1)                    |
//|   sides[]        : +1 long / -1 short                           |
//|   query_time     : Current bar timestamp                         |
//|   fitted_params  : M2NParams from FitM2N() called in OnInit()  |
//+------------------------------------------------------------------+
BetSizeResult BetSizeReserve(
   const datetime &open_t[],
   const datetime &close_t[],
   const int      &sides[],
   datetime        query_time,
   const M2NParams &fitted_params
   )
  {
   BetSizeResult result = {};
   result.bar_time = query_time;

   datetime query_arr[1];
   query_arr[0] = query_time;
   int al[], as_arr[];
   SweepLineActiveCounts(open_t, close_t, sides, query_arr, al, as_arr);

   result.active_long  = al[0];
   result.active_short = as_arr[0];
   result.c_t          = (double)(result.active_long - result.active_short);
   result.bet_size     = ReserveBetSize(result.c_t, fitted_params);
   result.avg_signal   = result.bet_size;

   return result;
  }
