//+------------------------------------------------------------------+
//|                                              BetSizingEA.mq5    |
//|  Complete Expert Advisor wiring the AFML bet-sizing module to   |
//|  a classifier-driven signal pipeline.                            |
//|                                                                  |
//|  Supports all four sizing methods selectable at runtime via     |
//|  the InpMethod input parameter.                                  |
//|                                                                  |
//|  Depends on:                                                     |
//|    BetSizingUtils.mqh  (utilities and structs)                  |
//|    EF3M.mqh            (mixture-of-Gaussians fitting)           |
//|    Ch10Snippets.mqh    (low-level snippet implementations)       |
//|    BetSizing.mqh       (user-level orchestration functions)      |
//+------------------------------------------------------------------+
#property strict
#property description "AFML Chapter 10 Bet Sizing EA"
#property version     "1.00"

#include "BetSizing.mqh"   // pulls in Ch10Snippets, EF3M, and BetSizingUtils

//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+

// --- Method selection ---
enum ENUM_SIZING_METHOD
  {
   METHOD_PROBABILITY = 0, // Probability-based (classifier output)
   METHOD_DYNAMIC     = 1, // Dynamic (forecast price divergence)
   METHOD_BUDGET      = 2, // Budget-constrained (directional only)
   METHOD_RESERVE     = 3  // Reserve (EF3M mixture CDF)
  };

input ENUM_SIZING_METHOD InpMethod        = METHOD_PROBABILITY;

// --- Shared ---
input double   InpMaxLots      = 1.0;    // Maximum position size in lots
input double   InpStepSize     = 0.05;   // Discretization grid (0=off)

// --- Probability method ---
input bool     InpAvgActive    = true;   // Correct for label concurrency
input int      InpNumClasses   = 2;      // 2 = binary classifier

// --- Dynamic method ---
input double   InpCalDivergence = 10.0;  // Calibration divergence (pips)
input double   InpCalBetSize    = 0.95;  // Target bet size at InpCalDivergence
input string   InpDynFunc       = "sigmoid"; // "sigmoid" or "power"

// --- Reserve method ---
input int      InpEF3MRuns     = 100;    // EF3M random restarts in OnInit

// --- Risk ---
input double   InpMinLots      = 0.01;   // Minimum lot size (below = flat)
input double   InpSlippage     = 3;      // Max slippage in points

//+------------------------------------------------------------------+
//| Global state                                                     |
//+------------------------------------------------------------------+

// Classifier output arrays — in a real EA these are populated by
// your signal generation and classifier inference logic.
// Here they are initialised with synthetic data in OnInit() for
// demonstration purposes.
datetime g_open_t[];
datetime g_close_t[];
double   g_prob[];
int      g_pred[];
int      g_sides[];

// Pre-fitted EF3M parameters (computed once in OnInit)
M2NParams g_reserve_params;

// Track the previous bar time to detect new bars
datetime g_last_bar_time = 0;

// Current open position in lots (signed: + long, - short)
double g_current_pos = 0.0;

//+------------------------------------------------------------------+
//| Forward declarations                                             |
//+------------------------------------------------------------------+
void     LoadHistoricalSignals();
void     AppendNewSignal(datetime bar_open, datetime bar_close,
                          double prob, int pred, int side);
double   GetForecastPrice();
void     AdjustPosition(const BetSizeResult &r);
bool     IsNewBar();
void     PrintDiagnostics(const BetSizeResult &r);
void     SyncCurrentPosition();

//+------------------------------------------------------------------+
//| Expert initialisation                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("BetSizingEA: initialising. Method=", (int)InpMethod);

   // 1. Load historical signals from persistent storage or
   //    regenerate from the classifier pipeline.
   //    This stub populates the arrays with synthetic data so the
   //    EA compiles and runs out of the box. Replace with your own
   //    classifier inference call.
   LoadHistoricalSignals();

   int n_hist = ArraySize(g_open_t);
   Print("BetSizingEA: loaded ", n_hist, " historical signals.");

   // 2. Seed the budget method's running maxima from history so
   //    the first live bars are not oversized.
   if(InpMethod == METHOD_BUDGET)
      SeedBudgetMaxima(g_open_t, g_close_t, g_sides);

   // 3. Fit the EF3M mixture parameters for the reserve method.
   //    This can take 1-3 seconds for large histories; it only
   //    runs once in OnInit.
   if(InpMethod == METHOD_RESERVE)
     {
      if(n_hist < 50)
        {
         Print("BetSizingEA: need at least 50 historical bets for "
               "EF3M. Consider warming up with METHOD_BUDGET first.");
         return INIT_FAILED;
        }

      // Build the c_t series from the historical data
      int al[], as_arr[];
      SweepLineActiveCounts(g_open_t, g_close_t, g_sides,
                            g_open_t, al, as_arr);
      double ct_series[];
      ArrayResize(ct_series, n_hist);
      for(int i = 0; i < n_hist; i++)
         ct_series[i] = (double)(al[i] - as_arr[i]);

      g_reserve_params = FitM2N(ct_series, InpEF3MRuns);
      PrintFormat("BetSizingEA: EF3M fit — "
                  "mu1=%.4f mu2=%.4f s1=%.4f s2=%.4f p1=%.4f LL=%.2f",
                  g_reserve_params.mu1, g_reserve_params.mu2,
                  g_reserve_params.s1,  g_reserve_params.s2,
                  g_reserve_params.p1,  g_reserve_params.log_likelihood);
     }

   // 4. Sync current open position from the broker
   SyncCurrentPosition();

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Print("BetSizingEA: deinitialising. Reason=", reason);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Only process logic on new bar opens to avoid micro-adjustments
   if(!IsNewBar()) return;

   datetime bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);

   // Sync the current position state from the broker
   SyncCurrentPosition();

   // --- Get new classifier output for this bar ---
   // In a real EA: call your inference pipeline here to get
   // prob, pred, side for the newly closed bar (index 1).
   // These synthetic values simulate a classifier with moderate
   // confidence returning a long signal.
   double new_prob  = 0.62 + 0.10 * MathSin((double)bar_time / 86400.0);
   int    new_pred  = (new_prob > 0.5) ? 1 : -1;
   int    new_side  = new_pred;
   datetime bar_close = bar_time + PeriodSeconds(PERIOD_CURRENT) * 10;
   AppendNewSignal(bar_time, bar_close, new_prob, new_pred, new_side);

   // --- Compute bet size ---
   BetSizeResult r = {};
   double mid_price = (SymbolInfoDouble(_Symbol, SYMBOL_BID) +
                       SymbolInfoDouble(_Symbol, SYMBOL_ASK)) * 0.5;

   switch(InpMethod)
     {
      case METHOD_PROBABILITY:
         r = BetSizeProbability(
               g_open_t, g_close_t, g_prob, g_pred,
               InpNumClasses, InpStepSize, InpAvgActive, bar_time);
         break;

      case METHOD_DYNAMIC:
         r = BetSizeDynamic(
               g_current_pos,
               InpMaxLots * 100.0,    // express max_pos in 0.01-lot units
               mid_price,
               GetForecastPrice(),
               InpCalDivergence * _Point,
               InpCalBetSize,
               InpDynFunc);
         break;

      case METHOD_BUDGET:
         r = BetSizeBudget(g_open_t, g_close_t, g_sides, bar_time);
         break;

      case METHOD_RESERVE:
         r = BetSizeReserve(g_open_t, g_close_t, g_sides,
                            bar_time, g_reserve_params);
         break;
     }

   PrintDiagnostics(r);

   // --- Adjust position ---
   AdjustPosition(r);

   g_last_bar_time = bar_time;
  }

//+------------------------------------------------------------------+
//| Load (or simulate) historical classifier signals.               |
//|                                                                  |
//| Replace this stub with your own persistent signal storage or    |
//| classifier pipeline call. The stub generates 200 synthetic      |
//| bets with random probabilities over the past 200 days so the   |
//| EA is immediately testable in the Strategy Tester.              |
//+------------------------------------------------------------------+
void LoadHistoricalSignals()
  {
   int n = 200;
   ArrayResize(g_open_t,  n);
   ArrayResize(g_close_t, n);
   ArrayResize(g_prob,    n);
   ArrayResize(g_pred,    n);
   ArrayResize(g_sides,   n);

   datetime base = TimeCurrent() - (datetime)(n * 86400);
   MathSrand(42);

   for(int i = 0; i < n; i++)
     {
      g_open_t[i]  = base + (datetime)(i * 86400);
      // Holding period between 5 and 15 bars (days here)
      int hold      = 5 + MathRand() % 11;
      g_close_t[i]  = g_open_t[i] + (datetime)(hold * 86400);
      g_prob[i]     = 0.50 + 0.40 * ((double)MathRand() / 32767.0);
      g_pred[i]     = (MathRand() % 2 == 0) ? 1 : -1;
      g_sides[i]    = g_pred[i];
     }
  }

//+------------------------------------------------------------------+
//| Append a new observation to the signal arrays.                  |
//| Called on each new bar after classifier inference.              |
//+------------------------------------------------------------------+
void AppendNewSignal(datetime bar_open, datetime bar_close,
                      double prob, int pred, int side)
  {
   int n = ArraySize(g_open_t);
   ArrayResize(g_open_t,  n + 1);
   ArrayResize(g_close_t, n + 1);
   ArrayResize(g_prob,    n + 1);
   ArrayResize(g_pred,    n + 1);
   ArrayResize(g_sides,   n + 1);

   g_open_t[n]  = bar_open;
   g_close_t[n] = bar_close;
   g_prob[n]    = prob;
   g_pred[n]    = pred;
   g_sides[n]   = side;
  }

//+------------------------------------------------------------------+
//| Return a synthetic forecast price for the dynamic method.       |
//| In a real EA: return the output of your regression model.      |
//+------------------------------------------------------------------+
double GetForecastPrice()
  {
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   // Simulate a small positive forecast divergence (5 pips)
   return bid + 5.0 * _Point;
  }

//+------------------------------------------------------------------+
//| Translate bet_size in [-1,1] to an order and submit it.        |
//|                                                                  |
//| Logic:                                                           |
//|   1. Compute target lots from bet_size * InpMaxLots.            |
//|   2. Determine direction from the sign of bet_size.             |
//|   3. Compare with current open position.                        |
//|   4. If the change is >= InpMinLots, close the old position     |
//|      and open a new one (or just go flat if target = 0).        |
//|                                                                  |
//| For the dynamic method, r.l_p provides a limit price; here we  |
//| use market orders for simplicity and include l_p in the log.   |
//+------------------------------------------------------------------+
void AdjustPosition(const BetSizeResult &r)
  {
   double target_lots = NormalizeDouble(InpMaxLots * MathAbs(r.bet_size),
                                        2);
   if(target_lots < InpMinLots)
      target_lots = 0.0;

   int target_dir = (r.bet_size > 0.0) ?  1 :
                    (r.bet_size < 0.0) ? -1 : 0;
   int current_dir = (g_current_pos > 0.0) ?  1 :
                     (g_current_pos < 0.0) ? -1 : 0;

   bool direction_change = (target_dir != current_dir);
   double lot_change     = MathAbs(target_lots - MathAbs(g_current_pos));

   // Do nothing if direction is unchanged and the lot change is
   // below the minimum — this is the execution-layer discretization
   // equivalent of DiscreteSignal
   if(!direction_change && lot_change < InpMinLots)
     {
      Print("AdjustPosition: change below minimum lot threshold. "
            "No order sent.");
      return;
     }

   // Close any existing position first
   if(MathAbs(g_current_pos) >= InpMinLots)
     {
      MqlTradeRequest close_req = {};
      MqlTradeResult  close_res = {};

      close_req.action   = TRADE_ACTION_DEAL;
      close_req.symbol   = _Symbol;
      close_req.volume   = NormalizeDouble(MathAbs(g_current_pos), 2);
      close_req.type     = (g_current_pos > 0.0) ? ORDER_TYPE_SELL
                                                  : ORDER_TYPE_BUY;
      close_req.price    = (close_req.type == ORDER_TYPE_SELL)
                           ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                           : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      close_req.deviation = (int)InpSlippage;
      close_req.magic    = 202410;
      close_req.comment  = "BetSizing close";
      close_req.type_filling = ORDER_FILLING_IOC;

      if(!OrderSend(close_req, close_res))
         Print("AdjustPosition: close failed — ",
               close_res.retcode, " ", close_res.comment);
      else
         Print("AdjustPosition: closed ", g_current_pos, " lots.");
     }

   // Open new position if target is non-zero
   if(target_lots >= InpMinLots && target_dir != 0)
     {
      MqlTradeRequest open_req = {};
      MqlTradeResult  open_res = {};

      open_req.action   = TRADE_ACTION_DEAL;
      open_req.symbol   = _Symbol;
      open_req.volume   = target_lots;
      open_req.type     = (target_dir == 1) ? ORDER_TYPE_BUY
                                             : ORDER_TYPE_SELL;
      open_req.price    = (open_req.type == ORDER_TYPE_BUY)
                          ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                          : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      open_req.deviation = (int)InpSlippage;
      open_req.magic    = 202410;
      open_req.comment  = "BetSizing open";
      open_req.type_filling = ORDER_FILLING_IOC;

      // Log the limit price from the dynamic method for reference
      if(InpMethod == METHOD_DYNAMIC && r.l_p > 0.0)
         PrintFormat("AdjustPosition: limit price reference = %.5f "
                     "(not enforced; using market order)", r.l_p);

      if(!OrderSend(open_req, open_res))
         Print("AdjustPosition: open failed — ",
               open_res.retcode, " ", open_res.comment);
      else
        {
         g_current_pos = target_lots * target_dir;
         PrintFormat("AdjustPosition: opened %.2f lots, dir=%d",
                     target_lots, target_dir);
        }
     }
   else
     {
      g_current_pos = 0.0;
      Print("AdjustPosition: target is flat.");
     }
  }

//+------------------------------------------------------------------+
//| Return true once per bar (on the first tick of each new bar).  |
//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime current_bar = iTime(_Symbol, PERIOD_CURRENT, 0);
   return (current_bar != g_last_bar_time);
  }

//+------------------------------------------------------------------+
//| Sync g_current_pos from the broker's open position state.      |
//+------------------------------------------------------------------+
void SyncCurrentPosition()
  {
   g_current_pos = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)  != 202410)  continue;

      double vol  = PositionGetDouble(POSITION_VOLUME);
      int    type = (int)PositionGetInteger(POSITION_TYPE);
      g_current_pos += (type == POSITION_TYPE_BUY) ? vol : -vol;
     }
  }

//+------------------------------------------------------------------+
//| Print a one-line diagnostic for each sizing decision.           |
//+------------------------------------------------------------------+
void PrintDiagnostics(const BetSizeResult &r)
  {
   PrintFormat(
      "[%s] method=%d bet=%.4f raw=%.4f avg=%.4f "
      "aL=%d aS=%d c_t=%.2f t_pos=%.0f l_p=%.5f",
      TimeToString(r.bar_time, TIME_DATE | TIME_MINUTES),
      (int)InpMethod,
      r.bet_size,
      r.raw_signal,
      r.avg_signal,
      r.active_long,
      r.active_short,
      r.c_t,
      r.t_pos,
      r.l_p
   );
  }
