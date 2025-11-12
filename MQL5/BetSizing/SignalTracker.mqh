//+------------------------------------------------------------------+
//|                                                SignalTracker.mqh |
//|                        Copyright 2025, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

/*
This file provides the CSignalTracker class.

Its purpose is to be a single, reusable object in your EA that tracks
all the necessary state for the bet sizing functions in BetSizing.mqh.

Simply tell it when signals open or close, and it will manage:
- Current active long/short counts
- Historical maximum long/short counts
- A history of c_t (active_long - active_short) for reserve sizing
- The current net position (optional)
*/

#include <Arrays\ArrayDouble.mqh>
#include <Math\Stat\Stat.mqh>
#include "BetSizing.mqh"

//+------------------------------------------------------------------+
//| Class CSignalTracker                                             |
//+------------------------------------------------------------------+
class CSignalTracker
  {
private:
   // --- State for BetSizeBudget ---
   int               m_active_long;
   int               m_active_short;
   int               m_max_long;
   int               m_max_short;

   // --- State for BetSizeReserve ---
   CArrayDouble      m_ct_history; // History of (active_long - active_short)
   CStatistics       m_stat;       // For calculating mean/stddev

   // --- State for BetSizeDynamic ---
   double            m_current_position; // Current net position (lots, contracts)

   // --- Private Helper ---
   void              UpdateCtHistory()
     {
      m_ct_history.Add(m_active_long - m_active_short);
     }

public:
   // --- Constructor ---
   void              CSignalTracker(void)
     {
      m_active_long = 0;
      m_active_short = 0;
      m_max_long = 0;
      m_max_short = 0;
      m_current_position = 0.0;
      m_ct_history.FreeMode(false); // We want a dynamic array
      m_ct_history.Clear();
     }
   // --- Destructor ---
   void             ~CSignalTracker(void)
     {
      m_ct_history.Clear();
     }

   // --- 1. STATE MODIFIERS (Call these from your EA) ---

   // Call this when your model generates a NEW signal
   void              OnSignalNew(int side) // side = 1 (long) or -1 (short)
     {
      if(side == 1)
        {
         m_active_long++;
         if(m_active_long > m_max_long)
            m_max_long = m_active_long;
        }
      else if(side == -1)
        {
         m_active_short++;
         if(m_active_short > m_max_short)
            m_max_short = m_active_short;
        }
      
      // Update the history for reserve sizing
      UpdateCtHistory();
     }

   // Call this when a signal EXPIRES or its position is closed
   void              OnSignalClose(int side) // side = 1 (long) or -1 (short)
     {
      if(side == 1)
        {
         m_active_long--;
         if(m_active_long < 0) m_active_long = 0; // Don't go below zero
        }
      else if(side == -1)
        {
         m_active_short--;
         if(m_active_short < 0) m_active_short = 0; // Don't go below zero
        }
      
      // Update the history for reserve sizing
      UpdateCtHistory();
     }

   // Call this to keep the tracker updated on your actual net position
   void              SetCurrentPosition(double pos)
     {
      m_current_position = pos;
     }

   // --- 2. GETTERS (Access the raw state) ---
   int               GetActiveLong(void){ return m_active_long; }
   int               GetActiveShort(void){ return m_active_short; }
   int               GetMaxLong(void){ return m_max_long; }
   int               GetMaxShort(void){ return m_max_short; }
   double            GetCurrentPosition(void){ return m_current_position; }
   int               GetCtHistoryTotal(void){ return m_ct_history.Total(); }

   // --- 3. CALCULATION METHODS (Get bet sizes) ---

   // --- Gets bet size from Budget (Section 10.2) ---
   double            GetBetSizeBudget(void)
     {
      // This function uses the tracker's internal state
      return BetSizeBudget(m_active_long, m_active_short, m_max_long, m_max_short);
     }

   // --- Gets bet size from Reserve (Section 10.4.c) ---
   // This is a practical MQL5 adaptation using a Z-Score, as the
   // EF3M Gaussian Mixture fitting is non-trivial.
   // This gives a *similar* sigmoid sizing based on the distribution.
   double            GetBetSizeReserve(void)
     {
      int total = m_ct_history.Total();
      if(total < 2) // Need at least 2 data points for stddev
        {
         Print("GetBetSizeReserve: Not enough c_t history to calculate. (Total=" + (string)total + ")");
         return 0.0;
        }
        
      double c_t = m_active_long - m_active_short;
      
      // Get mean and stddev from our history
      double mean = m_ct_history.Mean();
      double stddev = m_ct_history.StdDev();

      if(stddev == 0)
        {
         // History is flat, no variance
         return (c_t > mean) ? 1.0 : (c_t < mean) ? -1.0 : 0.0;
        }

      // Z-score the current c_t
      double z = (c_t - mean) / stddev;

      // Use the Normal CDF to get a bet size from -1.0 to 1.0
      // This is the same logic as BetSizeProbability
      return 2.0 * m_stat.NormalCdf(z, 0.0, 1.0) - 1.0;
     }
     
   // --- Helper for averaging active signals (Snippet 10.2) ---
   // NOTE: Your EA is responsible for tracking the list/array
   // of active signal *values*.
   double            GetAverageSignal(CArrayDouble *active_signal_values)
     {
      if(active_signal_values.Total() == 0)
         return 0.0;
         
      return active_signal_values.Mean();
     }
  };
//+------------------------------------------------------------------+
