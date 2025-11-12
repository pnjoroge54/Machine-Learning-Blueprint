# Bet Sizing Library - User Guide

This guide explains the MQL5 functions BetSizeProbability and BetSizeBudget and how to use them within your Expert Advisor (EA).

## 1. BetSizeProbability

This function calculates a bet size based on your model's confidence (probability). It's a direct translation of Snippet 10.1 (get_signal) and 10.3 (discrete_signal).

```cpp
double BetSizeProbability(  
   double prob,  
   int num_classes,  
   int pred,  
   double step_size = 0.0  
);
```

**Parameters:**

```text
* prob (double): The probability from your model for the predicted side. For a binary (long/short) model, this would be the probability of the winning class.  
* num_classes (int): The number of possible outcomes. For a long/short model, this is 2. If you also predict "neutral" or "exit", it would be 3.  
* pred (int): The predicted side. Use 1 for long, -1 for short. If you pass 0, the function will return the raw, unsigned confidence (a value from 0.0 to 1.0).  
* step_size (double): (Optional) If you want to discretize your bet size, set this. For example, 0.1 will round the final bet size to the nearest 0.1 (e.g., 0.67 -> 0.7).
```

### Example Usage (in an EA)

Imagine you have a machine learning model that gives you a probability and a side.

```cpp
#include <BetSizing.mqh>

// ... inside your OnTick() or signal generation function ...

void OnTick()  
{  
   // --- Your Model's Output ---  
   // (This is just an example, you get this from your model)  
   double model_probability = 0.75; // 75% confidence  
   int    model_prediction = 1;     // 1 = long  
     
   // --- Calculate Bet Size ---  
   int    num_classes = 2; // We are predicting long vs. short  
   double step_size = 0.05; // Round to nearest 5%  
     
   double bet_size = BetSizeProbability(  
      model_probability,  
      num_classes,  
      model_prediction,  
      step_size  
   );  
     
   // bet_size will be a value like 0.55 (it's z-scored, CDF'd, and discretized)  
   Print("Model Prob: ", model_probability, ", Side: ", model_prediction, ", Calculated Bet Size: ", bet_size);

   // You can now use this bet_size to calculate your trade volume  
   // double trade_volume = bet_size * max_account_risk_lots;  
   // ... execute trade ...  
}
```

### Note on average_active

The Python bet_size_probability function has an average_active=True parameter. This feature is **not** implemented in this MQL5 function.

* **Reason:** That parameter relies on batch-processing a complete pandas.DataFrame of *all* historical signals.  
* **MQL5 Alternative:** An MQL5 EA runs in real-time. To average signals, you would need to store recent signals (e.g., in a CArrayDouble) and calculate a moving average yourself, *within your EA's state*. This is a different (and more complex) logic than what the Python function does.

## 2. BetSizeBudget

This function calculates a bet size based on the *proportion* of currently active long vs. short signals, relative to their historical maximums. This implements the "bet sizing from a budget" concept.

### The Python (Batch) vs. MQL5 (Stateful) Concept

* **Python:** The bet_size_budget Python function takes a full history of events, calculates the number of concurrent bets at *every single point in time*, finds the *overall maximums*, and then calculates the bet size for all points.  
* **MQL5:** Your EA runs *now*. It doesn't have a "full history" in the same way. **Your EA is responsible for tracking the state.**

You must create and update variables in your EA to track:

1. current_active_long: How many long signals are "on" right now.  
2. current_active_short: How many short signals are "on" right now.  
3. historical_max_long: The highest current_active_long has *ever* reached.  
4. historical_max_short: The highest current_active_short has *ever* reached.

The BetSizeBudget function performs the final, simple calculation using these four numbers.

```cpp
double BetSizeBudget(  
   int active_long,  
   int active_short,  
   int max_active_long,  
   int max_active_short  
);
```

**Parameters:**

```text
* active_long (int): Your EA's count of *current* active long signals.  
* active_short (int): Your EA's count of *current* active short signals.  
* max_active_long (int): Your EA's *historical maximum* count of concurrent long signals.  
* max_active_short (int): Your EA's *historical maximum* count of concurrent short signals.
```

#### Example Usage (Conceptual EA)

This is a simplified example of how you might manage this state in an EA.

```cpp
#include <BetSizing.mqh>

// --- Global variables to track our EA's state ---  
// (Note: For a robust EA, you would wrap this in a class)  
int g_active_long  = 0;  
int g_active_short = 0;  
int g_max_long     = 0;  
int g_max_short    = 0;

// (You would also need a list/array of your active signals/positions  
// to know when they expire or close)

//+------------------------------------------------------------------+  
//| A function you call when your model generates a NEW signal       |  
//+------------------------------------------------------------------+  
void OnNewSignal(int side) // side = 1 for long, -1 for short  
{  
   if(side == 1)  
     {  
      g_active_long++;  
      // Update historical max  
      if(g_active_long > g_max_long)  
         g_max_long = g_active_long;  
     }  
   else if(side == -1)  
     {  
      g_active_short++;  
      // Update historical max  
      if(g_active_short > g_max_short)  
         g_max_short = g_active_short;  
     }  
       
   // Store this signal and its expiry time (t1) somewhere...  
}

//+------------------------------------------------------------------+  
//| A function you call when an OLD signal expires or a position closes |  
//+------------------------------------------------------------------+  
void OnSignalClose(int side) // side = 1 for long, -1 for short  
{  
   if(side == 1)  
     {  
      g_active_long--;  
      if(g_active_long < 0) g_active_long = 0; // Sanity check  
     }  
   else if(side == -1)  
     {  
      g_active_short--;  
      if(g_active_short < 0) g_active_short = 0; // Sanity check  
     }  
}

//+------------------------------------------------------------------+  
//| Your main trading logic function                                 |  
//+------------------------------------------------------------------+  
void ExecuteStrategy()  
{  
   // ***  
   // 1. First, check your list of active signals/positions.  
   //    Call OnSignalClose() for any that have expired or been closed.  
   // ***  
     
   // ***  
   // 2. Second, run your model to get new signals.  
   //    If you get a new signal:  
   //       OnNewSignal(new_signal_side);  
   // ***

   // 3. Now, get the bet size for any *new* trades  
   double new_bet_size = BetSizeBudget(  
      g_active_long,  
      g_active_short,  
      g_max_long,  
      g_max_short  
   );

   Print("Active L/S: ", g_active_long, "/", g_active_short,   
         " | Max L/S: ", g_max_long, "/", g_max_short,  
         " | New Bet Size: ", new_bet_size);  
           
   // You can now use new_bet_size to place a new trade  
   // ...  
}  
```
