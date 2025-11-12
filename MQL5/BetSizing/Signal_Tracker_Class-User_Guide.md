# **Signal Tracker Class \- User Guide**

You have created BetSizing.mqh (which holds the *calculations*) and SignalTracker.mqh (which holds the *state*). This guide explains how to use them together in your Expert Advisor (EA).

The CSignalTracker class is the "brain" of your EA. You just need to tell it when your signals start and stop, and it will handle all the complex state tracking for you.

## **1\. Setup in Your EA**

At the top of your EA's .mq5 file, include the new tracker class. **Make sure BetSizing.mqh is in the same folder or in your main MQL5/Include folder.**

//+------------------------------------------------------------------+  
//|                                                   MyStrategy.mq5 |  
//+------------------------------------------------------------------+  
\#include \<Trade\\Trade.mqh\>  
\#include "SignalTracker.mqh" // Include the new class

// \--- Global Variables \---  
CTrade            trade;  
CSignalTracker    g\_tracker; // Create one global instance of the tracker

// \--- EA Inputs \---  
input double  MaxPositionSize \= 10.0; // Max lots  
//... your other inputs

## **2\. How to Use the CSignalTracker**

Your EA's main job is to tell the g\_tracker object what is happening.

### **OnSignalNew(int side)**

Call this **once** when your model generates a new signal.

### **OnSignalClose(int side)**

Call this **once** when that same signal's *event* ends (e.g., its t1 is reached, a position is closed, or an opposite signal arrives).

### **SetCurrentPosition(double pos)**

Call this to update the tracker with your *actual* net position. This is crucial for bet\_size\_dynamic functions.

## **3\. Usage Examples**

Here is how you would use the tracker to implement each bet sizing strategy.

### **Example 1: BetSizeBudget**

This is the simplest. The tracker handles everything.

void OnNewSignalDetected(int side) // side=1 or \-1  
{  
   // 1\. Tell the tracker  
   g\_tracker.OnSignalNew(side);  
     
   // 2\. Get the new bet size  
   double bet\_size \= g\_tracker.GetBetSizeBudget();  
     
   // 3\. Calculate volume and trade  
   double volume \= bet\_size \* MaxPositionSize;  
   // ... execute trade ...  
}

void OnSignalExpired(int side) // side=1 or \-1  
{  
   // 1\. Tell the tracker  
   g\_tracker.OnSignalClose(side);  
}

### **Example 2: BetSizeReserve**

This is just as simple from the EA's perspective. The tracker manages the c\_t history internally.

void OnNewSignalDetected(int side) // side=1 or \-1  
{  
   // 1\. Tell the tracker  
   g\_tracker.OnSignalNew(side);  
     
   // 2\. Get the new bet size  
   // This will return 0.0 until it has enough history  
   double bet\_size \= g\_tracker.GetBetSizeReserve();  
     
   if(g\_tracker.GetCtHistoryTotal() \> 20\) // Wait for some history  
   {  
      // 3\. Calculate volume and trade  
      double volume \= bet\_size \* MaxPositionSize;  
      // ... execute trade ...  
   }  
}

// Remember to call g\_tracker.OnSignalClose(side) when it expires\!

### **Example 3: BetSizeDynamic (Sigmoid)**

This workflow is more involved, as it combines the tracker's *state* with *parameters* you provide.

Step A: Calibration (Do this once, e.g., in OnInit)  
You need to find your w\_param. Let's say you calibrate that a price divergence of 0.00500 (500 points) should result in a 0.95 bet size.  
double g\_w\_param \= 0.0;

int OnInit()  
{  
   // ...  
   double cal\_divergence \= 0.00500; // e.g., 500 points  
   double cal\_bet\_size \= 0.95;  
     
   g\_w\_param \= GetWSigmoid(cal\_divergence, cal\_bet\_size);  
   Print("w\_param calibrated to: ", g\_w\_param);  
   // ...  
   return(INIT\_SUCCEEDED);  
}

Step B: Real-Time Usage (e.g., in OnTick)  
In OnTick, you get new prices and model forecasts.  
void OnTick()  
{  
   // 1\. Get real-time data  
   double market\_price \= SymbolInfoDouble(\_Symbol, SYMBOL\_ASK);  
   double forecast\_price \= GetMyModelForecast(); // Your magic function  
     
   // 2\. Update tracker's state  
   // (You need logic to get your \*actual\* net position)  
   double current\_net\_pos \= GetCurrentNetPosition();   
   g\_tracker.SetCurrentPosition(current\_net\_pos);

   // 3\. Get the Target Position from BetSizing.mqh  
   double target\_pos \= GetTargetPosSigmoid(  
      g\_w\_param,  
      forecast\_price,  
      market\_price,  
      MaxPositionSize  
   );  
     
   // 4\. Get the current position from the tracker  
   double current\_pos \= g\_tracker.GetCurrentPosition();  
     
   // 5\. Decide to trade  
   if(target\_pos \!= current\_pos)  
   {  
      // 6\. Get the Limit Price from BetSizing.mqh  
      double limit\_price \= LimitPriceSigmoid(  
         target\_pos,  
         current\_pos,  
         forecast\_price,  
         g\_w\_param,  
         MaxPositionSize  
      );  
        
      double volume\_to\_trade \= target\_pos \- current\_pos;  
        
      Print("New trade\! Target: ", target\_pos, ", Current: ", current\_pos);  
      Print("Placing order for ", volume\_to\_trade, " lots at limit ", limit\_price);  
        
      // ... execute limit order ...  
        
      // NOTE: You do NOT call OnSignalNew/Close here.  
      // This system is position-based, not signal-event-based.  
      // You just update the current position state.  
   }  
}  
