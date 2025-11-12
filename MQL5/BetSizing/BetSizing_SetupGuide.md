# Bet Sizing Library - Setup Guide

This guide explains how to include and use the BetSizing.mqh library in your MQL5 Expert Advisor (EA) or script.

## 1. File Placement

1. **Save the File:** Save the BetSizing.mqh file into your MQL5 Include directory.  
   The standard path is:

   ```text
   C:\Users\[Your_Username]\AppData\Roaming\MetaQuotes\Terminal\[Terminal_ID]\MQL5\Include\
   ```

   You can also place it in a subdirectory, for example:

   ```text
   MQL5\Include\MyLibraries\BetSizing.mqh  
   ```

2. **Find Your Data Folder:** You can find this path easily in MetaEditor:  
   * Go to Tools > Open MQL5 Data Folder.  
   * Navigate into the MQL5 folder, then the Include folder.

## 2. Including the Library

Once the file is in place, you can include it at the top of your EA's .mq5 file (e.g., MyStrategy.mq5).

```cpp
//+------------------------------------------------------------------+  
//|                                                   MyStrategy.mq5 |  
//+------------------------------------------------------------------+  
#property copyright "Your Name"  
#property version   "1.00"

//--- Include the Bet Sizing Library \---  
// If you placed it directly in MQL5\Include\  
#include <BetSizing.mqh>

// If you placed it in MQL5\Include\MyLibraries\  
#include <MyLibraries\BetSizing.mqh>

// ... other includes like Trade.mqh ...  
#include <Trade\Trade.mqh>

//--- EA Inputs ---  
input int    magic_number = 12345;  
input double lot_size_limit = 1.0;  
// ...
```

## 3. Required Standard Libraries

The BetSizing.mqh file automatically includes the necessary MQL5 Standard Library for statistics:

```cpp
#include <Math\Stat\Stat.mqh>
```

You do **not** need to include this file again in your main EA, as BetSizing.mqh already handles it. The AlgLib library was not required for these specific functions, as the MQL5 Stat library provided all the needed functionality, simplifying the process as requested.

That's it! You are now ready to call the functions _BetSizeProbability()_ and _BetSizeBudget()_ from within your EA's code. See the **BetSizing_UserGuide.md** for usage examples.
