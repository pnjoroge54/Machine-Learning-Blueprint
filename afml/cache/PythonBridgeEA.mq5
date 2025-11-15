//+------------------------------------------------------------------+
//|                                               PythonBridgeEA.mq5 |
//|                                     MQL5 Client for AFML Bridge  |
//|                               Copyright 2024, Patrick M. Njoroge |
//|                  https://www.mql5.com/en/users/patricknjoroge743 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Patrick M. Njoroge"
#property link "https://www.mql5.com/en/users/patricknjoroge743"
#property version "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string PythonHost = "http://127.0.0.1"; // Python server host
input int PythonPort = 80;                    // Python server port
input double RiskPercent = 1.0;               // Risk per trade (%)
input int MagicNumber = 12345;                // EA magic number
input bool EnableTrading = true;              // Enable actual trading
input int HeartbeatInterval = 5000;           // Heartbeat interval (ms)
input int DataSendInterval = 1000;            // Market data send interval (ms)

//--- Global variables
int socket_handle = INVALID_HANDLE;
CTrade trade;
datetime last_heartbeat = 0;
datetime last_data_send = 0;
bool is_connected = false;

//--- Message buffer
uchar message_buffer[];
int buffer_size = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Setup trade object
    trade.SetExpertMagicNumber(MagicNumber);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_FOK);

    // Initialize buffer
    ArrayResize(message_buffer, 0);

    // Connect to Python bridge
    if (!ConnectToPython())
    {
        Print("Failed to connect to Python bridge - will retry");
    }

    Print("Python Bridge EA initialized successfully");
    return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    DisconnectFromPython();
    Print("Python Bridge EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
{
    static datetime last_reconnect = 0;

    // Check connection
    if (!is_connected)
    {
        // Try reconnect every 5 seconds (instead of 10)
        if (TimeCurrent() - last_reconnect >= 5)
        {
            Print("Attempting to reconnect to Python...");
            ConnectToPython();
            last_reconnect = TimeCurrent();
        }
        return;
    }

    // Send heartbeat
    if (TimeCurrent() - last_heartbeat >= HeartbeatInterval / 1000)
    {
        SendHeartbeat();
        last_heartbeat = TimeCurrent();
    }

    // Send market data
    if (TimeCurrent() - last_data_send >= DataSendInterval / 1000)
    {
        SendMarketData();
        last_data_send = TimeCurrent();
    }

    // Check for incoming signals
    ProcessIncomingMessages();
}

//+------------------------------------------------------------------+
//| Connect to Python bridge server                                 |
//+------------------------------------------------------------------+
bool ConnectToPython()
{
    if (socket_handle != INVALID_HANDLE)
    {
        SocketClose(socket_handle);
    }

    // Create socket
    socket_handle = SocketCreate();
    if (socket_handle == INVALID_HANDLE)
    {
        Print("Failed to create socket: ", GetLastError());
        return false;
    }

    // Connect to server
    if (!SocketConnect(socket_handle, PythonHost, PythonPort, 5000))
    {
        Print("Failed to connect to Python server: ", GetLastError());
        SocketClose(socket_handle);
        socket_handle = INVALID_HANDLE;
        return false;
    }

    is_connected = true;
    Print("Connected to Python bridge at ", PythonHost, ":", PythonPort);
    Print("Socket handle: ", socket_handle);
    Print("Testing initial communication...");

    // Test the connection immediately
    SendHeartbeat();

    // Request any pending signals
    RequestPendingSignals();

    return true;
}

//+------------------------------------------------------------------+
//| Disconnect from Python bridge                                   |
//+------------------------------------------------------------------+
void DisconnectFromPython()
{
    if (socket_handle != INVALID_HANDLE)
    {
        SocketClose(socket_handle);
        socket_handle = INVALID_HANDLE;
    }
    is_connected = false;
}

//+------------------------------------------------------------------+
//| Send JSON message to Python with length prefix                  |
//+------------------------------------------------------------------+
bool SendMessage(string json_message)
{
    if (socket_handle == INVALID_HANDLE || !is_connected)
    {
        return false;
    }

    // Convert string to UTF-8 byte array
    uchar data[];
    int str_len = StringToCharArray(json_message, data, 0, WHOLE_ARRAY, CP_UTF8);

    // Remove null terminator if present
    if (str_len > 0 && data[str_len - 1] == 0)
        str_len--;

    ArrayResize(data, str_len);

    // Create length prefix (4 bytes, little-endian)
    uint length = str_len;
    uchar length_bytes[4];
    length_bytes[0] = (uchar)(length & 0xFF);
    length_bytes[1] = (uchar)((length >> 8) & 0xFF);
    length_bytes[2] = (uchar)((length >> 16) & 0xFF);
    length_bytes[3] = (uchar)((length >> 24) & 0xFF);

    // Send length prefix
    if (SocketSend(socket_handle, length_bytes, 4) != 4)
    {
        Print("Failed to send length prefix: ", GetLastError());
        is_connected = false;
        return false;
    }

    // Send data
    if (SocketSend(socket_handle, data, str_len) != str_len)
    {
        Print("Failed to send message data: ", GetLastError());
        is_connected = false;
        return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Build JSON string manually (simple implementation)              |
//+------------------------------------------------------------------+
string BuildJsonString(string type, string additional_fields = "")
{
    string json = "{";
    json += "\"type\":\"" + type + "\"";
    json += ",\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + "\"";

    if (additional_fields != "")
    {
        json += "," + additional_fields;
    }

    json += "}";

    return json;
}

//+------------------------------------------------------------------+
//| Send heartbeat to Python                                        |
//+------------------------------------------------------------------+
void SendHeartbeat()
{
    string json = BuildJsonString("heartbeat");
    SendMessage(json);
}

//+------------------------------------------------------------------+
//| Send market data to Python                                      |
//+------------------------------------------------------------------+
void SendMarketData()
{
    MqlTick tick;
    if (!SymbolInfoTick(_Symbol, tick))
    {
        return;
    }

    // Build fields
    string fields = "";
    fields += "\"symbol\":\"" + _Symbol + "\"";
    fields += ",\"bid\":" + DoubleToString(tick.bid, _Digits);
    fields += ",\"ask\":" + DoubleToString(tick.ask, _Digits);
    fields += ",\"volume\":" + DoubleToString((double)tick.volume, 0);

    double spread = (tick.ask - tick.bid) / _Point;
    fields += ",\"spread\":" + DoubleToString(spread, 1);

    // Add recent bars
    MqlRates rates[];
    int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, 20, rates);
    if (copied > 0)
    {
        fields += ",\"bars\":[";
        for (int i = 0; i < copied; i++)
        {
            if (i > 0)
                fields += ",";

            fields += "{";
            fields += "\"time\":\"" + TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS) + "\"";
            fields += ",\"open\":" + DoubleToString(rates[i].open, _Digits);
            fields += ",\"high\":" + DoubleToString(rates[i].high, _Digits);
            fields += ",\"low\":" + DoubleToString(rates[i].low, _Digits);
            fields += ",\"close\":" + DoubleToString(rates[i].close, _Digits);
            fields += ",\"volume\":" + DoubleToString((double)rates[i].tick_volume, 0);
            fields += "}";
        }
        fields += "]";
    }

    string json = BuildJsonString("market_data", fields);
    SendMessage(json);
}

//+------------------------------------------------------------------+
//| Request pending signals from Python                             |
//+------------------------------------------------------------------+
void RequestPendingSignals()
{
    string json = BuildJsonString("request_signals");
    SendMessage(json);
}

//+------------------------------------------------------------------+
//| Process incoming messages from Python                           |
//+------------------------------------------------------------------+
void ProcessIncomingMessages()
{
    if (socket_handle == INVALID_HANDLE || !is_connected)
    {
        return;
    }

    // Read available data
    uchar temp_buffer[4096];
    int received = SocketRead(socket_handle, temp_buffer, 4096, 0);

    if (received == -1)
    {
        int error = GetLastError();
        if (error == 4014)
        {
            // ⚠️ THIS IS NORMAL - no data available yet
            return; // Just return, don't disconnect!
        }
        else if (error != 0) // Other errors
        {
            Print("Socket read error: ", error);
            is_connected = false;
        }
        return;
    }

    if (received == 0)
    {
        return;
    }

    // Append to buffer
    int old_size = buffer_size;
    buffer_size += received;
    ArrayResize(message_buffer, buffer_size);
    ArrayCopy(message_buffer, temp_buffer, old_size, 0, received);

    // Process complete messages
    while (buffer_size >= 4)
    {
        // Read length prefix (little-endian)
        uint message_length =
            message_buffer[0] |
            (message_buffer[1] << 8) |
            (message_buffer[2] << 16) |
            (message_buffer[3] << 24);

        // Check if we have complete message
        if (buffer_size < (int)(4 + message_length))
        {
            break; // Wait for more data
        }

        // Extract message
        uchar msg_data[];
        ArrayResize(msg_data, (int)message_length);
        ArrayCopy(msg_data, message_buffer, 0, 4, (int)message_length);

        string message = CharArrayToString(msg_data, 0, (int)message_length, CP_UTF8);

        // Remove processed message from buffer
        int remaining = buffer_size - (4 + (int)message_length);
        if (remaining > 0)
        {
            uchar temp[];
            ArrayResize(temp, remaining);
            ArrayCopy(temp, message_buffer, 0, 4 + (int)message_length, remaining);
            ArrayResize(message_buffer, remaining);
            ArrayCopy(message_buffer, temp, 0, 0, remaining);
            buffer_size = remaining;
        }
        else
        {
            ArrayResize(message_buffer, 0);
            buffer_size = 0;
        }

        // Process message
        ProcessMessage(message);
    }
}

//+------------------------------------------------------------------+
//| Process single JSON message from Python                         |
//+------------------------------------------------------------------+
void ProcessMessage(string json_string)
{
    // Simple JSON parsing for our specific format
    string msg_type = ExtractJsonString(json_string, "type");

    if (msg_type == "signal")
    {
        ProcessSignal(json_string);
    }
    else if (msg_type == "heartbeat_response")
    {
        // Connection alive - no action needed
    }
    else
    {
        Print("Unknown message type: ", msg_type);
    }
}

//+------------------------------------------------------------------+
//| Extract string value from simple JSON                           |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
    string search = "\"" + key + "\":\"";
    int start = StringFind(json, search);
    if (start == -1)
        return "";

    start += StringLen(search);
    int end = StringFind(json, "\"", start);
    if (end == -1)
        return "";

    return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Extract double value from simple JSON                           |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
    string search = "\"" + key + "\":";
    int start = StringFind(json, search);
    if (start == -1)
        return 0.0;

    start += StringLen(search);

    // Find end (comma, brace, or bracket)
    int end = start;
    while (end < StringLen(json))
    {
        ushort ch = StringGetCharacter(json, end);
        if (ch == ',' || ch == '}' || ch == ']')
            break;
        end++;
    }

    string value = StringSubstr(json, start, end - start);
    StringTrimLeft(value);
    StringTrimRight(value);

    return StringToDouble(value);
}

//+------------------------------------------------------------------+
//| Extract object from JSON (returns substring)                    |
//+------------------------------------------------------------------+
string ExtractJsonObject(string json, string key)
{
    string search = "\"" + key + "\":";
    int start = StringFind(json, search);
    if (start == -1)
        return "";

    start += StringLen(search);

    // Skip whitespace
    while (start < StringLen(json) && StringGetCharacter(json, start) == ' ')
        start++;

    if (start >= StringLen(json) || StringGetCharacter(json, start) != '{')
        return "";

    // Find matching closing brace
    int brace_count = 1;
    int end = start + 1;
    while (end < StringLen(json) && brace_count > 0)
    {
        ushort ch = StringGetCharacter(json, end);
        if (ch == '{')
            brace_count++;
        if (ch == '}')
            brace_count--;
        end++;
    }

    return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Process trading signal from Python                              |
//+------------------------------------------------------------------+
void ProcessSignal(string json_string)
{
    // Extract the "data" object
    string signal_data = ExtractJsonObject(json_string, "data");
    if (signal_data == "")
    {
        Print("Failed to extract signal data");
        return;
    }

    // Extract signal fields
    string signal_type = ExtractJsonString(signal_data, "signal_type");
    string symbol = ExtractJsonString(signal_data, "symbol");
    double entry_price = ExtractJsonDouble(signal_data, "entry_price");
    double stop_loss = ExtractJsonDouble(signal_data, "stop_loss");
    double take_profit = ExtractJsonDouble(signal_data, "take_profit");
    double position_size = ExtractJsonDouble(signal_data, "position_size");
    string strategy_name = ExtractJsonString(signal_data, "strategy_name");

    Print("Received ", signal_type, " signal for ", symbol,
          " from strategy: ", strategy_name);

    if (!EnableTrading)
    {
        Print("Trading disabled - signal ignored");
        SendExecutionReport(signal_type, "ignored", 0.0);
        return;
    }

    // Execute signal
    bool result = false;
    double execution_price = 0.0;

    if (signal_type == "BUY")
    {
        result = ExecuteBuy(symbol, position_size, stop_loss, take_profit);
        execution_price = SymbolInfoDouble(symbol, SYMBOL_ASK);
    }
    else if (signal_type == "SELL")
    {
        result = ExecuteSell(symbol, position_size, stop_loss, take_profit);
        execution_price = SymbolInfoDouble(symbol, SYMBOL_BID);
    }
    else if (signal_type == "CLOSE")
    {
        result = ClosePosition(symbol);
        execution_price = 0.0;
    }

    // Send execution report
    string status = result ? "executed" : "failed";
    SendExecutionReport(signal_type, status, execution_price);
}

//+------------------------------------------------------------------+
//| Execute buy order                                               |
//+------------------------------------------------------------------+
bool ExecuteBuy(string symbol, double size, double sl, double tp)
{
    double volume = CalculateVolume(size);
    double price = SymbolInfoDouble(symbol, SYMBOL_ASK);

    // Normalize SL/TP
    if (sl > 0)
        sl = NormalizeDouble(sl, _Digits);
    if (tp > 0)
        tp = NormalizeDouble(tp, _Digits);

    bool result = trade.Buy(volume, symbol, price, sl, tp, "Python Signal");

    if (result)
    {
        Print("BUY order executed: ", volume, " lots at ", price);
    }
    else
    {
        Print("BUY order failed: ", trade.ResultRetcodeDescription());
    }

    return result;
}

//+------------------------------------------------------------------+
//| Execute sell order                                              |
//+------------------------------------------------------------------+
bool ExecuteSell(string symbol, double size, double sl, double tp)
{
    double volume = CalculateVolume(size);
    double price = SymbolInfoDouble(symbol, SYMBOL_BID);

    // Normalize SL/TP
    if (sl > 0)
        sl = NormalizeDouble(sl, _Digits);
    if (tp > 0)
        tp = NormalizeDouble(tp, _Digits);

    bool result = trade.Sell(volume, symbol, price, sl, tp, "Python Signal");

    if (result)
    {
        Print("SELL order executed: ", volume, " lots at ", price);
    }
    else
    {
        Print("SELL order failed: ", trade.ResultRetcodeDescription());
    }

    return result;
}

//+------------------------------------------------------------------+
//| Close position for symbol                                       |
//+------------------------------------------------------------------+
bool ClosePosition(string symbol)
{
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(ticket))
        {
            if (PositionGetString(POSITION_SYMBOL) == symbol &&
                PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                bool result = trade.PositionClose(ticket);
                if (result)
                {
                    Print("Position closed: ", ticket);
                }
                else
                {
                    Print("Failed to close position: ", trade.ResultRetcodeDescription());
                }
                return result;
            }
        }
    }

    Print("No position found for ", symbol);
    return false;
}

//+------------------------------------------------------------------+
//| Calculate position volume based on risk                         |
//+------------------------------------------------------------------+
double CalculateVolume(double position_size)
{
    if (position_size > 0)
    {
        // Use provided size
        return NormalizeDouble(position_size, 2);
    }

    // Calculate based on risk percentage
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * RiskPercent / 100.0;

    // Simple volume calculation (can be enhanced)
    double volume = 0.01; // Minimum volume

    // Normalize to broker's allowed values
    double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    volume = MathMax(volume, min_volume);
    volume = MathMin(volume, max_volume);
    volume = MathFloor(volume / volume_step) * volume_step;

    return NormalizeDouble(volume, 2);
}

//+------------------------------------------------------------------+
//| Send execution report to Python                                 |
//+------------------------------------------------------------------+
void SendExecutionReport(string signal_type, string status, double execution_price)
{
    string fields = "";
    fields += "\"signal_id\":\"" + IntegerToString(TimeCurrent()) + "\"";
    fields += ",\"signal_type\":\"" + signal_type + "\"";
    fields += ",\"status\":\"" + status + "\"";
    fields += ",\"execution_price\":" + DoubleToString(execution_price, _Digits);

    string json = BuildJsonString("execution_report", fields);
    SendMessage(json);
}

//+------------------------------------------------------------------+