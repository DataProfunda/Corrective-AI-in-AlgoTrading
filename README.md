<h1>Corrective AI in AlgoTrading</h1>

<h2>Description</h2>
My project facilitates application of Corrective AI to existing trading strategy. Strategy (Moving Average) sends signal to buy/sell and XGBoost predict the propability if this particular trade is profitable.
Data is downloaded from Oanda via tpqoa liblary. 
Inspired by dr. Ernest Chan.
<br />

<h2>Results</h2>

<b> XGBoost shows over 2% higher profit</b> </br>
<b> Improved Sharpe Ratio </b> </br>
<b> Improved Calmar  </b> </br>

<h2>Languages and Utilities Used</h2>

- <b>Python</b> 
- <b>Tensorflow</b>
- <b>XGBoost<b/>
- <b>Oanda<b/>

<h2>Program walk-through:</h2>

<br />
1. Set your access api in oanda.cfg<br/>
 Fill you api credentials in oand.cfg file

```python
[oanda]
account_id = 
access_token = 
account_type = practice
```

2. Download the data.

Specify wanted pairs and target that you want to trade.
   
```python
symbols = ['AUD_CAD', 'BCO_USD'] # Used currency pair
env = prfenv.ProfundaEnv(symbols = symbols,target = 'AUD_CAD', start_date = '2021-12-20 00:00:00', end_date = '2023-12-20 00:00:00', freq='M5', api='oanda.cfg')
```

<br />
<br />
3. Results.<br/>

![image](https://github.com/DataProfunda/Corrective-AI-in-AlgoTrading/assets/69935274/c8d409fe-5f3f-4b96-af6a-ea74339b1a94)



<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
