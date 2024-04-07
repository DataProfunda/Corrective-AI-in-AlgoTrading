# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 05:53:29 2024

@author: koste
"""
#from datetime import datetime
import os
import pandas as pd
import tpqoa
import numpy as np
import math 

class ProfundaEnv:
    
    def __init__(self,symbols, target, start_date : str, end_date : str, freq = 'S30', leverage = 1, api = '../cert/oanda.cfg', path = '../data/',
                 price = 'M', lags = 6, min_accuracy = 0.45, min_performance = 0.45, split = [0.8, 0.1, 0.1], 
                 data=None, data_target=None,
                 features =  ['date',  'roll_mean', 'roll_std', 'max', 'min', 'rsi','exp_ma',
                              'exp_mal', 'mal', 'ma','linear_reg','linear_regl', 'max_l', 'min_l',
                              'quad_reg','quad_regl', 'vol'], bias = 0.001, reverse_window = 300):
        self.symbols = symbols
        self.target = target
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.api = api
        self.path = path
        self.price = price
        self.lags = lags
        self.leverage = leverage
        self.min_accuracy = min_accuracy
        self.min_performance = min_performance 
        self.split = split
        self.data = data
        self.data_target = data_target
        self.features = features
        self.reverse_window = reverse_window
        self.bias = bias
        
        self.accuracy_list = list()
        
        if self.data is not None and self.data_target is not None:       
            self.input_cols = self.data.columns #Input data columns names           
        else:
            self.download_data()
        
    def download_data(self):
        """Method for downloading data from Oanda
        if symbols are list, the list of symbols would be downloaded"""        
        
        
        if isinstance(self.symbols, str):
            
            self.file_name = self.symbols +'_' +self.start_date + '_' + self.end_date + '_' + self.freq + '.csv'
            self.file_name = self.file_name.replace(' ', '_').replace('-', '_').replace(':', '_')
            
            self.file_path = self.path + self.file_name
            
            if os.path.exists(self.file_path):
                self.raw_data = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
                print('File exists')
                
            else:
                #print()
                #self.raw_data = tpqoa.tpqoa(self.api).get_history(self.symbol, self.start_date, self.end_date, granularity=, price= self.price)
                self.raw_data = tpqoa.tpqoa(self.api).get_history(self.symbols,self.start_date, self.end_date , granularity = self.freq, price = self.price)
                
                if not os.path.isdir('../data'):
                    os.mkdir('../data')
                
                self.raw_data.to_csv(self.file_path)
                
            self.data = pd.DataFrame(self.raw_data[['c', 'volume']], columns = [self.symbols,self.symbols+'_vol'])
            
        elif isinstance(self.symbols, list):
            
            self.data = pd.DataFrame()
            
            for symbol in self.symbols:
                self.file_name = symbol +'_' + self.start_date + '_' + self.end_date + '_' + self.freq + '.csv'
                self.file_name = self.file_name.replace(' ', '_').replace('-', '_').replace(':', '_')
                
                self.file_path = self.path + self.file_name
                
                if os.path.exists(self.file_path):
                    self.raw_data = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
                    print('File exists')
                    
                else:
                    #print()
                    #self.raw_data = tpqoa.tpqoa(self.api).get_history(self.symbol, self.start_date, self.end_date, granularity=, price= self.price)
                    self.raw_data = tpqoa.tpqoa('oanda.cfg').get_history(symbol,self.start_date, self.end_date , granularity = self.freq, price = self.price)
                    
                    if not os.path.isdir('../data'):
                        os.mkdir('../data')
                    
                    self.raw_data.to_csv(self.file_path)
                
                print('Downloaded: ' + self.file_name )
                temp_df = pd.DataFrame(columns = [symbol,symbol + '_vol'])
                temp_df[symbol] = self.raw_data['c']
                temp_df[symbol + '_vol'] = self.raw_data['volume']
                
                print(temp_df.head())
                
                self.data = pd.merge(self.data, temp_df, how='outer', left_index=True, right_index=True)
                              
            
        else:
            raise ValueError(f"Value provided for symbols: ({self.symbols}) is neither list or a string.")
            
        self.preprocess_data(features = self.features, standardize = False)
        

    def preprocess_data(self, features, window = 10, mean = None, std = None, standardize = True):
        """Is used for preparing forward filling data, and bfill for making sure that there are no missing values adding features and standardize
        features
        
        Parameters
        ----------
        features : list
            date - Deriving date features year, month, day, hour, minute, second, day_of_week and add to dataframe
            reindex - if there are missing periods reindexing the data to get rid of gaps
            roll_mean - add rolling mean of returns and value of target price
            roll_std - add rolling std of returns and value of target price
        
        window : int
            Size of the window if in features there are rolling fucntion
            
        mean : float
            Mean of the columns
        
        std : flaot
            Standard deviation of the columns
        
        standardize : bool , default True
            Standardize the data
        """
        print('date')
            
        #Deriving date features
        if 'date' in features:
            self.date_derive()
              
        #Dealing with missing data
        
        if False:
            self.data = self.data.ffill(axis=0)
            self.data = self.data.bfill(axis=0) 
            
            if self.data.isna().sum().any():
                raise ValueError('There are sill missing values after bfill and ffill!')

        
        if 'reindex' in features:
            print('reindex')
        #Reindexing the dateindex, because of missing week days
            self.data.index = pd.date_range(start=self.data.index.min(), periods=len(self.data), freq='30S')

        print('return')
        #Target return 
        self.data['return'] = np.log(self.data[self.target] / self.data[self.target].shift(1)) #Log return of the target price
        self.data['rev_return'] = self.data.loc[::-1,'return'].rolling(self.reverse_window).sum()
        self.data = self.data.apply(self.ret_dumm, axis=1)
        print(self.data.columns)
        self.data = self.data.drop('rev_return', axis=1)
        self.data.dropna(inplace=True)

        if 'roll_mean' in features:
            print('roll_mean')
            self.data['roll_mean'] = self.data[self.target].rolling(window).mean()
            self.data['roll_mean_ret'] = self.data['return'].rolling(window).mean()

        print('roll_std')    
        if 'roll_std' in features:
            self.data['roll_std'] = self.data[self.target].rolling(window).std()
            self.data['roll_std_ret'] = self.data['return'].rolling(window).std()
        
        print('max')        
        if 'max' in features:           
            self.data['max'] = self.data[self.target].rolling(window).max()
            self.data['roll_max_ret'] = self.data['return'].rolling(window).max()

        print('max_l')        
        if 'max_l' in features:
            self.data['max_l'] = self.data[self.target].rolling(window**2).max()

        print('min')     
        if 'min' in features:           
            self.data['min'] = self.data[self.target].rolling(window).min()
            self.data['roll_min_ret'] = self.data['return'].rolling(window).min()  

        print('min_l')    
        if 'min_l' in features:
            self.data['min_l'] = self.data[self.target].rolling(window**2).min()
            
        print('exp_ma')    
        if 'exp_ma' in features:           
            #self.data['min'] = self.data[self.target].rolling(window).min()
            self.data['exp_mavg'] = self.data[self.target].ewm(span=window).mean()

        print('exp_mal') 
        if 'exp_mal' in features:           
            #self.data['min'] = self.data[self.target].rolling(window).min()
            self.data['exp_mavg_l'] = self.data[self.target].ewm(span=window**2).mean()        
        
        print('ma') 
        if 'ma' in features:           
            #self.data['min'] = self.data[self.target].rolling(window).min()
            self.data['ma'] = self.data[self.target].rolling(window).mean()  

        print('mal') 
        if 'mal' in features:           
            #self.data['min'] = self.data[self.target].rolling(window).min()
            self.data['mal'] = self.data[self.target].rolling(window**2).mean() 
        
        print('linear_reg') 
        if 'linear_reg' in features:
            self.data['reg'] = self.data[self.target].rolling(window).apply(lambda x: np.polyfit(np.arange(window), x, 1)[0], raw=True)

        print('linear_reg_l') 
        if 'linear_reg_l' in features:
            self.data['regl'] = self.data[self.target].rolling(window**2).apply(lambda x: np.polyfit(np.arange(window**2), x, 1)[0], raw=True)

        print('quad_reg') 
        if 'quad_reg' in features:
            self.data['quad_reg'] = self.data[self.target].rolling(window).apply(lambda x: np.polyfit(np.arange(window), x, 2)[0], raw=True)

        print('quad_reg_l') 
        if 'quad_reg_l' in features:
            self.data['quad_regl'] = self.data[self.target].rolling(window**2).apply(lambda x: np.polyfit(np.arange(window**2), x, 2)[0], raw=True)

        if 'vol' not in features:
            for sym in self.symbols:
                self.data =  self.data.drop([ sym+'_vol'], axis=1)

           
        
        self.data.dropna(inplace=True)
        self.target_cols = ['t_up','t_zero','t_down']
        self.data_target = self.data[self.target_cols].copy()
        self.data = self.data.drop(self.target_cols, axis=1)
        #Standardize data
        if standardize and ( mean is None or std is None ) :
            self.mean = self.data.mean()
            self.std = self.data.std()
            self.std = self.std.replace(0, 1)
            
        elif standardize:
            self.data = (self.data - self.mean) / (self.std)
        
        

        #Create target value 
        #self.data['t'] = np.where(self.data['return'] > 0, 1, 0)
        #self.data['t'] = self.data['t'].astype(int)
        
        if 'rsi' in features:
            
            delta = self.data[self.target].diff()
            gain = delta.where(delta > 0, 0)
            loss = delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            self.data['rsi'] = rsi
        
            self.data['rsi'] = self.data['rsi'].fillna(0.01).replace(-np.inf, 0)
            self.data['rsi'] = (self.data['rsi'] - self.data['rsi'].mean()) / self.data['rsi'].std()
        
        
        #self.input_cols = self.data.drop(['rev_return'], axis=1).columns #Input data columns names
        
        
    def date_derive(self):     
        '''Creating date columns from index
        '''
        def date_add(row):
            
            row_name = row.name
            row['year'] = row_name.year
            row['month'] = row_name.month
            row['day'] = row_name.day
            row['hour'] = row_name.hour
            row['minute'] = row_name.minute
            row['second'] = row_name.second
            row['day_of_week'] = row_name.weekday()
        
            return row   
        
        self.data[['year', 'month', 'day', 'hour', 'minute', 'second', 'day_of_week']] = 0
        self.data = self.data.apply(date_add, axis = 1)
        
    def step(self, action):
        
        correct = action == self.data['t'].iloc[self.bar]
        ret = self.data['return'].iloc[self.bar] * self.leverage
        
        reward_1 = 1 if correct else 0 
        reward_2 = abs(ret) if correct else -abs(ret)
        reward = reward_1 + reward_2 * self.leverage
        self.treward += reward_1
        self.bar += 1
        self.accuracy = self.treward / (self.bar - self.lags)
        
        #print('Accuracy: ' , self.accuracy)
       
        self.performance *= math.exp(reward_2)
        if self.bar >= len(self.data):
            done = True
        elif reward_1 == 1:
            done = False
        elif (self.accuracy < self.min_accuracy and
              self.bar > self.lags + 15):
            done = True
        elif (self.performance < self.min_performance and
              self.bar > self.lags + 15):
            done = True
        else:
            done = False
        state = self._get_state()
        info = {}
        return state, reward, done, info        
        
     
    def get_state(self,bar):
        return self.data[self.input_cols].iloc[bar - self.lags:bar].values
        
    def _get_state(self):
        return self.data[self.input_cols].iloc[self.bar - self.lags:self.bar].values
               
    def reset(self):
        ''' Resetting env state
        '''
        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        self.bar = self.lags
        state = self._get_state()
        
        return state
    
    def train_val_test_split(self):
        
        if sum(self.split) == 1:
            data_len = len(self.data)
            learn_split = int(data_len * self.split[0])         
            val_split = int(data_len * self.split[1])      
            test_split = int(data_len * self.split[2])      
            
            learn_env = ProfundaEnv(self.symbols, self.target, self.start_date, self.end_date, data = self.data.iloc[:learn_split,:], data_target=self.data_target.iloc[:learn_split,:])
            val_env = ProfundaEnv(self.symbols, self.target, self.start_date, self.end_date, data =  self.data.iloc[learn_split : learn_split + val_split,:],data_target=self.data_target.iloc[learn_split : learn_split + val_split,:])
            test_env = ProfundaEnv(self.symbols, self.target, self.start_date, self.end_date, data =  self.data.iloc[learn_split + val_split : learn_split + val_split + test_split,:], data_target=self.data_target.iloc[learn_split + val_split : learn_split + val_split + test_split,:])
            return learn_env, val_env, test_env
        
    def ret_dumm(self,row):
        if abs(row['rev_return']) > self.bias:
            if row['rev_return'] > self.bias:
                row['t_up'] = 1
                row['t_zero'] = 0
                row['t_down'] = 0 
            else:
                row['t_up'] = 0
                row['t_zero'] = 0
                row['t_down'] = 1 
        else:
            row['t_up'] = 0
            row['t_zero'] = 1
            row['t_down'] = 0 

        return row
                
    def get_input_output_values(self):
        return self.data, self.data_target          
        
            
        