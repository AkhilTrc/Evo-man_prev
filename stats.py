# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:06:21 2020

@author: Ayuub Hussein
"""

#test_data cleaning
import pandas as pd
df2 = pd.read_fwf('evoman_test_logs_enemy2_algo1.txt')
df2=df2.dropna()
df2.columns = ["1", "2", "3", "4", "enemy", "6","fitness","8","9","player_life","11","12","enemy_life", "14","15"]

df=df2[['enemy','fitness','player_life','enemy_life']]
#print(df)



# training data cleaning
df3=pd.read_fwf('evoman_training_logs_enemy3_algo1.txt', sep = " ,")
df3=df3.dropna()
df3.columns = ["1", "2", "3", "4", "enemy", "6","7"]
df3[['fitness','player_life_non_numerical','enemy_life_non_numerical','11']]=df3['7'].str.split(";",expand=True,)
df4=df3[['enemy','fitness', 'player_life_non_numerical', 'enemy_life_non_numerical']]
df4[['player_life_string','player_life_numerical']]=df4['player_life_non_numerical'].str.split(":",expand=True)
df4[['enemy_life_string','enemy_life_numerical']]=df4['enemy_life_non_numerical'].str.split(":",expand=True)
df4=df4.drop(['player_life_non_numerical','enemy_life_non_numerical','enemy_life_string', 'player_life_string'], axis=1)
print(df4)






