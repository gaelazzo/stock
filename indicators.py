# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:04:11 2021

@author: Nino
"""
import talib as talib

def evaluateAverageVolume(df, length=20):    
    df["averageVolume"] = df["Volume"].rolling(window=length).mean()
    
    
def evaluateBollingerBands(df, length=20, mul=2):    
    #sets BBwma,BBdev,BBup BBlow columns of the dataframe
    df["BBwma"] = df["Close"].rolling(window=length).mean()
    df["BBdev"] = df["Close"].rolling(window=length).std()
    df["BBup"] = df["BBwma"]+ mul* df["BBdev"]
    df["BBlow"] = df["BBwma"]- mul* df["BBdev"]


def evaluateRSI(df, length=14, oversoldLine=30, overboughtLine=70):    
    #sets RSI,overbought,oversold columns of the dataframe
    RSI = talib.RSI(df["Close"], timeperiod=length)
    df["overbought"] = overboughtLine
    df["oversold"] = oversoldLine
    df["RSI"]=RSI
    