# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:04:11 2021

@author: Nino
"""
import talib as talib
from talib import MA_Type

def evaluateAverageVolume(df, length=20):    
    df["averageVolume"] = df["Volume"].rolling(window=length).mean()
    
# https://stock-analysis-engine.readthedocs.io/en/latest/talib.html

# all indicators described
# http://www.tadoc.org/index.htm

#'DEMA', 'EMA', 'KAMA', 'MAMA', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'
# DEMA                 Double Exponential Moving Average
# EMA                  Exponential Moving Average
# KAMA                 Kaufman Adaptive Moving Average
# MAMA                 MESA Adaptive Moving Average
# SMA                  Simple Moving Average
# T3                   Triple Exponential Moving Average (T3)
# TEMA                 Triple Exponential Moving Average
# TRIMA                Triangular Moving Average
# WMA                  Weighted Moving Average


# BBANDS               Bollinger Bands

# HT_TRENDLINE         Hilbert Transform - Instantaneous Trendline
# MA                   Moving average
# MAVP                 Moving average with variable period
# MIDPOINT             MidPoint over period
# MIDPRICE             Midpoint Price over period
# SAR                  Parabolic SAR
# SAREXT               Parabolic SAR - Extended

def evaluateAverage(df, length=20, avgType="EMA"):
    """

    Parameters
    ----------
    df: DataSet

    length: int
    number of data items to mediate

    avgType: string
    one of 'DEMA', 'EMA', 'KAMA', 'MAMA', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'

    Returns
    evaluated average as a column

    """
    if avgType == "EMA":
        return talib.EMA(df["Close"].values, timeperiod=length)

    if avgType == "DEMA":
        return talib.DEMA(df["Close"].values, timeperiod=length)

    if avgType == "KAMA":
        return talib.KAMA(df["Close"].values, timeperiod=length)

    if avgType == "MAMA":
        return talib.MAMA(df["Close"].values, timeperiod=length)

    if avgType == "SMA":
        return talib.SMA(df["Close"].values, timeperiod=length)

    if avgType == "T3":
        return talib.T3(df["Close"].values, timeperiod=length)

    if avgType == "TRIMA":
        return talib.TRIMA(df["Close"].values, timeperiod=length)

    if avgType == "WMA":
        return talib.WMA(df["Close"].values, timeperiod=length)

def evaluateBollingerBands(df, length=20, mul=2):
    """

    Parameters
    ----------
    df: DataSet
    length: int
    how many data rows to consider

    mul:float
    multiplier for the standard deviation to be added or subtracted from the average line

    Returns
    -------

    """
    #sets BBwma,BBdev,BBup BBlow columns of the dataframe
    # SMA is Simple Moving Average
    # upper, middle, lower = talib.BBANDS(df["Close"].values,
    #                   nbdevup = float(mul),
    #                   nbdevdn = float(mul),
    #                   matype = MA_Type.SMA)
    df["BBwma"] = df["Close"].rolling(window=length).mean()
    df["BBdev"] = df["Close"].rolling(window=length).std()
    df["BBup"] = df["BBwma"] + mul* df["BBdev"]
    df["BBlow"] = df["BBwma"] - mul* df["BBdev"]


def evaluateRSI(df, length=14, oversoldLine=30, overboughtLine=70):    
    #sets RSI,overbought,oversold columns of the dataframe
    RSI = talib.RSI(df["Close"], timeperiod=length)
    df["overbought"] = overboughtLine
    df["oversold"] = oversoldLine
    df["RSI"] = RSI
    