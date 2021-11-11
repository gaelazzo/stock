# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 12:27:40 2021

@author: Nino
"""
import urllib.request

import yfinance as yf
import tickers as tk;
import certifi
import os.path

import indicators

from datetime import datetime, timedelta
from patterns import patternList,applyPatterns

import dateutil.relativedelta
from datetime import date

import math
import pandas as pd  


#tech_stocks = ['AAPL', 'MSFT', 'INTC','IWDE.MI'] #
#yahoo_financials_tech = yhf(tech_stocks)

#print(yahoo_financials_tech.get_historical_price_data("2021-10-01", "2021-10-17", "daily"))
cachedPrevClose = {
}
cachedTickerHistory= {
}


def StringAfterPattern(text,tag1,tag2):
    """
    It assumes a text like 
    .... tag1 ... tag2 .. < S >
    and returns S
    
    Parameters
    ----------
    text : string
        text to be searched into
    tag1 : string
        first tag name to be searched
    tag2 : string
        second tag name to be searched

    Returns
    -------
    string
        the string S

    """

    posPattern= text.find(tag1);
    posBeforeValuePattern = text.find(tag2,posPattern);
    posValuePattern = text.find(">",posBeforeValuePattern)+1;
    posStopValuePattern = text.find("<",posValuePattern);
    strPattern = text[posValuePattern:posStopValuePattern];
    return strPattern;

    
def getLastPrices(tickers):
    """
    Reads current and last prices of a list of tickers

    Parameters
    ----------
    tickers : array of strings
        array of bloomberg codes

    Returns
    -------
    dict
        {"currPrices":{ticker:value}, "prevClose":{ticker:value}}

    """
    if (len(tickers)==0):
        return {"currPrices":{}, "prevClose":{}}
    # data = yhf(tickers);
    # currPrices  = data.get_current_price();
    # prevClose = data.get_prev_close_price();    
    # return {"currPrices":currPrices, "prevClose":prevClose}
    data = yf.download(tickers=" ".join(tickers),period="2d",interval="1d") 
    currPrices= {}
    prevClose= {}
    
    toComplete = []
    if len(tickers)==1:
          symbol = tickers[0]
          prevCloseData = None 
          if len(data.index)==2:
              currPrices[symbol] = round(data["Close"][1],6) 
              prevCloseData = round(data["Close"][0],6)
          else:
              currPrices[symbol] = round(data["Close"][0],6) 
        
          if (prevCloseData!= prevCloseData or prevCloseData is None):
               toComplete.append(symbol)
          else:
              prevClose[symbol] = prevCloseData
        
         
    else:      
        for symbol in tickers:    
            prevCloseData = None 
            try:           
                if len(data["Close"][symbol])==2:
                    currPrices[symbol] = round(data["Close"][symbol][1],6) 
                    prevCloseData = round(data["Close"][symbol][0],6)
                else:
                    currPrices[symbol] = round(data["Close"][symbol][0],6) 
                        
            except:
                pass
            
            if (prevCloseData!= prevCloseData or prevCloseData is None):
               toComplete.append(symbol)
            else:
               prevClose[symbol] = prevCloseData
            
    if(len(toComplete)>0):
        for symbol in toComplete:  
            if(symbol in cachedPrevClose):
                prevClose[symbol]= cachedPrevClose[symbol];
                continue;
            #leggere da https://www.google.com/finance/quote/IWDE:BIT 
            ## prendere il testo che ha come class P6K39c dopo il testo "Chiudura Precedente" o PREVIOUS CLOSE
            code = symbol.replace(".MI",":BIT");
            url = f"https://www.google.com/finance/quote/{code}";
            response =  urllib.request.urlopen(url,cafile=certifi.where());
            html = response.read()
            text = html.decode()  
            closeStr =  StringAfterPattern(text,"closing price","P6K39c");
            close = float(closeStr.replace(",",".").replace("€","").strip())
            prevClose[symbol]=  close;
            cachedPrevClose[symbol] = close;
            
    return {"currPrices":currPrices, "prevClose":prevClose}


# %a  Locale’s abbreviated weekday name.
# %A  Locale’s full weekday name.      
# %b  Locale’s abbreviated month name.     
# %B  Locale’s full month name.
# %c  Locale’s appropriate date and time representation.   
# %d  Day of the month as a decimal number [01,31].    
# %f  Microsecond as a decimal number [0,999999], zero-padded on the left
# %H  Hour (24-hour clock) as a decimal number [00,23].    
# %I  Hour (12-hour clock) as a decimal number [01,12].    
# %j  Day of the year as a decimal number [001,366].   
# %m  Month as a decimal number [01,12].   
# %M  Minute as a decimal number [00,59].      
# %p  Locale’s equivalent of either AM or PM.
# %S  Second as a decimal number [00,61].
# %U  Week number of the year (Sunday as the first day of the week)
# %w  Weekday as a decimal number [0(Sunday),6].   
# %W  Week number of the year (Monday as the first day of the week)
# %x  Locale’s appropriate date representation.    
# %X  Locale’s appropriate time representation.    
# %y  Year without century as a decimal number [00,99].    
# %Y  Year with century as a decimal number.   
# %z  UTC offset in the form +HHMM or -HHMM.
# %Z  Time zone name (empty string if the object is naive).    
# %%  A literal '%' character.

def toDayHourMin(t):
    return t.strftime("%d %m %H") #:%M

def getDateFormat(interval):
    """
    

    Parameters
    ----------
    interval : string
        one of 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

    Returns
    -------
    string
        date format to be used for pretty printing while the interval is selected

    """
     #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    intervalToDateFormat = {
        "1m": "%d %H:%M",
        "2m": "%d %H:%M",
        "5m": "%d %H:%M",
        "10m": "%m %d %H:%M",
        "15m": "%m %d %H:%M",
        "30m": "%m %d %H:%M",
        "45m": "%m %d %H:%M",
        "60m": "&m %d %H",
        "90m": "%m %d %H:%M",
        "1h": "%m %d %H",
        "5h": "%m %d %H",
        "1d": "%y %m %d",
        "5d": "%y %m %d",
        "1wk": "%y %m %d",
        "2wk": "%y %m %d",
        "3wk": "%y %m %d",
        "1mo": "%y %m %d",
        "3mo": "%y %m %d "
        }
    return intervalToDateFormat[interval]
    
    
    
def fixXAsses(df,interval):
    """
    Evaluates prettyDate column

    Parameters
    ----------
    df : Dataframe
        
    interval : one of 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        

    Returns
    -------
    Changes in-place df adding a prettyDate column with the dataframe index formatted

    """
    
    
    
    fmt = getDateFormat(interval)
    # newIndex = pd.Index([t.strftime(fmt) for t in df.index])
    # df.set_index(newIndex,drop=True, inplace=True)
    df["prettyDate"] = [t.strftime(fmt) for t in df.index]
    
#/* ["1m","5m", "15m", "30m", "45m", "1h", "1d", "5d", "1wk", "1mo", "3mo"] */
 # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
IntervalAdapters = {"5h":{"newInterval":"1h", "factor":5},
                    "10m":{"newInterval":"5m","factor":2},
                    "45m":{"newInterval":"15m","factor":3},
                    "5d":{"newInterval":"1d","factor":5},
                    "3wk":{"newInterval":"1wk","factor":3},
                    "2wk":{"newInterval":"1wk","factor":2}
                    }

# from page: ["1d","1wk", "1mo", "1y", "3y","5y", "10y", "max"] 
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# periodAdapters =  { "1mo":{"newPeriod":"3mo", "skip":2/3},
#                     "3mo":{"newPeriod":"6mo", "skip":1/2},
#                     "1wk":{"newPeriod":"1mo", "skip":3/4},
#                     "1y":{"newPeriod":"3y", "skip":2/3},
#                     "3y":{"newPeriod":"5y", "skip":2/5},
#                     "1d":{"newPeriod":"5d", "skip":0},
#                     "3d":{"newPeriod":"5d", "skip":2/5},
#                     "4d":{"newPeriod":"10d", "skip":3/5},
#                     "5y":{"newPeriod":"10y", "skip":1/2}
#                    }


def squeezeData(_df, step):
    """
    Groups a datafram step rows at a time, recalculating columns .
    For each step rows in the source dataframe, there will be one row in 
     the output dataframe

    Parameters
    ----------
    _df : Dataframe
        
    step : int
        size of the blocks to be created

    Returns
    -------
    Dataframe
        a new Dataframe with the grouped rows

    """
    
    toDel=[]
    dfc = _df.copy()
    if (step is None): return dfc;
    nElement= len(dfc.index)
    for i in range(0,nElement,step):
        first= i
        idx = dfc.index[first]
        last = min(i+step,nElement)
        low= dfc["Low"][first]
        high= dfc["High"][first]
        volume = dfc["Volume"][first]
        dividends = dfc["Dividends"][first]
        
        for j in range(first+1,last):
            low = min(low,dfc["Low"][j])
            high = max(high,dfc["High"][j])
            volume +=   dfc["Volume"][j]
            dividends +=   dfc["Dividends"][j]
            toDel.append(j)
        #print(f"Assigning {dfc['Close'][last-1]} to {idx} where last is {last}")
        dfc["Close"].at[idx]=dfc["Close"][last-1]
        dfc["Adj Close"].at[idx]=dfc["Adj Close"][last-1]
        dfc["Low"].at[idx]=low
        dfc["High"].at[idx]=high
        dfc["Volume"].at[idx]= volume
        dfc["Dividends"].at[idx]= dividends
    
    #print ("dropping "+str(toDel))
    return dfc.drop(dfc.index[toDel])
    
def skipFirstRows(df,num):
    """
    Remove first num rows of df and every rows where RSI or BB bands 
     have not been evalued

    Parameters
    ----------
    df : Dataframe
        
    num : int
        rows to be removed

    Returns
    -------
    Dataframe
        Database where the specified rows have been removed

    """
    #step = int(len(df.index)/num)
    #if(step==0): return df
    #toDel= [x for x in range(len(df.index)) if x%step != 0]
    tot = len(df.index)
    toDel= [x for x in range(tot) if x<num 
                    or math.isnan(df["RSI"][x]) 
                    or math.isnan(df["BBlow"][x]) 
            ]
    return df.drop(df.index[toDel])

allowedInterval = ["1m","2m","5m","10m", "15m","30m","45m","60m","90m",
                   "1h","5h","12h",
                   "1d","5d","1wk","2wk", "3wk", "1mo","3mo"]
#allowedPeriod =  ["4h","12h", "1d","3d","1wk", "1mo","3mo", "1y", "3y","5y", "10y", "max"] 


def adjustDateForInterval(d,interval):
    """
    Yahoo does not allow to fetch too-ancient data for some intervals,
     so this function checks those limits and when the Date is out of range
     it is changed to the minimum value allowd for that interval

    Parameters
    ----------
    d : Date
        
    interval : string
        

    Returns
    -------
    Date fixed to be used with yahoo financial functions

    """
    today = datetime.today().date()    
    #print (today,type(today))
    #print (start,type(start))
    diffDay = (today-d).days;    
    

#7 days worth of 1m granularity data are allowed to be fetched per request.
# la granularità 1m è ammessa solo per gli ultimi 7 giorni
    if (interval=="1m" and diffDay>7):
        return today - timedelta(days=7)
    
    if (interval=="5m" and diffDay>60):
        return today - timedelta(days=60)

    if (interval=="1h" and diffDay>=730):
        return today - timedelta(days=730)
    
    if (interval.endswith("wk")):
        while(d.weekday()!=0):
             d = d - timedelta(days=1)
             #print(f"decremented {start.strftime('%Y-%m-%d')}")
        return d


    return d

#yahoo misses joining last day of the week in last set    
def fixMonthLastDay(df):
    """
    Groups the last two rows of the dataframe

    Parameters
    ----------
    df : dataframe        

    Returns
    -------
    dataframe
        

    """
    
    
    if (len(df.index)<2): return df
    
    nElement= len(df.index)
    lastIdx = df.index[nElement-1]
    lastWeek = df.index[nElement-2]
    #print(lastIdx.weekday())
    #print(lastWeek.weekday())
    
    if lastWeek.weekday()!=0 : return df    
    if lastIdx.weekday()==0 : return df
    
    dfc = df.copy()
    low= dfc["Low"][nElement-2]
    high= dfc["High"][nElement-2]
    volume = dfc["Volume"][nElement-2]
    volume = dfc["Volume"][nElement-2]
    div = dfc["Dividends"][nElement-2]
    
    low = min(low,dfc["Low"][nElement-1])
    high = max(high,dfc["High"][nElement-1])
    volume +=   dfc["Volume"][nElement-1]
    div +=   dfc["Dividends"][nElement-1]
    
    
    #print(f"{dfc['Close'][lastWeek]} to {dfc['Close'][lastIdx]}")
    
    dfc["Close"].at[lastWeek]=dfc["Close"][nElement-1]
    dfc["Adj Close"].at[lastWeek]=dfc["Adj Close"][nElement-1]
    #print(f"{dfc['Close'][lastWeek]}")
    dfc["Low"].at[lastWeek]=low
    dfc["High"].at[lastWeek]=high
    dfc["Volume"].at[lastWeek]= volume
    dfc["Dividends"].at[lastWeek]= div
    
    return dfc.drop(dfc.index[nElement-1])
        
        
def getDataRange(bloomberg, start, end, interval):
    """
    Gets data from yahoo, fixing data and extending yahoo allowed intervals,
     managing the grouping for new admitted intervals like 10m, 2wk and so on
     Uses a cached version for old data 

    Parameters
    ----------
    bloomberg : string
        bloomberg code
    start : Date or Datetime
        start date for data range to get, if it is datetime, it is converted to Date
    end : Date
        stop date for data range to get, if it is datetime, it is converted to Date
    interval : string
       one of allowedInterval = ["1m","2m","5m","10m", "15m","30m","45m","60m","90m",
                   "1h","5h","12h",
                   "1d","5d","1wk","2wk", "3wk", "1mo","3mo"]

    Raises
    ------
    Exception
        if date is not Date or Datetime

    Returns
    -------
    df : Dataframe
        Requested data

    """
    if (hasattr(start, "date") and callable(getattr(start, "date"))): start= start.date()
    if (hasattr(end, "date") and callable(getattr(end, "date"))):  end= end.date()
    if (not isinstance(start ,date)): raise Exception("getDataRange",
                                                               f"start must be a date (not a {type(start) })")
    if (not isinstance(end ,date)): raise Exception("getDataRange",
                                                              f"end must be a date (not a {type(end) })")
    
    if (interval not in allowedInterval):
        raise Exception("getDataRange","interval must be between "+str(allowedInterval))
        
    adapter=None
    factor=None
    intervalToUse=interval
    if interval in IntervalAdapters:        
        intervalToUse= IntervalAdapters[interval]["newInterval"]        
        factor= IntervalAdapters[interval]["factor"] 
    
    #st = yf.Ticker(bloomberg)
    # print(f"start: {adjustDateForInterval(start,intervalToUse).strftime('%Y-%m-%d')}")
    # print(f"end: {end.strftime('%Y-%m-%d')}")
    # print(f"intervalToUse: {intervalToUse}")
    df = getCachedHistory(bloomberg=bloomberg, start=adjustDateForInterval(start,intervalToUse),
                               end=end, interval=intervalToUse) 
        # st.history(start=adjustDateForInterval(start,intervalToUse),
        #                     end=end, interval=intervalToUse, back_adjust=False,auto_adjust=False)
    
    if (interval.endswith("wk")):
        df = fixMonthLastDay(df)
        
    df.dropna(axis='index', inplace=True)
    
    if (factor is not None):
        df = squeezeData(df, factor);

    return df



def getDayBufferInterval(offset,interval,hoursPerDay):
    """
    Evaluates how many days are necessary in order to ensure an offset (expressed in intervals units)

    Parameters
    ----------
    offset : int
        how many units of data are needed for the buffer
    interval : string
        an allowd interval, must end with m,h,d,wk,mo, ex. 2mo
    hoursPerDay : int
        How many hours to consider for a working day

    Raises
    ------
    Exception
        raised if interval not allowed

    Returns
    -------
    totDays : int
        number of days to be read to get the desired buffer

    """
    if (interval not in allowedInterval):
      raise Exception("getDayBufferInterval",f"interval ({interval}) must be between {str(allowedInterval)}")
    
    totDays=None
    #"1m","2m","5m","15m","30m","60m","90m"
    if interval.endswith("m"): 
        minutes = int(interval[:-1])
        totMin = offset* minutes 
        totHours = int( math.ceil(totMin/60))
        totDays = int( math.ceil(totHours/hoursPerDay))
        
    
    if interval.endswith("h"):
          totHours = int(interval[:-1])*offset
          totDays = int( math.ceil(totHours/hoursPerDay))
          
 
   
    if interval.endswith("d"):
           totDays = int(interval[:-1])*offset            
   
    if interval.endswith("wk"):
           totDays = (int(interval[:-2])* offset)*7            
   
    if interval.endswith("mo"):
           totDays = (int(interval[:-2])*offset) *22  #working days per month
   
    #print(f"totDays:{totDays} ")
    return totDays

allowedPeriod =  ["1h", "4h","12h", "1d","2d","3d","4d","6d", "8d", "15d", 
                              "1wk", "1mo", "2mo", "3mo","6mo", "1y", "3y","5y", "10y", "max"] 

def subtractPeriod(d,period,hoursPerDay):
    """
    Subtracts a period from a date

    Parameters
    ----------
    d : Date or DateTime
        Source date
    period : string
      one of  ["1h", "4h","12h", "1d","2d","3d","4d","6d", "8d", "15d", 
                              "1wk", "1mo", "2mo", "3mo","6mo", "1y", "3y","5y", "10y", "max"] 
        "max" is an alias for 30y
    hoursPerDay : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        if period not in allowed periods

    Returns
    -------
    same type as d
        the input date minus the specified  periods

    """
    if (period not in allowedPeriod):
         raise Exception("getPeriodDayLen",f"period ({period})must be between {str(allowedPeriod)}")
   
    delta = None
    
    if period.endswith("h"):
         totHours= int(period[:-1])  
         delta = dateutil.relativedelta.relativedelta(hours=int( math.ceil(totHours/hoursPerDay)))
         
    if period.endswith("d"):
         #take special attention on counting only working days
         totDays = int(period[:-1])  
         res = d
         while (totDays>0):
             if ( is_business_day(res)): totDays=totDays-1
             res = res - timedelta(days=1)
         return res
         #delta = dateutil.relativedelta.relativedelta(days=totDays)

    if period.endswith("wk"):
         totWeeks = int(period[:-2])  
         delta = dateutil.relativedelta.relativedelta(weeks=totWeeks)
    
    if period.endswith("mo"):
         totMonths = int(period[:-2])  
         delta = dateutil.relativedelta.relativedelta(months=totMonths)

    if period.endswith("y"):
         totYears = int(period[:-1])  
         delta = dateutil.relativedelta.relativedelta(years=totYears)

    if period == "max":
         totYears = 30  
         delta = dateutil.relativedelta.relativedelta(years=30)
    
    return d - delta    
     
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

def wantedRows(period,interval,hoursPerDay):
    """
    Evaluates how many rows are expected given a combination of period and interval

    Parameters
    ----------
    period : string
        must end with h,d,wk,mo,y
    interval : string
        must end with m,h,d,wk,mo
    hoursPerDay : int
        how many hours per day to consider

    Returns
    -------
    int
        number of expected rows when running a data request with the specified params

    """
    if period.endswith("h"):
          totPeriodHours= int(period[:-1])  
          totIntervalMin= int(interval[:-1])  
          return int ((totPeriodHours*60) / totIntervalMin)
    
    if period.endswith("d"):
          totPeriodDays= int(period[:-1])  
          if (interval.endswith("h")):
              totIntervalHours= int(interval[:-1])  
              #adds a row for every day, to display the close
              return int ((totPeriodDays*hoursPerDay) / totIntervalHours)+totPeriodDays
 
          if (interval.endswith("d")):           
              totIntervalDays= int(interval[:-1])  
              #adds a row for every day, to display the close
              return totIntervalDays
          
          if (interval.endswith("m")):
              totIntervalMin= int(interval[:-1])  
              #adds a row for every day, to display the close
              return int ((totPeriodDays*60*hoursPerDay) / totIntervalMin)+totPeriodDays
    
    if period.endswith("wk"):
          totPeriodWeeks= int(period[:-2])  
          totIntervalHours= int(interval[:-1])  
          #adds a row for every day, to display the close
          return int ((totPeriodWeeks*5*hoursPerDay) / totIntervalHours)+totPeriodWeeks*5
      
    if period.endswith("mo"):
          totPeriodMonths= int(period[:-2])  
          if (interval.endswith("h")):
               totIntervalHours= int(interval[:-1])  
               #adds a row for every day, to display the close
               return int ((22*(hoursPerDay+1)) / totIntervalHours) 
                               
          else:              
              totIntervalDays= int(interval[:-1])  
              return int ((totPeriodMonths*22) / totIntervalDays)
      
    if period.endswith("y"):
          totPeriodYears= int(period[:-1])  
          if (interval.endswith("d")):
               totIntervalDays= int(interval[:-1])  
               return int ((365*totPeriodYears) / totIntervalDays)
          
          if (interval.endswith("wk")):
               totIntervalWeeks= int(interval[:-2])  
               return int ((52*totPeriodYears) / totIntervalWeeks)
           
          if (interval.endswith("mo")):
               totIntervalmonths= int(interval[:-2])  
               return int ((12*totPeriodYears) / totIntervalmonths)
           
    return None



    
def evaluateDateRange(period,interval,end,bufferLen,hoursPerDay):
    """
     returns a range of dates that stops on end and includes a supplementary
      number of business days to evaluate at least additional bufferLen rows

    Parameters
    ----------
    period : string
        allowed period
    interval : string
        allowed interval
    end : Date
        last day of the interval requested
    bufferLen : int
        number of working days to be read
    hoursPerDay : int
        hours per day to consider

    Returns
    -------
    start : Date
        initial date of the requested interval
    stop : TYPE
        end date of the requested interval

    """
     
      
    stop = end
    while (not is_business_day(stop)):
         stop = stop + timedelta(days=1)
    stop = stop + timedelta(days=1)

    
    firstDay = subtractPeriod(end,period,hoursPerDay)
    buffer = getDayBufferInterval(bufferLen,interval,hoursPerDay)
    
    
    toDecrease = buffer
    
    #skips toDecrease working days 
    start = firstDay   # -  timedelta(days=buffer)    
    start = start - timedelta(days=1)
    while not is_business_day(start): 
        start = start - timedelta(days=1)
    
    while toDecrease>0:
        if is_business_day(start): toDecrease = toDecrease-1
        start = start - timedelta(days=1)
    
    #print(f"firstDay={firstDay}, buffer={buffer}, start={start}, stop={stop}")
    return (start, stop)


    
def getHistoryData(bloomberg, period, interval, 
                       patternsToCalc=patternList, 
                       evaluateRSI=True,
                       evaluateBollinger=True,
                       endDate=None):
    """
    

    Parameters
    ----------
    bloomberg : string
        bloomberg (yahoo) code for the ticker
    period : string
        period (ex. 1mo to read a month of data)
    interval : string
        interval (ex. 1h to consider a data every hour)
    patternsToCalc : string[], optional
        List of pattern code to evaluate. The default is patternList.
    evaluateRSI : bool, optional
        True to evaluate RSI. The default is True.
    evaluateBollinger : bool, optional
        True to evaluate Bollinger bands. The default is True.
    endDate:Date
    Returns
    -------
    df : Dataframe
        Requested data
        bullishPatterns is a column with bullish patterns
        bearishPatterns is a column with bearish patterns
        errorsPatterns is a column with errors evaluating patterns
        bloomberg is a column with the requested ticker code

    """      
    if endDate is None: endDate = datetime.today().date()
    
    #print(f"today = {today}")

    hoursPerDay=8.5 #those should be evalued basing on bloomberg code
    bufferLen = 25 #those should be evalued basing on bloomberg code

    

    start,stop = evaluateDateRange(period,interval, endDate,bufferLen,hoursPerDay)
   
    
    
   
    df= getDataRange(bloomberg,start,stop,interval)
    #print(f"{len(df)} rows read")
    # Create a line graph
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    #df = st.history(period=periodToUse, interval=intervalToUse)
    
    df.dropna(axis='index', inplace=True)
    
    indicators.evaluateAverageVolume(df)
    
    patterns = applyPatterns(df["Open"],df["High"],df["Low"],df["Close"], patternsToCalc)
    df["bullishPatterns"] = patterns["bullish"]
    df["bearishPatterns"] = patterns["bearish"]
    df["errorsPatterns"] = patterns["errors"]
    
    if (evaluateBollinger):
        indicators.evaluateBollingerBands(df)
    
    if (evaluateRSI):
        indicators.evaluateRSI(df)
    # df["Delta"] = df["Close"].diff() #-df["Open"]
    # df["Delta"] = df["Delta"][1:]
    # U,D = df["Delta"].copy(), df["Delta"].copy()
    # U[U<0] = np.Nan
    # D[D>0] = np.Nan
    # roll_up14 = U.rolling(14).mean()
    # roll_down14 = D.abs().rolling(14).mean()
    # RS = roll_up14 / roll_down14
    # RSI = 100.0 - (100.0 / (1.0 + RS))
  
    
    
 
    #df = skipFirstRows(df,25)
    
    if (period != "max"):
        wanted = wantedRows(period,interval,hoursPerDay)+1
        if (len(df.index)>wanted):
            df=skipFirstRows(df, len(df.index)-wanted)
    
    # print(f"{len(df)} rows given as result for {bloomberg} {period}{interval}")

    fixXAsses(df,interval)
    #data = adaptToCandletick(df);
    df["bloomberg"]= f"{bloomberg}"
    #data["xhoverformat"]= getDateFormat(df)
    return df.copy()



# 1m last 7 days
# 2m 5m 15m last 60 days
# 1h last 730 days
#7 days worth of 1m granularity data are allowed to be fetched per request.
# la granularità 1m è ammessa solo per gli ultimi 7 giorni

#5m data not available for startTime=1262300400 and endTime=1634908029. 
#  The requested range must be within the last 60 days.

def getDateFromIdx(v,interval):
    if (fitDateType(interval)=="datetime"): return v
    if (hasattr(v, "date") and callable(getattr(v, "date"))): return v.date()    
    return v

def getDate(v):
    if (hasattr(v, "date") and callable(getattr(v, "date"))): return v.date()    
    return v

def getNameForIndex(interval):
     if fitDateType(interval)=="date": return "Date"
     return "Datetime"
     
def fitDateType(interval):
    if (interval in ["1d","1wk","1mo","3mo"]): return "date"
    return "datetime"

def fitColumnType(df,columnName,interval):
    if fitDateType(interval)=="date":
          df[columnName] = pd.to_datetime(df[columnName]).dt.date
          return
    #print(df.columns)
    df[columnName] = pd.to_datetime(df[columnName]) #.dt.tz_convert("Europe/Rome")
    
def deleteUnsecureLastRow(df,interval):
    """
    Remove last row if it is still changing
    Parameters
    ----------
    df : DataFrame       
    interval : string in 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        

    Returns
    -------
    DataFrame with the  unsecure row eventually removed

    """
    maxDate=  df.index.max()
    maxDateAsDate = getDate(maxDate)    
    today = datetime.today().date()    
    #print(f"maxDateAsDate is {maxDateAsDate}")
    
    if fitDateType(interval)=="datetime" and maxDateAsDate<today:
         # print("datetime => No need to remove")
         return df #no need to remove last row
    
    # if range=="1d" and maxDateAsDate<today:
    #     return #no need to remove last row
    
    if interval=="1wk" :
        maxDateAsDate = maxDateAsDate + timedelta(days=7)      
    
    if interval=="1mo" :
        maxDateAsDate = maxDateAsDate + dateutil.relativedelta.relativedelta(months=1)
        
    if interval=="3mo" :
        maxDateAsDate = maxDateAsDate + dateutil.relativedelta.relativedelta(months=3)
    
    # print(f"new maxDateAsDate is {maxDateAsDate}")   
    
    if maxDateAsDate<today:
        # print("No need to remove")
        return df #no need to remove last row

    # print("One row removed")     
    return df.drop(df.tail(1).index) # drop last 1 rows
    
   
   
#interval admitted 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
def getCachedHistory(bloomberg,start,end,interval):          
    """
    

    Parameters
    ----------
    bloomberg : string
        bloomberg code of ticker, ex. ISP.MI, AAPL
    start : Date
        start Date to read data, included
    end : Date
        stop Date to read data, excluded
    interval : string
        one of 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

    Returns
    -------
    DataFrame
        Data retrieved from cache and from yahoo  finance

    """
    
    def fixIndex(dataFrame,interval):
        indexName = getNameForIndex(interval)
        if dataFrame.index.name is None:
                # print(f"dataFrame.index.name is None")
                if indexName in dataFrame.columns: 
                        fitColumnType(dataFrame,indexName,interval)
                        # print(f"fixing column type")
                        dataFrame.set_index(indexName,inplace=True)                        
                else:
                     # print(f"fixing index type")
                     if fitDateType(interval)=="date":
                         dataFrame.index = pd.to_datetime(dataFrame.index).date
                     # print(f"{df.index}")
                if dataFrame.index.name is None:
                    dataFrame.index.name=indexName 
        else:         
                # print(f"dataFrame.index.name is not None")
                if fitDateType(interval)=="date":
                         dataFrame.index = pd.to_datetime(dataFrame.index).date
                         dataFrame.index.name=indexName 
                    
    today = datetime.today().date()        
    fileName = f"{bloomberg}_history_{interval}.csv"
    filePath = "datasets/"+fileName;    
    st = None
    indexName = getNameForIndex(interval)
    if (fileName in cachedTickerHistory):
        df = cachedTickerHistory[fileName]        
    else:
        if not os.path.isfile(filePath):
                st = yf.Ticker(bloomberg)
                #forward the request to yahoo
                df = st.history(start=adjustDateForInterval(start,interval),
                                 end=adjustDateForInterval(end, interval),
                                 interval=interval, 
                                 back_adjust=False,
                                 auto_adjust=False)   
                fixIndex(df,interval)
                dfToSave = deleteUnsecureLastRow(df,interval)
                #df.loc[[index for index,row in df.iterrows() if getDateFromIdx(index)<today]]
                dfToSave.to_csv(filePath)
                cachedTickerHistory[fileName]=dfToSave
                return df;
    
    #read existing data
    # print(f"reading {filePath}")
    df = pd.read_csv(filePath)  ##,infer_datetime_format=True
    # print(df)
    # print(df.index)
    # print(df.dtypes)
    fixIndex(df,interval)
    
    
    #df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    
    # if(df.index.name != indexName):
    #     print (f"Changing index name from {df.index.name} to {indexName}")
    #     df.index.set_names(indexName) 
    #     print(df.index)
    
    fileStart = getDateFromIdx(df.index.min(),interval)
    fileStartDate  =getDate(fileStart)
    fileStop =  getDateFromIdx(df.index.max(),interval)
    fileStopDate  =getDate(fileStop)
    initialFileStart= fileStart
    initialFileStop= fileStop
    # print(f"fileStart {fileStart} fileStop{fileStop}")

    # toSave = False
    #integrates missing rows before fileStart
    if (fileStartDate>start):
          if (st is None):st = yf.Ticker(bloomberg)
          startToRead = start #- timedelta(days=5) #just to be sure          
          stopToRead=  fileStartDate #+ timedelta(days=1) #just to be sure   
          # print(f"{bloomberg} fileStart>start  startToRead{startToRead} stopToRead{stopToRead}  so toSave set to true") 
          adjStart=adjustDateForInterval(startToRead,interval)
          adjEnd=adjustDateForInterval(stopToRead, interval)        
          print(f"reading {bloomberg} from adjStart:{adjStart} to adjEnd:{adjEnd} interval:{interval}[1]")
          dff = st.history(start=adjStart,
                            end=adjEnd,
                            interval=interval, 
                            back_adjust=False,
                            auto_adjust=False)
          if len(dff.index)>0:
              print("Some Data was read [1]")
              fixIndex(dff,interval)
              newRows = dff[dff.index < fileStart]
              df = df.append(newRows)
              df.sort_index(inplace=True)
              # dff.index.set_names(indexName)    
              # dff.index.name=indexName 
              # dfmerge = df.combine_first(dff)          
              # df = dfmerge
              fileStart = getDateFromIdx(df.index.min(),interval)
              fileStartDate  =getDate(fileStart)
              fileStop =  getDateFromIdx(df.index.max(),interval)
              fileStopDate  =getDate(fileStop)
              # print(f"now fileStart {fileStart} fileStop{fileStop}")
              # toSave=True

    endToUpdate=False
    idxType= fitDateType(interval)
    if idxType=="datetime" and (fileStopDate == today or end>=today): 
        endToUpdate=True
    if idxType=="date" and (fileStopDate < end or end==today): 
        endToUpdate=True
        
    
    #if data is entirely present don't read anything
    if endToUpdate:
         #data is not entirely present so some data has to be read 
        readableStop  = end
        
        # today's stop must never be read or saved so it is not to check
        if (end>today and idxType=="date"): readableStop = today
            
        startToRead = end - timedelta(days=1) #just to be sure  
        if (startToRead>fileStopDate):
            startToRead = fileStopDate + timedelta(days=1) 
        
        #integrates missing rows after fileStop
        if (fileStopDate<readableStop):
              startToRead = fileStopDate #+ timedelta(days=1) 
              if startToRead== end: startToRead = startToRead- timedelta(days=1) 
              # print(f"{bloomberg} fileStopDate<readableStop {fileStopDate}<{readableStop} ")               
              # toSave=True
        
        if (st is None):st = yf.Ticker(bloomberg)
        # print(f" fileStop<end  startToRead{startToRead} end{end}")      
        adjStart=adjustDateForInterval(startToRead,interval)
        adjEnd=adjustDateForInterval(end, interval)        
        print(f"reading {bloomberg} from adjStart:{adjStart} to adjEnd:{adjEnd} interval:{interval}[2]")
        dff = st.history(start=adjStart,
            end=adjEnd,
            interval=interval, 
            back_adjust=False,
            auto_adjust=False)
        if len(dff.index)>0:
            print("Some Data was read [2]")
            fixIndex(dff,interval)
            # print(f"{dff.index}")
            newRows = dff[dff.index > fileStop]
            # print(f"New Rows :  {newRows}")            
            # print(f"Original  {df.index.min()} to {df.index.max()}") 
            # print(f"New data from  {dff.index.min()} to {dff.index.max()} ")               
            # print(f"dff.index.name == df.index.name  ?{dff.index.name} to {df.index.name} ")  
            assert dff.index.name == df.index.name
            df = df.append(newRows,sort=True)
            # print(f"Result into  {df.index.min()} - {df.index.max()}")
            
            fileStart = getDateFromIdx(df.index.min(),interval)
            fileStartDate  =getDate(fileStart)
            fileStop =  getDateFromIdx(df.index.max(),interval)
            fileStopDate  =getDate(fileStop)
            # print(f"now2 fileStart {fileStart} from {df.index.min()} fileStop{fileStop} from {df.index.max()}")              
                       
        #removes today before saving data
        dfToSave = deleteUnsecureLastRow(df,interval)
        #dfToSave = df.loc[[index for index,row in df.iterrows() if getDateFromIdx(index)<today]]
        fileStop =  getDateFromIdx(dfToSave.index.max(),interval)
        if (initialFileStart!=fileStart or initialFileStop!=fileStop):
            dfToSave.to_csv(filePath)
            cachedTickerHistory[fileName]=dfToSave
            # print(f"now3 fileStart {fileStart} from {initialFileStart} fileStop {fileStop} from {initialFileStop}")              
          
    #removes extra rows from result
    print(f"extracting rows")
    dfReturn = df.loc[[index for index,row in df.iterrows() if getDate(index)>=start and getDate(index)<end]]
    print(f"returning rows")
    return dfReturn;
        