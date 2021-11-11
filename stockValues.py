# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 18:43:26 2021

@author: Nino
"""

import tickers;
import yfinance as yf
from datetime import datetime
import os.path
import csv
import urllib.request
import requests
from bs4 import BeautifulSoup
from lxml import html
import pandas as pd
import talib
from patterns import patternList,hotPatterns,applyPatterns
import certifi
import dataretriever as rd

import ssl

ssl._create_default_https_context = ssl._create_unverified_context



def getStockPatterns(tickerList):
    t= tickerList;
    stocks = []
    for ticker in t:
        if ticker["kind"]=="I": continue
        bloomberg = ticker["bloomberg"];
        stock= {"reuters": ticker["reuters"],
                 "name": ticker["name"]
                };
        today = datetime.today()
        try:
            df = rd.getHistoryData(bloomberg,"2d","1d",hotPatterns)
            #df = pd.read_csv(f"datasets/{today.strftime('%Y')}/daily/{bloomberg}")
            #currentPatterns=[]
            # for pattern in hotPatterns:
            #     patternFunction  = getattr(talib,pattern)
            #     try:
            #         result = patternFunction(df["Open"],df["High"],df["Low"],df["Close"])
            #         last = result.tail(1).values[0]
            #         if (last!=0): 
            #             if (last>0):
            #                 currentPatterns.append(patterns[pattern]+ " bullish")
            #             else:
            #                 currentPatterns.append(patterns[pattern]+ " bearish")
            #     except:
            #         currentPatterns.append(f"exception evaluating {pattern}")
            #         break;
            #currentPatterns = applyPatterns(df["Open"],df["High"],df["Low"],df["Close"], hotPatterns)
            if (len(df.index)==0): continue
           
            if len(df["bullishPatterns"][-1])>0:
                lastPatterns =df["bullishPatterns"][-1]
                strPatterns = [f"{x['name']}" for x in lastPatterns] 
                stock["patternsBullish"]= " ".join(strPatterns);
            else:
                stock["patternsBullish"]=""
            
            if len(df["bearishPatterns"][-1])>0:
                lastPatterns = df["bearishPatterns"][-1]
                strPatterns = [f"{x['name']}" for x in lastPatterns ] 
                stock["patternsBearish"]= " ".join(strPatterns);
            else:
                stock["patternsBearish"]=""
        except:
            stock["patternsBullish"]="Errors"
            stock["patternsBearish"]="Errors"
        
        info= getCachedInfo(bloomberg,ticker["isin"],ticker["etf"],ticker["kind"]);
        getTechnicalFromBorsaItaliana(info,ticker["isin"],ticker["etf"],ticker["kind"])
        
        #dividends= yf.Ticker(yahooCode).dividends;
        #recommendations= yf.Ticker(yahooCode).recommendations;
        #info= yf.Ticker(ticker["ticker"]).info;
        #stock["info"] = info;
        stock["bloomberg"] = bloomberg;
        
        if ticker["etf"]=="S":
            stock["borsaitaliana"]  = f"https://www.borsaitaliana.it/borsa/etf/scheda/{ticker['isin']}.html?lang=it";
        else:
            stock["borsaitaliana"]  = f"https://www.borsaitaliana.it/borsa/azioni/scheda/{ticker['isin']}.html?lang=it";

        
        if ("shortName" in  info):
            stock["name"] = info["shortName"];
            
            
        for field in ["resistance","support","volatility","strength"]:
            if field in info:
                stock[field]= info[field];
            
        
        maxDay = float(df["High"][-1]);
        minDay = float(df["Low"][-1]);
        # if (existsNumber(info,"dayHigh") and existsNumber(info,"dayLow") ):
        #     maxDay = float(info["dayHigh"]);
        #     minDay = float(info["dayLow"]);
        #     stock["dayRange"]= f"{minDay} - {maxDay}"
  
        if (existsNumber(info,"fiftyTwoWeekHigh") and existsNumber(info,"fiftyTwoWeekLow") ):
            maxYear = float(info["fiftyTwoWeekHigh"]);
            minYear = float(info["fiftyTwoWeekLow"]);
            stock["yearRange"]= f"{minYear} - {maxYear}"
        
        stock["value"]= float(df["Close"][-1]);
        # if ("regularMarketPrice" in  info):
        #     currClose= info["regularMarketPrice"]
        #     stock["value"]= currClose
      
        # prevClose=None
        prevClose = float(df["Close"][-2]);
        stock["value"]= prevClose;
        # if ("regularMarketPreviousClose" in info):
        #     prevClose = info["regularMarketPreviousClose"]
        # else:
        #     if ("previousClose" in info):
        #         prevClose = info["previousClose"]
                
        # if (prevClose is not None):
        #     diff = ((float(currClose) - float(prevClose))*100/float(prevClose))
        #     stock["diff"] = round(diff,2);
        #     stock["prevValue"]= prevClose;
            
        #stock["grafico"] = f"https://charts2.finviz.com/chart.ashx?t={yahooCode}&ty=c&ta=1&p=d&s=l" 
        stock["grafico"] = f"https://www.borsaitaliana.it/media/img/technicalanalysis/{ticker['isin']}_1.png"
        stocks.append(stock);
    return stocks;

def getStockTechnical(tickerList):
    t= tickerList;
    stocks = []
    for ticker in t:
        if ticker["kind"]=="I": continue
        bloomberg = ticker["bloomberg"];
        stock= {"reuters": ticker["reuters"],
                 "name": ticker["name"]
                };
        info= getCachedInfo(bloomberg,ticker["isin"],ticker["etf"],ticker["kind"]);
        getTechnicalFromBorsaItaliana(info,ticker["isin"],ticker["etf"],ticker["kind"])
        
        #dividends= yf.Ticker(yahooCode).dividends;
        #recommendations= yf.Ticker(yahooCode).recommendations;
        #info= yf.Ticker(ticker["ticker"]).info;
        stock["info"] = info["support"];
        stock["bloomberg"] = bloomberg;
        
        if ticker["etf"]=="S":
            stock["borsaitaliana"]  = f"https://www.borsaitaliana.it/borsa/etf/scheda/{ticker['isin']}.html?lang=it";
        else:
            stock["borsaitaliana"]  = f"https://www.borsaitaliana.it/borsa/azioni/scheda/{ticker['isin']}.html?lang=it";

        
        if ("shortName" in  info):
            stock["name"] = info["shortName"];
            
            
        for field in ["resistance","support","volatility","strength"]:
            if field in info:
                stock[field]= info[field];
            
            
        if (existsNumber(info,"dayHigh") and existsNumber(info,"dayLow") ):
            maxDay = float(info["dayHigh"]);
            minDay = float(info["dayLow"]);
            stock["dayRange"]= f"{minDay} - {maxDay}"
  
        if (existsNumber(info,"fiftyTwoWeekHigh") and existsNumber(info,"fiftyTwoWeekLow") ):
            maxYear = float(info["fiftyTwoWeekHigh"]);
            minYear = float(info["fiftyTwoWeekLow"]);
            stock["yearRange"]= f"{minYear} - {maxYear}"
        
        if ("regularMarketPrice" in  info):
            currClose= info["regularMarketPrice"]
            stock["value"]= currClose
      
        prevClose=None
        if ("regularMarketPreviousClose" in info):
            prevClose = info["regularMarketPreviousClose"]
        else:
            if ("previousClose" in info):
                prevClose = info["previousClose"]
                
        if (prevClose is not None):
            diff = ((float(currClose) - float(prevClose))*100/float(prevClose))
            stock["diff"] = round(diff,2);
            stock["prevValue"]= prevClose;
            
        stock["grafico"] = f"https://www.borsaitaliana.it/media/img/technicalanalysis/{ticker['isin']}_1.png"
        stocks.append(stock);
    return stocks;
    
def getStockValues(tickerList):
    t= tickerList;
    stocks = []
    for ticker in t:
        bloomberg = ticker["bloomberg"];
        stock= {"reuters": ticker["reuters"],
                 "name": ticker["name"]
                };
        info= getCachedInfo(bloomberg,ticker["isin"],ticker["etf"],ticker["kind"]);
        #dividends= yf.Ticker(yahooCode).dividends;
        #recommendations= yf.Ticker(yahooCode).recommendations;
        #info= yf.Ticker(ticker["ticker"]).info;
        stock["info"] = info;
        stock["bloomberg"] = bloomberg;
        
        if ticker["etf"]=="S":
            stock["borsaitaliana"]  = f"https://www.borsaitaliana.it/borsa/etf/scheda/{ticker['isin']}.html?lang=it";
        else:
            stock["borsaitaliana"]  = f"https://www.borsaitaliana.it/borsa/azioni/scheda/{ticker['isin']}.html?lang=it";


        
        if ("shortName" in  info):
            stock["name"] = info["shortName"];
            
        # if ("longName" in  info):
        #     stock["name"] = info["longName"];
        if existsNumber(info,"trailingAnnualDividendRate"):
            dividendAmount = float(info["trailingAnnualDividendRate"])  
        else:
            dividendAmount= "n/d";
        stock["dividend"] = f"{dividendAmount}";

        if existsNumber(info,"dividendYield"):
            dividendPerc = round(float(info["dividendYield"])*100,2)  
        else:
            dividendPerc = "n/d";
        stock["dividendYield"] = f"{dividendPerc}";
            
            
        if existsNumber(info,"52WeekChange"):
            stock["diffYear"] = round(float(info["52WeekChange"])*100,2) 
        else:
            stock["diffYear"] = "n/d";
        
        stock["pe"]="-";
        if ("trailingPE" in  info):
            stock["pe"] = round(float(info["trailingPE"]),2);
        
        if ("logo_url" in  info):
            stock["logo"]= info["logo_url"]
            
        if (existsNumber(info,"dayHigh") and existsNumber(info,"dayLow") ):
            maxDay = float(info["dayHigh"]);
            minDay = float(info["dayLow"]);
            stock["dayRange"]= f"{minDay} - {maxDay}"
  
        if (existsNumber(info,"fiftyTwoWeekHigh") and existsNumber(info,"fiftyTwoWeekLow") ):
            maxYear = float(info["fiftyTwoWeekHigh"]);
            minYear = float(info["fiftyTwoWeekLow"]);
            stock["yearRange"]= f"{minYear} - {maxYear}"
        
        currClose=None
        if ("regularMarketPrice" in  info):
            currClose= info["regularMarketPrice"]
        
        stock["value"]= currClose

        prevClose= None;
        if ("regularMarketPreviousClose" in info):
            prevClose = info["regularMarketPreviousClose"]
        else:
            if ("previousClose" in info):
                prevClose = info["previousClose"]

        if (prevClose is not None):
            diff = ((float(currClose) - float(prevClose))*100/float(prevClose))
            stock["diff"] = round(diff,2);
            stock["prevValue"]= prevClose;
              
        stock["grafico"] = f"https://www.borsaitaliana.it/media/img/technicalanalysis/{ticker['isin']}_1.png"
        stocks.append(stock);
    return stocks;

def existsNumber(dict,key):
        if not key in dict: 
            return False;        
        n= dict[key];
        return isNumber(n)


def isNumber(n):
        if n is None:
            return False;
        if (isinstance(n,float) or isinstance(n,int) or n.isnumeric()):
            return True;
        xx = n.replace('.','',1);
        return  xx.isnumeric();    
    

    
    
def getCachedInfo(tickerCode,isin,etf,kind):    
    
     # yData= yf.Ticker(tickerCode);
     # info = yData.info;
     # return info
          
    fileName = f"{tickerCode}_info_"+datetime.today().strftime('%Y-%m-%d')+".csv"
    filePath = "cache\\"+fileName;
    if os.path.isfile(filePath):
            #legge il file
            with open(filePath, encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                result = {} 
                for row in reader:
                    try:
                        result[ row[0]]= row[1]
                    except:
                        pass
            
            
            #converte il file in un oggetto
                    
            #lo restituisce
            return result;
    else:
          #legge i dati da internet
          yData= yf.Ticker(tickerCode);
          info = yData.info;
          
          if (kind!="I"):
              if not existsNumber(info,"dayLow") or not existsNumber(info,"dayHigh"):
                  getMinMaxFromBorsaItaliana(info,isin,etf,kind);
              if not existsNumber(info,"fiftyTwoWeekLow") or not existsNumber(info,"fiftyTwoWeekHigh"):
                  getMinMaxFromBorsaItaliana(info,isin,etf,kind);
                      
              if not existsNumber(info,"52WeekChange") :              
                  getMinMaxFromBorsaItaliana(info,isin,etf,kind);
              
          toWrite = []              
          for k,v in info.items():
                toWrite.append([k,v]);
          
          #scrive i dati nel file
          with open(filePath, 'w',newline="", encoding='utf-8') as csvfile:            
              writer = csv.writer(csvfile)                  
              writer.writerow(["key","value"]);
              writer.writerows(toWrite);
          
            
          #restituire i dati letti
          return yData.info;


def StringAfterPatternBorsaItaliana(text,pattern1,pattern2):
        posPattern= text.find(pattern1);
        posBeforeValuePattern = text.find(pattern2,posPattern);
        posValuePattern = text.find(">",posBeforeValuePattern)+1;
        posStopValuePattern = text.find("<",posValuePattern);
        strPattern = text[posValuePattern:posStopValuePattern];
        return strPattern;


def getMinMaxFromBorsaItaliana(info,isin,etf,kind):
        url=''
        if (isin is None ): 
             info["dayLow"]='n.a.'
             info["dayHigh"]='n.a.'
             return
        isin = isin.strip()
        if etf=="S":
            url = f"https://www.borsaitaliana.it/borsa/etf/scheda/{isin}.html?lang=it";
        else:
            if (kind=='E'):
                url = f"https://www.borsaitaliana.it/borsa/azioni/scheda/{isin}.html?lang=it";
            else:
                url= f"https://www.borsaitaliana.it/borsa/obbligazioni/mot/btp/scheda/{isin}.html?lang=it"



        #print(f'getMinMaxFromBorsaItaliana: loading {url}',flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response =  urllib.request.urlopen(req, cafile=certifi.where());
        #print(f'getMinMaxFromBorsaItaliana: loading {url} done',flush=True)
        html = response.read()
        text = html.decode()
        
        strMinOggi= StringAfterPatternBorsaItaliana(text,"Min oggi","t-text -right");
        info["dayLow"] = strMinOggi.replace(",",".")
        
        strMaxOggi= StringAfterPatternBorsaItaliana(text,"Max oggi","t-text -right");
        info["dayHigh"] = strMaxOggi.replace(",",".")

        strMinAnno= StringAfterPatternBorsaItaliana(text,"Min Anno","t-text -right").strip();        
        minAnno = strMinAnno.split("-")[0].strip();
        info["fiftyTwoWeekLow"] = minAnno.replace(",",".")
        
        strMaxAnno= StringAfterPatternBorsaItaliana(text, "Max Anno","t-text -right").strip();
        maxAnno = strMaxAnno.split("-")[0].strip();
        info["fiftyTwoWeekHigh"] = maxAnno.replace(",",".")
        
        perfAnno= StringAfterPatternBorsaItaliana(text, "Performance 1 anno","t-text -right").strip();
        perf = perfAnno.strip();
        if (perf!=""):
            info["52WeekChange"] = float(perf.replace(",",".").replace("+","").replace("%","").strip())/100;
        
        
        
    
def getTechnicalFromBorsaItaliana(info,isin,etf,kind):
        url=''
        if (isin is None ): 
             info["support"]="n.a."
             return
        isin = isin.strip()
        
        
        if etf=="S":
            url = f"https://www.borsaitaliana.it/borsa/etf/analisi-tecnica.html?isin={isin}&lang=it";
        else:
            url = f"https://www.borsaitaliana.it/borsa/azioni/analisi-tecnica.html?isin={isin}&lang=it";
           

        try:
            #print(f'getTechnicalFromBorsaItaliana: loading {url}',flush=True)
            response =  urllib.request.urlopen(url,cafile=certifi.where());
            #print(f'getTechnicalFromBorsaItaliana: loading {url} done',flush=True)
        except:
            info["support"]="n.a."
            return
        html = response.read()
        text = html.decode()
        
        supp1= StringAfterPatternBorsaItaliana(text,"Supporto 1","strong");
        supp1= supp1.replace(",",".")
        if isNumber(supp1):
             supp1 = round(float(supp1.replace(",",".")),2)
        
        supp2= StringAfterPatternBorsaItaliana(text,"Supporto 2","strong");
        supp2=supp2.replace(",",".")
        if isNumber(supp2):
            supp2 = round(float(supp2.replace(",",".")),2)

        if isNumber(supp1) and isNumber(supp2):
            info["support"]=f"{supp2} {supp1}"
        else: 
            info["support"]=f"Non numbers {supp2} {supp1}"

        res1= StringAfterPatternBorsaItaliana(text,"Resistenza 1","strong");
        res1=res1.replace(",",".")
        if isNumber(res1):
            res1 = round(float(res1),2)
        
        res2= StringAfterPatternBorsaItaliana(text,"Resistenza 2","strong");
        res2=res2.replace(",",".")
        if isNumber(res2):
            res2 = round(float(res2.replace(",",".")),2)

        if isNumber(res1) and isNumber(res2):
            info["resistance"]=f"{res1} {res2}"        
        
        
        s= StringAfterPatternBorsaItaliana(text,"Forza Relativa","strong");
        s = s.replace(",",".")
        if isNumber(s) :
            info["strength"]=f"{s}"        
    
        v= StringAfterPatternBorsaItaliana(text,"Volatilit&agrave;","strong");
        v = v.replace(",",".")
        if isNumber(v) :
            info["volatility"]=f"{s}"        
    

def reloadDailyData():
    t= tickers.getTickers();
    stocks = []
    for ticker in t:
        bloomberg = ticker["bloomberg"];
        today = datetime.today()
        df  = yf.download(bloomberg, start=today.strftime("%Y-01-01"), end=today.strftime("%Y-%m-%d"), period="1d")
        df.to_csv(f"datasets/{today.strftime('%Y')}/daily/{bloomberg}")
                          
        
def getGraph(bloomberg,periodicity,frequency):
        data = yf.download(tickers=bloomberg,period=periodicity,interval=frequency)


        
                          
