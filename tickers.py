# -*- coding: utf-8 -*-

import csv
#from bs4  import BeautifulSoup

tickers= [];
# tickers_xls = pd.ExcelFile(r"data/tickers.xlsx") #use r before absolute file path 
    

def defaultStock():
    return {
        "reuters":"",
        "name":"",
        "bloomberg":"",
        "isin":"",
        "kind":"E",
        "etf":"N",
        "checkedKindE":"checked='checked'",
        "checkedKindB":"",
        "checkedKindI":"",
        "checkedEtfN":"checked='checked'",
        "checkedEtfS":"",
        "checkedETC":"",
        "checkedETN":"",
        }


def getStockToEdit(c):
    matches = [x for x in tickers if x["reuters"] == c]
    if (matches): 
        m= matches[0].copy();
        m["checkedKindE"]="";
        if (m["kind"] == "E"):
            m["checkedKindE"]="checked='checked'";
            
        m["checkedKindB"]="";
        if (m["kind"] == "B"):
            m["checkedKindB"]="checked='checked'";
            
        m["checkedKindI"]="";
        if (m["kind"] == "I"):
            m["checkedKindI"]="checked='checked'";
            
        m["checkedETF"]=""
        if (m["etf"] == "S"):
            m["checkedEtfS"]="checked='checked'";
        if (m["etf"] == "N"):
            m["checkedEtfN"]="checked='checked'";
        if (m["etf"] == "ETC"):
            m["checkedETC"]="checked='checked'";
        if (m["etf"] == "ETN"):
            m["checkedETN"]="checked='checked'";
        return m;
    return None;



def getTickerByIsin(isin):
    matches = [x for x in tickers if x["isin"] == isin]
    if (matches): 
        return matches[0];
    return None

def getTickerByBloomberg(c):
    matches = [x for x in tickers if x["bloomberg"] == c]
    if (matches): 
        return matches[0];
    return None
    
def getTickerByReuters(c):
    matches = [x for x in tickers if x["reuters"] == c]
    if (matches): 
        return matches[0];
    return None

def reload():
    tickers.clear();
    with open('data/tickers.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            tickers.append(row)
    line_count = len(tickers)
    print(f'tickers.csv: Processed {line_count} lines.')
    
    
def saveData():    
    with open('data/tickers.csv', 'w',newline="") as csvfile:
        fieldnames = ['reuters', 'name','bloomberg','isin','kind','etf']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in tickers:
            writer.writerow(row)
    
    
    
def getTickers():
    return  tickers;



def deleteTickerByReuters(reutersCode):    
    if (reutersCode is None) :
        return
    
    if (reutersCode == "") :
        return
    
    match = getTickerByReuters(reutersCode)
    if (match is not None) : 
        tickers.remove(match);
        saveData()
    


def addTicker(reuters,name,bloomberg,isin,kind,etf):
    if (reuters is None) :
        return "You must enter Reuters stock code"
    if (reuters=="") :
        return "You must enter Reuters stock code"
    m =  getTickerByReuters(reuters)
    if (m is not None) : 
        m["reuters"]=reuters.strip();
        m["name"]=name.strip();
        m["bloomberg"]=bloomberg.strip();
        m["isin"]=isin.strip();
        m["kind"]=kind;
        
        m["etf"]=etf;
        saveData()
        return reuters + " has been updated"
    
    tickers.append({'reuters':reuters,"name":name,"bloomberg":bloomberg,"isin":isin,"kind":kind,"etf":etf})
    tickers.sort( key=lambda x: x["name"])
    saveData()
    
reload();
    
