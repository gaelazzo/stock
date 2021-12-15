# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:43:07 2021

@author: Nino
"""

from datetime import datetime, timedelta
import csv
import tickers as tk

from commissionCalculator import CommissionCalculator
from dateutil.relativedelta import relativedelta
from datetime import date
import dataretriever as retriever

portfolioList = []  # name,code,price,q,currPrice,marketValue,percDay,diffDay,percTotal,diffTotal, prevPrice,totprice,totComm
openPositions = []
closedPositions = []
commCalc = CommissionCalculator()

totals = {"totalValue": 0,
          "totalPL": 0,
          "dailyPL": 0,
          "closedPL": 0,
          "tax": 0,
          "credits": 0,
          "comm": 0
          }


def getPortfolioTickers():
    result = []
    usedCodes = []
    for openPos in openPositions:
        if openPos["bloomberg"] in usedCodes:
            continue
        usedCodes.append(openPos["bloomberg"])
        result.append(tk.getTickerByBloomberg(openPos["bloomberg"]));

    return result


# openpositions.csv:   name,code,ticker,buyprice,q,buydate,buycomm
def reload():
    totals = {"totalValue": 0,
              "totalPL": 0,
              "dailyPL": 0,
              "closedPL": 0,
              "tax": 0,
              "credits": 0,
              "comm": 0
              }

    commCalc.reset()
    reloadOpenPositions()
    reloadClosedPositions()
    evaluateClosedPositions()
    evaluatePortfolio()


def defaultOpenPosition():
    return {
        "name": "",
        "reuters": "",
        "buyprice": "",
        "q": "",
        "buydate": datetime.today().strftime("%d/%m/%Y"),
        "buycomm": 6.0
    }


def reloadOpenPositions():
    openPositions.clear()
    with open('data/openpositions.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            code = row["reuters"]
            m = tk.getTickerByBloomberg(row["bloomberg"])
            if (code is None):
                row["reuters"] = m["reuters"]
                row["name"] = m["name"]
            row["buyprice"] = float(row["buyprice"])
            row["buydate"] = datetime.strptime(row["buydate"], "%d/%m/%Y")
            openPositions.append(row)
            commCalc.incOperation(row["buydate"])
            totalBuy = row["buyprice"] * float(row["q"])

            if (row["buydate"].date() < date.today()
                    and m["kind"] == "E" and m["etf"] == "N"):
                row["tobin"] = round(totalBuy * 0.001, 2)

    openPositions.sort(key=lambda x: x["buydate"])
    line_count = len(openPositions)
    print(f'openpositions.csv: Processed {line_count} lines.')


def reloadClosedPositions():
    closedPositions.clear()

    totals["tax"] = 0
    totals["credits"] = 0
    with open('data/closedpositions.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            code = row["reuters"]
            if (code is None):
                print(f'examining {row}')
                m = tk.getTickerByBloomberg(row["bloomberg"])
                row["reuters"] = m["reuters"]
                row["name"] = m["name"]
            else:
                m = tk.getTickerByReuters(row["reuters"])
                row["name"] = m["name"]
            if (row["tax"] != ""):
                totals["tax"] = totals["tax"] + float(row["tax"])
            if (row["credits"] != ""):
                totals["credits"] = round(totals["credits"] + float(row["credits"]), 2)
            if ("buycomm" not in row):
                row["buycomm"] = 0
            if ("sellcomm" not in row):
                row["sellcomm"] = 0
            closedPositions.append(row)
            row["buydate"] = datetime.strptime(row["buydate"], "%d/%m/%Y")
            row["selldate"] = datetime.strptime(row["selldate"], "%d/%m/%Y")
            for f in ["buycomm", "sellcomm", "buyprice", "sellprice", "tax", "credits"]:
                row[f] = float(row[f])
            commCalc.incOperation(row["buydate"])
            commCalc.incOperation(row["selldate"])

    closedPositions.sort(key=lambda x: x["buydate"])
    line_count = len(closedPositions)
    print(f'closedpositions.csv: Processed {line_count} lines.')


def saveOpenPositions():
    with open('data/openpositions.csv', 'w', newline="") as csvfile:
        fieldnames = ['name', 'reuters', 'bloomberg', 'buyprice', 'q', 'buydate', 'buycomm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in openPositions:
            rr = dict(
                (k, v) for k, v in row.items() if k in fieldnames
            )
            rr["buydate"] = datetime.strftime(rr["buydate"], "%d/%m/%Y")
            writer.writerow(rr)


def saveClosedPositions():
    with open('data/closedpositions.csv', 'w', newline="") as csvfile:
        fieldnames = ['name', 'reuters', 'bloomberg', 'buyprice', 'q', 'buydate', 'sellprice', 'selldate', 'totcomm',
                      'tax', 'credits', "buycomm", "sellcomm"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in closedPositions:
            rr = dict(
                (k, v) for k, v in row.items() if k in fieldnames
            )
            rr["buydate"] = datetime.strftime(rr["buydate"], "%d/%m/%Y")
            rr["selldate"] = datetime.strftime(rr["selldate"], "%d/%m/%Y")
            writer.writerow(rr)

        # evaluate commissions of open and closed operations


def evaluateCommissions():
    operations = []
    for openPos in openPositions:
        operations.append({"kind": "B", "date": openPos["buydate"], "reuters": openPos["reuters"],
                           "price": float(openPos["buyprice"]), "q": int(openPos["q"]), "row": openPos})

    for closedPos in closedPositions:
        operations.append({"kind": "B", "date": closedPos["buydate"], "reuters": closedPos["reuters"],
                           "price": float(closedPos["buyprice"]), "q": int(closedPos["q"]), "row": closedPos})
        operations.append({"kind": "S", "date": closedPos["selldate"], "reuters": closedPos["reuters"],
                           "price": float(closedPos["sellprice"]), "q": int(closedPos["q"]), "row": closedPos})

    operations.sort(key=lambda x: x["date"])

    calculator = CommissionCalculator()
    for op in operations:
        if ("evaluated" in op):
            continue
        opSet = [o for o in operations if
                 o["kind"] == op["kind"] and o["date"] == op["date"] and o["price"] == op["price"] and o["reuters"] ==
                 op["reuters"]]
        totQ = sum(int(c["q"]) for c in opSet)
        totAmount = round(totQ * op["price"], 2)
        calculator.incOperation(op["date"])
        commission = calculator.getCommission(totAmount, op["date"])
        # n= len(opSet)
        commField = "buycomm" if op["kind"] == "B" else "sellcomm"
        for o in opSet:
            o["evaluated"] = True
            # currComm= round(commission/n,2)
            # if ("selldate" in op["row"]): #only fixes commissions on closed positions for now
            o["row"][commField] = commission;  # currComm;
            commission = 0  # commission-currComm;
            # n=n-1

    for closedPos in closedPositions:
        closedPos["totcomm"] = round(float(closedPos["buycomm"]) + float(closedPos["sellcomm"]), 2)


def evaluateCurrPrice(stocks):
    usedCodes = []
    for s in stocks:
        if s["bloomberg"] in usedCodes:
            continue
        usedCodes.append(s["bloomberg"])

    prices = retriever.getLastPrices(usedCodes)
    currValues = prices["currPrices"]

    for s in stocks:
        symbol = s["bloomberg"]

        s["buyprice"] = round(float(s["buyprice"]), 6)
        s["currprice"] = round(currValues[symbol], 2)

        if not "sellprice" in s:
            s["sellprice"] = float(s["currprice"])

        if "currprice" in s:
            s["diffTotal"] = round((float(s["sellprice"]) - s["currprice"]) * int(s["q"]), 2)
        else:
            s["diffTotal"] = round((float(s["sellprice"]) - s["buyprice"]) * int(s["q"]), 2)


def evaluatePortfolio():
    usedCodes = []
    totals["totalValue"] = 0;
    totals["totalPL"] = 0;
    totals["dailyPL"] = 0;
    evaluateCommissions();

    # for openPos in openPositions:
    #     if openPos["bloomberg"] in usedCodes:
    #         continue
    #     usedCodes.append(openPos["bloomberg"])    
    # codesToRequest = " ".join(usedCodes);
    # print (codesToRequest)

    usedCodes = [x["bloomberg"] for x in openPositions]
    prices = retriever.getLastPrices(usedCodes)
    currValues = prices["currPrices"]
    prevClose = prices["prevClose"]
    # prevClose = {}
    # data = yf.download(tickers=codesToRequest,period="2d",interval="1d") if len(usedCodes)>0 else []

    # for symbol in usedCodes:        
    #     try:           
    #         currValues[symbol] = round(data["Close"][symbol][1],6) if len(usedCodes)>1 else round(data["Close"][1],6)
    #         prevClose[symbol] = round(data["Close"][symbol][0],6) if len(usedCodes)>1 else round(data["Close"][0],6)
    #         if (prevClose[symbol]!=prevClose[symbol] or prevClose[symbol] is None):
    #             #currValues[symbol] = round(data["Close"][symbol][0],6) ##first row in this case is the current value
    #             ticker= tk.getTickerByBloomberg(symbol);
    #             prevClose[symbol] = float(getStockTechnical([ticker])[0]["prevValue"]);
    #     except:
    #         ticker= tk.getTickerByBloomberg(symbol);
    #         ##first row in this case is the current value
    #         currValues[symbol] = round(data["Close"][symbol][0],6) if len(usedCodes)>1 else round(data["Close"][0],6)

    #         prevClose[symbol] =  float(getStockTechnical([ticker])[0]["prevValue"]);
    #         pass;

    totalPrevValue = 0
    startValue = 0
    totalDiff = 0
    totalDiffDay = 0;
    portfolioList.clear()
    for openPos in openPositions:
        # openpositions.csv:   name,code,buyprice,q,buydate,buycomm
        stock = next((t for t in portfolioList if t["reuters"] == openPos["reuters"]), None)
        if (stock is None):
            tickRow = tk.getTickerByReuters(openPos["reuters"]);
            symbol = tickRow["bloomberg"]
            stock = {"reuters": openPos["reuters"],
                     "bloomberg": openPos["bloomberg"],
                     "buyprice": 0.0,
                     "q": 0,
                     "name": tickRow["name"],
                     "totPrice": 0.0,
                     "totComm": 0,
                     "marketValue": 0,
                     "diffDay": 0,
                     "percDay": 0,
                     "prevMarketValue": 0
                     }
            # evaluate stock["prevPrice"] and stock["currPrice"]
            stock["currPrice"] = currValues[symbol] if symbol in currValues else 0
            stock["prevPrice"] = prevClose[symbol] if symbol in prevClose else 0
            portfolioList.append(stock);

        stock["q"] = int(stock["q"]) + int(openPos["q"])  # total quantity
        stock["marketValue"] = float(stock["q"]) * stock["currPrice"]  ##total market value

        stock["totPrice"] = float(stock["totPrice"]) + float(openPos["buyprice"]) * float(
            openPos["q"])  ##total buy price
        stock["buyprice"] = stock["totPrice"] / float(stock["q"])  ## medium buy price
        stock["totComm"] = float(stock["totComm"]) + float(openPos["buycomm"])

        if (openPos["buydate"].date() == datetime.today().date()):
            prevPrice = float(openPos["buyprice"])
        else:
            prevPrice = stock["prevPrice"]  # prevClose[symbol]

        stock["prevMarketValue"] = stock["prevMarketValue"] + round(float(prevPrice) * float(openPos["q"]), 2)

    for stock in portfolioList:
        stock["dayColor"] = ""

        if (stock["prevMarketValue"] is not None and stock["prevMarketValue"] != 0):
            stock["diffDay"] = stock["marketValue"] - stock["prevMarketValue"]  ##total market value -
            stock["percDay"] = round((stock["diffDay"] / stock["prevMarketValue"]) * 100, 2)
            if (stock["diffDay"] > 0):
                stock["dayColor"] = "greenFont"
            if (stock["diffDay"] < 0):
                stock["dayColor"] = "redFont"

        # stock["percTotal"]= round(((stock["marketValue"]-stock["totPrice"]) / stock["totPrice"])*100,2)
        stock["diffTotal"] = stock["marketValue"] - stock["totPrice"] - stock["totComm"]
        stock["percTotal"] = round(((stock["diffTotal"]) / stock["totPrice"]) * 100, 2)

        stock["totColor"] = ""
        if (stock["diffTotal"] > 0):
            stock["totColor"] = "greenFont"
        if (stock["diffTotal"] < 0):
            stock["totColor"] = "redFont"

        for field in ["currPrice", "buyprice", "prevPrice"]:
            stock[field] = fixValue(stock[field])

        for field in ["marketValue", "diffTotal", "diffDay", "totPrice"]:
            if stock[field] is not None: stock[field] = round(stock[field], 2)

        startValue = startValue + stock["totPrice"]
        if stock["prevMarketValue"] is not None: totalPrevValue = totalPrevValue + stock["prevMarketValue"]
        totalDiffDay = totalDiffDay + stock["diffDay"]
        totalDiff = totalDiff + stock["diffTotal"]

        # stock["grafico"] = f"https://www.borsaitaliana.it/media/img/technicalanalysis/{ticker['isin']}_1.png"

    for stock in portfolioList:
        totals["totalValue"] = round(float(totals["totalValue"]) + stock["marketValue"], 2)
        totals["totalPL"] = round(totals["totalPL"] + stock["diffTotal"], 2)
        totals["totalPercPL"] = round(100.0 * totals["totalPL"] / startValue, 2) if startValue > 0 else None
        # stock["marketValue"] = round(stock["marketValue"]/1000,1)

    totals["dailyPL"] = round(totalDiffDay, 2)
    totals["dailyPercPL"] = round(100.0 * totalDiffDay / startValue, 2) if startValue > 0 else None

    # totals["totalValue"] = '{:0,.2f}'.format(totals["totalValue"])
    return portfolioList


def fixValue(n):
    if (n > 1 or n < 1):
        return round(n, 4);
    return round(n, 4);


def getPortfolio():
    # logging.info("getPortfolio called")
    evaluatePortfolio()
    # logging.info(openPositions)
    return portfolioList


def addOpenPosition(reuters, price, q, buydate):
    if (reuters is None):
        return "You must enter a stock code"
    if (reuters == ""):
        return "You must enter a stock code"
    m = tk.getTickerByReuters(reuters)
    if (m is None):
        return f"{reuters} does not exists"

    # name,code,buyprice,q,buydate,buycomm

    openPosition = {}
    openPosition["reuters"] = reuters;
    openPosition["name"] = m["name"];
    openPosition["bloomberg"] = m["bloomberg"];
    openPosition["buyprice"] = price;
    openPosition["q"] = q

    buydate = datetime.strptime(buydate, "%d/%m/%Y")

    openPosition["buydate"] = buydate;
    commCalc.incOperation(buydate)
    buycomm = commCalc.getCommission(round(price * q, 2), buydate)
    openPosition["buycomm"] = buycomm;

    openPositions.append(openPosition);

    saveOpenPositions()
    evaluatePortfolio()
    return f"{m['name']} has been added"


def closePosition(reuters, sellprice, q, selldate):
    if (reuters is None):
        return "You must enter a stock code"
    if (reuters == ""):
        return "You must enter a stock code"
    m = tk.getTickerByReuters(reuters)
    if (m is None):
        return f"{reuters} does not exists"

    # name,code,buyprice,q,buydate,buycomm
    ## Select the last open position with that code
    tickerOpenPositions = [x for x in openPositions if x["reuters"] == reuters]

    tickerOpenPositions.sort(key=lambda x: x["buydate"])

    ##only once
    selldate = datetime.strptime(selldate, "%d/%m/%Y")

    while (q > 0):
        curr = tickerOpenPositions.pop()
        currOpen = int(curr["q"])
        toSell = min(q, currOpen)
        closePosition = {"name": m["name"],
                         "reuters": reuters,
                         "bloomberg": m["bloomberg"],
                         "q": toSell,
                         "buyprice": float(curr["buyprice"]),
                         "buydate": curr["buydate"],
                         "sellprice": float(sellprice),
                         "selldate": selldate
                         }
        closedPositions.append(closePosition);
        if (toSell == currOpen):
            openPositions.remove(curr);
        else:
            curr["q"] = int(curr["q"]) - toSell;
        q = q - toSell;

    evaluateCommissions();

    evaluatePortfolio()
    saveOpenPositions()
    saveClosedPositions()
    evaluateClosedPositions()
    return f"{m['name']} has been closed"


def fixImportNum(s):
    s = s.replace("€", "")
    s = s.replace(".", "")
    s = s.replace(",", ".")
    return float(s)


def importOpenPosition(filename):
    openPositions.clear()

    logf = open("importOpenPosition.log", "w")
    logf.write(f"test write")
    logf.close()

    with open(filename, encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file,
                                quotechar='"',
                                quoting=csv.QUOTE_ALL
                                )

        for row in reader:
            # logf = open("importOpenPosition.log", "w")
            # logf.write("#".join(row.keys()))
            # logf.close()
            reuters = row["Simbolo"].replace(".MI", "")
            stock = {"reuters": reuters}
            stock["name"] = row["Nome"]
            stock["buydate"] = datetime.strptime(row["Data Ap."], "%d/%m/%Y")
            q = int(round(float(row["Quantità"]), 0))
            stock["q"] = q
            m = tk.getTickerByReuters(reuters)
            if (m is None):
                logf = open("importOpenPosition.log", "w")
                logf.write(f"{reuters} not found")
                logf.close()
            if (m is not None):
                stock["bloomberg"] = m["bloomberg"]
                stock["name"] = m["name"]

            stock["buyprice"] = fixImportNum(row["Prez Ap."])
            openPositions.append(stock)

    openPositions.sort(key=lambda x: x["buydate"])
    saveOpenPositions()
    evaluatePortfolio()


def getMinDateFromRange(dataRange):
    minDay = None

    if (dataRange == "day" or dataRange == "difftoday"):
        if datetime.today().weekday() > 0:
            minDay = datetime.today().date()
        else:
            minDay = datetime.today().date() - timedelta(days=2)

    if (dataRange == "yesterday"):
        minDay = datetime.today().date() - timedelta(days=1)

    if (dataRange == "week"):
        minDay = datetime.today().date() - timedelta(weeks=1)

    if (dataRange == "month"):
        minDay = datetime.today().date() - relativedelta(months=1)

    if (dataRange == "month3"):
        minDay = datetime.today().date() - relativedelta(months=3)

    if (dataRange == "month4"):
        minDay = datetime.today().date() - relativedelta(months=4)

    if (dataRange == "month6"):
        minDay = datetime.today().date() - relativedelta(months=6)

    if (dataRange == "year"):
        today = datetime.today().date()
        minDay = datetime(today.year, 1, 1).date()

    return minDay


def getClosedPositions(dataRange="all"):
    if dataRange == "all":
        return sorted(closedPositions, key=lambda x: x["selldate"], reverse=True)

    minDay = getMinDateFromRange(dataRange)
    if minDay is None:
        return sorted(closedPositions, key=lambda x: x["selldate"], reverse=True)

    operations = [x for x in closedPositions if x["selldate"].date() >= minDay]
    return sorted(operations, key=lambda x: x["selldate"], reverse=True)


def getOpenPositions(dataRange="all"):
    if dataRange == "all":
        return openPositions

    minDay = getMinDateFromRange(dataRange)
    if minDay is None:
        return openPositions

    operations = [x for x in openPositions if x["buydate"].date() >= minDay]

    return operations


def getOpenTotals(opList=None):
    if opList is None:
        return totals;

    DetailedTotals = {
        "closedPL": 0,
        "comm": 0,
        "tobin": 0,
        "net": 0}

    for op in opList:
        for field in ["tobin"]:
            if field in op:
                DetailedTotals[field] = round(DetailedTotals[field] + op[field], 2)
        comm = float(op["buycomm"])
        DetailedTotals["comm"] = round(DetailedTotals["comm"] + comm, 2);
        totalBuy = float(op["buyprice"]) * float(op["q"])
        totalSell = float(op["sellprice"]) * float(op["q"])  # -comm
        diff = round(totalSell - totalBuy, 2);
        DetailedTotals["closedPL"] = round(DetailedTotals["closedPL"] + diff, 2);

    DetailedTotals["net"] = round(DetailedTotals["closedPL"] - DetailedTotals["tobin"] - DetailedTotals["comm"], 2)
    return DetailedTotals


def getTotals(opList=None):
    if opList is None:
        return totals;

    DetailedTotals = {
        "tax": 0,
        "credits": 0,
        "closedPL": 0,
        "comm": 0,
        "tobin": 0,
        "net": 0}

    for op in opList:
        for field in ["tax", "credits", "tobin"]:
            if field in op:
                DetailedTotals[field] = round(DetailedTotals[field] + op[field], 2)
        comm = float(op["totcomm"])
        DetailedTotals["comm"] = round(DetailedTotals["comm"] + comm, 2);
        totalBuy = float(op["buyprice"]) * float(op["q"])
        totalSell = float(op["sellprice"]) * float(op["q"])  # -comm
        diff = round(totalSell - totalBuy, 2);
        DetailedTotals["closedPL"] = round(DetailedTotals["closedPL"] + diff, 2);

    DetailedTotals["net"] = round(
        DetailedTotals["closedPL"] - DetailedTotals["tobin"] - DetailedTotals["comm"] - DetailedTotals["tax"], 2)
    return DetailedTotals


def evaluateClosedPositions():
    totals["tax"] = 0
    totals["credits"] = 0
    totals["closedPL"] = 0
    totals["comm"] = 0
    totals["tobin"] = 0
    totals["net"] = 0
    evaluateCommissions();

    closedPositions.sort(key=lambda x: x["selldate"])
    for pos in closedPositions:
        comm = float(pos["totcomm"])
        totals["comm"] = round(totals["comm"] + comm, 2);
        totalBuy = float(pos["buyprice"]) * float(pos["q"])
        totalSell = float(pos["sellprice"]) * float(pos["q"])  # -comm
        diff = round(totalSell - totalBuy, 2);

        totals["closedPL"] = round(totals["closedPL"] + diff, 2);

        pos["PL"] = round(diff - comm, 2);
        pos["percPL"] = round(100 * diff / totalBuy, 2)
        pos["tax"] = 0;
        pos["credits"] = 0;
        taxrate = 0.26
        m = tk.getTickerByReuters(pos["reuters"])
        if (m["kind"] == "B"):
            taxrate = 0.125;
        pos["tobin"] = 0
        if (pos["buydate"] != pos["selldate"]
                and m["kind"] == "E"
                and m["etf"] == "N"):
            pos["tobin"] = round(totalBuy * 0.001, 2)
            totals["tobin"] = round(totals["tobin"] + pos["tobin"], 2)

        if (diff < 0):
            pos["credits"] = -diff
            totals["credits"] = round(totals["credits"] - diff, 2)
        else:
            compensazione = 0
            if (m["etf"] == "N"):
                compensazione = round(min(totals["credits"], diff), 2)
            taxable = diff - compensazione;
            tax = round(taxable * taxrate, 2)
            pos["tax"] = tax;
            totals["tax"] = round(totals["tax"] + tax, 2);
            totals["credits"] = round(totals["credits"] - compensazione, 2)
            pos["credits"] = -round(compensazione, 2)

    totals["net"] = round(totals["closedPL"] - totals["tobin"] - totals["comm"] - totals["tax"], 2)

    closedPositions.sort(key=lambda x: x["buydate"])
    saveClosedPositions();


def evaluateFilteredClosedPositions(reuters, dataRange="all"):
    rows = []

    minDay = getMinDateFromRange(dataRange)

    for pos in closedPositions:
        if (pos["reuters"] != reuters):
            continue
        if (minDay is not None):
            sellDate = datetime.strptime(pos["selldate"], "%d/%m/%Y")
            if (sellDate < minDay):
                continue;
        rows.append(pos)

    return {"totals": getTotals(rows), "rows": rows}


def importClosedPosition(filename):
    closedPositions.clear()

    logf = open("importClosedPosition.log", "w")
    logf.write(f"test write")
    logf.close()

    with open(filename, encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file,
                                quotechar='"',
                                quoting=csv.QUOTE_ALL
                                )

        for row in reader:
            # logf = open("importOpenPosition.log", "w")
            # logf.write("#".join(row.keys()))
            # logf.close()
            reuters = row["Simbolo"].replace(".MI", "")
            stock = {"reuters": reuters}
            stock["name"] = row["Nome"]
            stock["buydate"] = row["Data Ap."]
            stock["selldate"] = row["Data Chiusura"]
            q = int(round(float(row["Quantità"]), 0))
            stock["q"] = q
            m = tk.getTickerByReuters(reuters)
            if (m is None):
                logf = open("importClosedPosition.log", "w")
                logf.write(f"{reuters} not found")
                logf.close()
            if (m is not None):
                stock["bloomberg"] = m["bloomberg"]

            openPrice = fixImportNum(row["Prez Ap."])
            stock["buyprice"] = openPrice
            closePrice = fixImportNum(row["Prez Ch."])
            stock["sellprice"] = closePrice
            profPerditaNetto = fixImportNum(row["Netto Profitti/Perdite"])
            diffNetta1 = closePrice - openPrice
            diffNettaTot = diffNetta1 * q;
            commissioni = diffNettaTot - profPerditaNetto
            stock["totcomm"] = round(commissioni, 2)

            closedPositions.append(stock)

        closedPositions.sort(key=lambda x: datetime.strptime(x["buydate"], "%d/%m/%Y"))
        saveClosedPositions();


reload()
