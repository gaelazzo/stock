# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:03:13 2021
@author: Nino
"""

import math
from flask import render_template

import logging

# pip install  .\TA_Lib-0.4.21-cp310-cp310-win_amd64.whl

from patterns import patternList, hotPatterns
import tickers as tk

import stockValues as stockVal
import portfolio as pf
import dataretriever as dr

import json
import analyzer as an

import plotly

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import ssl

import os
from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'csv'}

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.template_filter()
def format_datetime(value, fmt='date'):
    if fmt == 'medium':
        fmt = "%Y-%m-%d %HH:%mm"
    elif fmt == 'date':
        fmt = "%Y-%m-%d"
    return value.strftime(fmt)


logging.basicConfig()


@app.route("/")
def hello_world():
    return render_template("index.html", patterns=patternList)


@app.route("/restart")
def reload():
    tk.reload()
    pf.reload()
    an.reload()
    return render_template("index.html", patterns=patternList)


@app.route("/stocks/delTicker", methods=['POST'])
def delTicker():
    ticker = request.form.get("reuters")
    tk.deleteTickerByReuters(ticker)
    return render_template("tickers.html", tickers=tk.getTickers(),
                           msg=f'ticker {ticker} deleted',
                           stock=tk.defaultStock())


@app.route("/stocks/editTicker", methods=['GET'])
def editTicker():
    ticker = request.args.get("reuters")
    app.logger.info('Processing default request')

    app.logger.info(ticker)
    s = tk.getStockToEdit(ticker)
    return render_template("tickers.html", tickers=tk.getTickers(),
                           stock=s)


@app.route("/stocks/addTicker", methods=['POST'])
def addTicker():
    # url = f"https://it.finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"

    # page = requests.get(url).text
    # print(page)
    # return page
    # # soup = BeautifulSoup( page)

    msg = tk.addTicker(request.form.get('reuters').strip(),
                       request.form.get('name'),
                       request.form.get('bloomberg'),
                       request.form.get('isin'),
                       request.form.get('kind'),
                       request.form.get('etf'))
    return render_template("tickers.html", tickers=tk.getTickers(),
                           msg=msg,
                           stock=tk.defaultStock())


@app.route("/stocks")
def showTickers():
    return render_template("tickers.html",
                           tickers=tk.getTickers(),
                           stock=tk.defaultStock())


@app.route("/data/graph", methods=['GET'])
def getGraphData():
    bloomberg = request.args.get("bloomberg", "")
    freq = request.args.get("freq", "null")
    period = request.args.get("period", "null")

    tick = tk.getTickerByBloomberg(bloomberg)
    df = dr.getHistoryData(bloomberg, period, freq, hotPatterns)

    # fig = go.Figure(layout=go.Layout(height=600, width=1200))
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        specs=[[dict(secondary_y=True)],
                               [dict(secondary_y=False)]],
                        shared_yaxes=False,
                        vertical_spacing=0.02,
                        subplot_titles=(f"{bloomberg} {tick['name']} {period} {freq}", "RSI"),
                        row_width=[0.2, 0.8])


    fig.add_trace(
        go.Scatter(
            x=df["prettyDate"],
            y=df["BBlow"],
            line=dict(color="#195C8E"),
            legendgroup="BB",
            # legendgrouptitle_text="Bollinger Bands",
            name="Bollinger Bands"
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["prettyDate"],
            y=df["BBup"],
            line=dict(color="rgb(229,229,242)"),  # ,0.05  "rgba(229,229,242,0.01)"
            name="BB band",
            fill="tonexty",
            fillcolor="rgba(229,229,242,0.3)",  # ,0.05 rgba(229,229,242,0.3)
            legendgroup="BB",
            showlegend=False,
            # color  =  "rgba(0xD8,0xD9,0xD9,0.1)" #"#D8D9D9"
            # opacity=0.5
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["prettyDate"],
            y=df["BBup"],
            line=dict(color="#195C8E"),
            legendgroup="BB",
            name="BB up",
            showlegend=False,
            # color  =  "rgba(0xD8,0xD9,0xD9,0.1)" #"#D8D9D9"
            # opacity=0.5
        ),
        row=1, col=1,
    )
    # in the middle: D8D9D9   https://htmlcolorcodes.com/    fill= "tonexty"
    fig.add_trace(
        go.Scatter(
            x=df["prettyDate"],
            y=df["BBwma"],
            line=dict(color="#E54B11"),
            name="BBMA",
            legendgroup="BB",
            showlegend=False,
        ),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(
        x=df["prettyDate"],
        y=df["RSI"],
        name="RSI",
        legendgroup="RSI",
    ),
        row=2, col=1
    )
    fig.add_trace(go.Scatter(
        x=df["prettyDate"],
        y=df["overbought"],
        name="overbought",
        legendgroup="RSI",
        showlegend=False,
    ),
        row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["prettyDate"],
        y=df["oversold"],
        name="oversold",
        legendgroup="RSI",
        showlegend=False,
    ),
        row=2, col=1)

    barColors = []
    for i in range(len(df.index)):
        if df["Close"][i] > df["Open"][i]:
            barColors.append("rgb(0,202,115)")
        else:
            barColors.append("rgb(255,105,96)")
    maxVolume = (df['Volume'].max())
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Bar.html
    fig.add_trace(go.Bar(x=df["prettyDate"],
                         y=df['Volume'],
                         yaxis="y2",
                         name="Volume",
                         marker_color=barColors),
                  secondary_y=True,
                  row=1, col=1)
    if df["Dividends"].sum() != 0:
        divs = df[df["Dividends"] != 0]
        _min = df["Low"].min()
        _max = df["High"].max()
        _displace = (_max - _min) / 20
        fig.add_trace(go.Scatter(
            x=divs["prettyDate"],
            y=divs['High'] + _displace,
            mode="markers",
            # size="DivSize",
            # symbol= "D",
            # text = divs["divs"],
            customdata=divs["Dividends"],
            marker=dict(
                color="green",
                size=18
                #          colorscale='Viridis',
                #          line_width=10
            ),
            hovertemplate="Dividends: %{customdata}",
            showlegend=False,
        ),
            row=1, col=1)

    # activePatterns = pd.DataFrame(df,columns=["patterns","Low","High"]) .apply(filter_Patterns, axis=1)
    # activePatterns = df[df["patterns"].len() != 0]
    activeBullishPatterns = df[df.apply(lambda x: len(x["bullishPatterns"]) != 0, axis=1)].copy()

    if len(activeBullishPatterns) > 0:
        activeBullishPatterns["patternDescription"] = ""
        _min = df["Low"].min()
        _max = df["High"].max()
        _displace = (_max - _min) / 20
        for i in range(len(activeBullishPatterns)):
            idx = activeBullishPatterns.index[i]
            data = activeBullishPatterns["bullishPatterns"].at[idx]
            strPatterns = [f"{x['name']}" for x in data]  #
            activeBullishPatterns["patternDescription"].at[idx] = "<br>".join(strPatterns)
        # print(activeBullishPatterns['High'])
        # print(_displace)
        fig.add_trace(go.Scatter(
            x=activeBullishPatterns["prettyDate"],
            y=activeBullishPatterns['Low'] - _displace,
            mode="markers",
            # size="DivSize",
            # symbol= "D",
            # text = divs["divs"],
            customdata=activeBullishPatterns["patternDescription"],
            marker=dict(
                color="green",
                size=18,
                symbol="triangle-up"
                #          colorscale='Viridis',
                #          line_width=10
            ),
            hovertemplate="%{customdata}",
            showlegend=False,
            name="",
            legendgroup="patterns",
        ),
            row=1, col=1)

    activeBearishPatterns = df[df.apply(lambda x: len(x["bearishPatterns"]) != 0, axis=1)].copy()
    if len(activeBearishPatterns) > 0:
        activeBearishPatterns.assign(patternDescription="")
        _min = df["Low"].min()
        _max = df["High"].max()
        _displace = (_max - _min) / 20
        for i in range(len(activeBearishPatterns)):
            idx = activeBearishPatterns.index[i]
            data = activeBearishPatterns["bearishPatterns"].at[idx]
            strPatterns = [f"{x['name']}" for x in data]  #
            activeBearishPatterns.loc[idx, "patternDescription"] = "<br>".join(strPatterns)  # .at[idx]
        fig.add_trace(go.Scatter(
            x=activeBearishPatterns["prettyDate"],
            y=activeBearishPatterns['High'] + _displace,
            mode="markers",
            # size="DivSize",
            # symbol= "D",
            # text = divs["divs"],
            customdata=activeBearishPatterns["patternDescription"],
            marker=dict(
                color="Red",
                size=18,
                symbol="triangle-down"

                #          colorscale='Viridis',
                #          line_width=10
            ),
            hovertemplate="%{customdata}",
            showlegend=False,
            name="",
            legendgroup="patterns",
        ),
            row=1, col=1)

    fig.add_trace(
        go.Candlestick(
            opacity=1,
            x=df["prettyDate"],
            open=df["Open"],
            close=df["Close"],
            high=df["High"],
            low=df["Low"],
            name=f"{bloomberg} {period} {freq}",
            increasing={"line": {"color": 'rgb(34,84,55)',
                                 "width": 1},
                        "fillcolor": "green" },   # "rgba(0,202,115,255)"
            decreasing={"line": {"color": 'rgb(91,26,19)',
                                 "width": 1},
                        "fillcolor": "red"   # "rgb(255,105,96,255) "
                        },
            legendgroup="Candlestick"

        ),        
        row=1, col=1
    )
    fig.layout.yaxis2.showgrid = False
    # fig.update_xaxes(showgrid=False)

    # fig.update_xaxes(type='category')

    fig.update_layout(xaxis={"fixedrange": True,
                             "rangeslider": {"visible": False}},
                      yaxis={"type": "log",
                             "title": "Price"},
                      # yaxis2= {"title":"Volume",
                      #          "overlaying":'y',
                      #          "scaleanchor":'y',
                      #          "scaleratio":0.0000001
                      #          },
                      height=800,
                      width=1200,
                      template="simple_white"
                      )
    fig.update_yaxes(range=[0, maxVolume * 10],
                     secondary_y=True,
                     showticklabels=False)

    # df=df.reset_index()
    # df.columns = ['Date']+list(df.columns[1:])
    # max = (df['Open'].max())
    # min = (df['Open'].min())
    # range = max - min
    # margin = range * 0.05
    # max = max + margin
    # min = min - margin

    # fig = px.area(df, x='Date', y="Open",
    #     hover_data=("Open","Close","Volume"), 
    #     range_y=(min,max), template="seaborn" )

    # fig.show()

    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def filter_patterns(row):
    if len(row['patterns']) == 0:
        return False
    else:
        return True


@app.route("/graph", methods=['GET'])
def graph():
    bloomberg = request.args.get("bloomberg", "")
    freq = request.args.get("freq", "null")
    period = request.args.get("period", "null")

    return render_template("graph.html",
                           portfolio=pf.getPortfolioTickers(),
                           stocks=tk.getTickers(),
                           bloomberg=bloomberg,
                           _freq=freq,
                           _period=period)


@app.route("/patterns", methods=['GET'])
def stockPatterns():
    stocksPatterns = stockVal.getStockPatterns(tk.getTickers())
    return render_template("graphicalpatterns.html",
                           stocks=stocksPatterns)


@app.route("/stocks/readDaily")
def reloadDailyData():
    stockVal.reloadDailyData()
    return render_template("tickers.html",
                           tickers=tk.getTickers(),
                           stock=tk.defaultStock())


@app.route("/stockValues", methods=['GET'])
def stockValues():
    stocksData = stockVal.getStockValues(tk.getTickers())
    return render_template("stockValues.html",
                           stocks=stocksData)


@app.route("/portfolioValues", methods=['GET'])
def portfolioValues():
    stocksData = stockVal.getStockValues(pf.getPortfolioTickers())
    portfolio = pf.getPortfolio()
    rend = {}
    for s in stocksData:
        divYield = float(s["dividendYield"]) if s["dividendYield"] != "n/d" else None
        div = float(s["dividend"]) if s["dividend"] != "n/d" else None
        rend[s["reuters"]] = {"divYeld": divYield,
                              "div": div}
    for s in portfolio:
        rend[s["reuters"]]["q"] = float(s["q"])
        rend[s["reuters"]]["marketValue"] = float(s["q"]) * float(s["currPrice"])

    totalValue = 0
    totalRend = 0
    for stock in stocksData:
        r = rend[stock["reuters"]]
        totalValue += r["marketValue"]
        if r["div"] is not None and not math.isnan(r["div"]):
            r["rend"] = r["div"] * r["q"]
        else:
            if r["divYeld"] is not None and not math.isnan(r["divYeld"]):
                r["rend"] = r["divYeld"] * r["marketValue"] / 100
        if "rend" in r:
            totalRend += r["rend"]
            stock["rend"] = round(r["rend"], 2)

    totalRend = round(totalRend, 2)
    mediaRend = round(100 * totalRend / totalValue, 2)
    return render_template("stockValues.html",
                           stocks=stocksData,
                           totalRend=totalRend,
                           mediaRend=mediaRend)


@app.route("/stockTechnicalDetail/", methods=['GET'])
def stockTecnicalDetail():
    tickReq = tk.getTickerByReuters(request.args.get("reuters"))
    stocksPatterns = stockVal.getStockPatterns([tickReq])[0]
    stocksData = stockVal.getStockTechnical([tickReq])[0]
    stock = stocksData
    for k in stocksPatterns:
        stock[k] = stocksPatterns[k]

    return render_template("technicalDetail.html",
                           stock=stock)


@app.route("/stockTechnical/", methods=['GET'])
def stockTecnical():
    stocksTech = stockVal.getStockTechnical(tk.getTickers())
    return render_template("technical.html",
                           stocks=stocksTech)


@app.route("/portfolio", methods=['GET'])
def showPortfolio():
    dd = pf.getPortfolio()
    return render_template("portfolio.html",
                           total=pf.getTotals(),
                           stocks=dd,
                           currStock=pf.defaultOpenPosition())


@app.route("/portfolio/addPosition", methods=['POST'])
def addPosition():
    # addOpenPosition(code,price,q,acqdate,acqcomm
    pf.addOpenPosition(request.form.get('reuters').strip().upper(),
                       float(request.form.get('buyprice').strip().replace(",", ".")),
                       int(request.form.get('q').strip()),
                       request.form.get('buydate').strip().replace("-", "/"))

    return redirect("/portfolio")

    # portfolio = pf.getPortfolio();
    # return  render_template("portfolio.html", 
    #                         total = pf.getTotals(),
    #                         stocks=portfolio,                            
    #                         currStock=pf.defaultOpenPosition()) 


@app.route("/portfolio/closePosition", methods=['POST'])
def closePosition():
    # url = f"https://it.finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"

    # addOpenPosition(code,price,q,acqdate,acqcomm
    pf.closePosition(request.form.get('reutersSellCode').strip().upper(),
                     float(request.form.get('sellprice').strip().replace(",", ".")),
                     int(request.form.get('qSell').strip()),
                     request.form.get('selldate').strip().replace("-", "/"))

    return redirect("/portfolio")


@app.route("/analyzer")
def analyzer():
    listCode = request.args.get("listCode")
    if listCode == "" : listCode = None
    data = an.getPageData(listCode)
    return render_template("analyzer.html", data=data)


@app.route("/analyzer/evaluateTickerData")
def evaluateTickerData():
    tickerCode = request.args.get("bloomberg")
    period = request.args.get("period")
    frequency = request.args.get("frequency")
    sl = float(request.args.get("sl"))
    tp = float(request.args.get("tp"))
    data = an.evaluateTickerData(tickerCode, period, frequency, sl, tp)
    return data


@app.route("/evaluateClosedPositions")
def evaluateClosedPositions():
    pf.evaluateClosedPositions()
    return render_template("history.html",
                           total=pf.getTotals(),
                           stocks=pf.getClosedPositions())


@app.route("/openPositions")
def getOpenPositions():
    Range = request.args.get("range")
    operations = pf.getOpenPositions(Range)
    pf.evaluateCurrPrice(operations)
    for op in operations:
        op["PL"] = round((float(op["sellprice"]) - float(op["buyprice"])) * int(op["q"]), 2)
        op["percPL"] = round(100 * (float(op["sellprice"]) - float(op["buyprice"])) / op["buyprice"], 2)
    tot = pf.getOpenTotals(operations)
    # adds two columns: currPrice and diffTotal
    tot["diff"] = round(sum(s["diffTotal"] for s in operations), 2)

    return render_template("openPositions.html",
                           dataRange=Range,
                           total=tot,
                           stocks=operations)


@app.route("/history")
def getHistory():
    Range = request.args.get("range")
    operations = pf.getClosedPositions(Range)
    tot = pf.getTotals(operations)
    if Range == "difftoday":
        # adds two columns: currPrice and diffTotal
        pf.evaluateCurrPrice(operations)
        tot["diff"] = sum(s["diffTotal"] for s in operations)

    return render_template("history.html",
                           dataRange=Range,
                           total=tot,
                           stocks=operations)


@app.route("/filteredHistory", methods=['GET'])
def getFilteredHistory():
    tickReq = None
    if request.args.get("bloomberg") != "":
        tickReq = tk.getTickerByBloomberg(request.args.get("bloomberg"))
    if request.args.get("reuters") != "":
        tickReq = tk.getTickerByReuters(request.args.get("reuters"))

    hist = pf.evaluateFilteredClosedPositions(tickReq["reuters"])

    stocksPatterns = stockVal.getStockPatterns([tickReq])[0]
    stocksData = stockVal.getStockTechnical([tickReq])[0]
    stock = stocksData
    for k in stocksPatterns:
        stock[k] = stocksPatterns[k]

    return render_template("history.html",
                           stock=stock,
                           total=hist["totals"],
                           stocks=hist["rows"])


@app.route("/snapshot")
def snapshot():
    pass


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploadOpen', methods=['GET', 'POST'])
def upload_OpenPositions():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            pf.importOpenPosition(path)
            dd = pf.getPortfolio()
            return render_template("portfolio.html",
                                   total=pf.getTotals(),
                                   stocks=dd,
                                   currStock=pf.defaultOpenPosition())


@app.route('/uploadClosed', methods=['GET', 'POST'])
def upload_ClosedPositions():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            pf.importClosedPosition(path)
            pf.getPortfolio()
            return render_template("index.html", patterns=patternList)


if __name__ == "__main__":
    # dictConfig({
    #     'version': 1,
    #     'formatters': {'default': {
    #         'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    #     }},
    #     'handlers': {'wsgi': {
    #         'class': 'logging.StreamHandler',
    #         'stream': 'ext://flask.logging.wsgi_errors_stream',
    #         'formatter': 'default'
    #     }},
    #     'root': {
    #         'level': 'INFO',
    #         'handlers': ['wsgi']
    #     }
    # })
    app.run(debug=True)
