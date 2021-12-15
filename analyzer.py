import csv
import json
import tickers as tk
import dataretriever as dr
import patterns
import indicators
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, \
    f1_score

# random search linear regression model on the auto insurance dataset
from scipy.stats import loguniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import random
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from pprint import pprint

from sklearn.preprocessing import OneHotEncoder

analyzeList = []


def reload():
    analyzeList.clear()
    with open('data/analyzeList.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            analyzeList.append(row)
    line_count = len(analyzeList)
    print(f'analyzeList.csv: Processed {line_count} lines.')


def getPageData(listCode="default"):
    if (listCode == None): listCode = "default"

    allGroups = list(set([x["listCode"] for x in analyzeList]))

    groupCodes = list(set([x["bloomberg"] for x in analyzeList if x["listCode"] == listCode]))

    tkdata = map(
        lambda code: tk.getTickerByBloomberg(code),
        groupCodes
    )

    return {"list": allGroups, "tickers": list(tkdata), "currentList": listCode}


def categorizeRsi(v, overbought, oversold):
    third = (overbought - oversold) / 3
    if (v >= overbought): return "RSIoverbought"
    if (v <= oversold): return "RSIoversold"
    if (v >= overbought - third): return "RSIbought"
    if (v <= oversold + third): return "RSIsold"
    return "RSImiddle"


def categorizeBB(v, up, low):
    third = (up - low) / 3
    if (v >= up): return "BBoverbought"
    if (v <= low): return "BBoversold"
    if (v >= up - third): return "BBbought"
    if (v <= low + third): return "BBsold"
    return "BBmiddle"


def evaluateTickerData(bloomberg, period, interval,
                       stopLossPerc=1,
                       takeProfitPerc=2,
                       patternsToCalc=patterns.hotPatterns,
                       endDate=None,
                       outDataFrame=False):
    if endDate is None:
        endDate = datetime.today().date() + timedelta(days=1) #this date is excluded from data range
    #print(f"endDate is {endDate}")
    df = dr.getHistoryData(bloomberg, period, interval,
                           patternsToCalc=patternsToCalc,
                           evaluateRSI=True,
                           evaluateBollinger=True,
                           endDate=endDate)
    hoursPerDay = 8.5  # those should be evaluated basing on bloomberg code
    bufferLen = 25  # those should be evaluated basing on bloomberg code

    start, stop = dr.evaluateDateRange(bloomberg, period, interval, endDate, bufferLen, hoursPerDay)
    #print(f"start is {start} and stop is {stop}")

    df["profitLoss"] = None
    df["profitLossInfo"] = None
    df["profitLossBB"] = None
    df["profitLossInfoBB"] = None
    df["profitLossSimple"] = None
    df["profitLossSimpleInfo"] = None
    # df["profitLoss2"]=None
    # df["profitLossInfo2"]=None
    # df["stopLoss"]=None
    # df["takeProfit"]=None
    # df["subIndexTaken"]=None
    df["rifValue"] = None

    for i in range(len(df) - 1):
        idx = df.index[i]
        subDF = df[i + 1:]  # from row i on

        rifValue = subDF["Open"].at[subDF.index[0]]  ##suppose to buy at market price next period
        stopLoss = rifValue * ((100 - stopLossPerc) / 100)
        takeProfit = rifValue * ((100 + takeProfitPerc) / 100)

        result, additionalInfo = fixedLimits(subDF, bloomberg, subDF.index[0], stop, interval, stopLoss, takeProfit)
        df["profitLoss"].at[idx] = result
        df["profitLossInfo"].at[idx] = additionalInfo

        result, additionalInfo = bollingerLimits(subDF, bloomberg, subDF.index[0], stop, interval)
        df["profitLossBB"].at[idx] = result
        df["profitLossInfoBB"].at[idx] = additionalInfo

        result, additionalInfo = closeNextDay(subDF, rifValue)
        df["profitLossSimple"].at[idx] = result
        df["profitLossSimpleInfo"].at[idx] = additionalInfo

        # rifValue = subDF["Open"].at[subDF.index[0]] ##suppose to buy at market price next period
        # stopLoss = subDF["BBlow"]
        # takeProfit = subDF["BBhigh"]

        # result, additionalInfo = strategyLimits(subDF,bloomberg,subDF.index[0],stop,interval, stopLoss,takeProfit)
        # # df["stopLoss"].at[idx]=round(stopLoss,2) # df["subIndexTaken"].at[idx]=subDF.index[0]
        df["rifValue"].at[idx]=rifValue
        # # df["takeProfit"].at[idx]=round(takeProfit,2)
        # df["profitLoss2"].at[idx]=result
        # df["profitLossInfo2"].at[idx]=additionalInfo

    df["prevClose"] = None
    df["diffPerc"] = None
    prevValues = df.shift(periods=1)
    # set previous close for every row
    for index, r in df.iterrows():
        if prevValues["Close"].at[index] != prevValues["Close"].at[index]:
            continue  # skips first row
        df["prevClose"].at[index] = prevValues["Close"].at[index]

    # evaluate close difference in perc. high perc, Low perc, openPerc
    df["diffPerc"] = (
            ((200 * ((df["Close"] - df["prevClose"]) / df["prevClose"])).astype(np.double).round(0)) / 2).round(1)
    df["HighPerc"] = (
            ((200 * ((df["High"] - df["prevClose"]) / df["prevClose"])).astype(np.double).round(0)) / 2).round(1)
    df["LowPerc"] = (((200 * ((df["Low"] - df["prevClose"]) / df["prevClose"])).astype(np.double).round(0)) / 2).round(
        1)
    df["OpenPerc"] = (
            ((200 * ((df["Open"] - df["prevClose"]) / df["prevClose"])).astype(np.double).round(0)) / 2).round(1)

    df["VolumePerc"] = (df["Volume"] / df["averageVolume"]).round(decimals=1)

    # evaluate data

    # foreach bullish and bearish pattern, add a separate column to df
    for p in patternsToCalc:
        # print("adding column "+p)
        df[p] = 0

    df["anyPattern"] = 0
    # foreach row set a value for the pattern
    for index, r in df.iterrows():
        # print(index,r)
        # print("Bullish")
        # print(r["bullishPatterns"])
        for c in r["bullishPatterns"]:
            # print(c["code"],c["name"])
            df[c["code"]].at[index] = 1  # "bullish"
            df["anyPattern"].at[index] = 1
            # print("Bearish")
        # print(r["bearishPatterns"])
        for c in r["bearishPatterns"]:
            # print(c["code"],c["name"])
            df[c["code"]].at[index] = -1  # "bearish"
            df["anyPattern"].at[index] = 1  # "bullish"

    # remove unused pattern columns
    df.drop(['bullishPatterns', 'bearishPatterns', "errorsPatterns"], axis=1, inplace=True)

    # remove some RSI columns

    # categorize RSI, RSI prec
    for col in ['RSIoverbought', 'RSIoversold', 'RSIbought', 'RSIsold', "RSImiddle"]:
        df[col] = 0
        df[col + "prev"] = 0

    for index, r in df.iterrows():
        field = categorizeRsi(df["RSI"].at[index],
                              df["overbought"].at[index],
                              df["oversold"].at[index]
                              )
        df[field].at[index] = 1

    prevValues = df.shift(periods=1)
    for index, r in df.iterrows():
        if prevValues["RSI"].at[index] != prevValues["RSI"].at[index]:
            continue  # skips first row
        field = categorizeRsi(prevValues["RSI"].at[index],
                              prevValues["overbought"].at[index],
                              prevValues["oversold"].at[index]
                              )
        df[field + "prev"].at[index] = 1

    # df.drop(['overbought', 'oversold',"RSI"], axis=1,inplace=True)

    # categorize basing on Bollinger bands
    # df["BBwma"] = df["Close"].rolling(window=length).mean()
    # df["BBdev"] = df["Close"].rolling(window=length).std()
    # df["BBup"] = df["BBwma"]+ mul* df["BBdev"]
    # df["BBlow"] = df["BBwma"]- mul* df["BBdev"]
    for col in ['BBoverbought', 'BBoversold', 'BBbought', 'BBsold', "BBmiddle"]:
        df[col] = 0
        df[col + "prev"] = 0

    for index, r in df.iterrows():
        field = categorizeBB(df["Close"].at[index],
                             df["BBup"].at[index],
                             df["BBlow"].at[index]
                             )
        df[field].at[index] = 1

    prevValues = df.shift(periods=1)
    for index, r in prevValues.iterrows():
        field = categorizeBB(prevValues["Close"].at[index],
                             prevValues["BBup"].at[index],
                             prevValues["BBlow"].at[index]
                             )
        df[field + "prev"].at[index] = 1

        # df["profitLoss"] = df["profitLoss"].astype('category')

    df.to_csv("data/tempresult.csv")

    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    df.drop(empty_cols, axis=1, inplace=True)

    if outDataFrame:
        return df

    return {"data": json.loads(df.to_json(orient="split"))}


def cleanDataFrame(df):
    d = df.copy(True)
    for col in [
        'RSIoverbought', 'RSIoversold', 'RSIbought', 'RSIsold', "RSImiddle",
        'BBoverbought', 'BBoversold', 'BBbought', 'BBsold', "BBmiddle"]:
        for c in [col, col + "prev"]:
            d[c] = d[c].astype('category')

    for c in ['diffPerc', 'HighPerc', 'LowPerc', 'VolumePerc']:
        d[c] = d[c] * 5
        d[c] = d[c].round(0)
        d[c] = d[c] / 5

    def cvProfit(v):
        if v == 1:
            return "profit"
        if v == -1:
            return "loss"
        return "-"

    # d["profitLoss"] = d["profitLoss"].apply(cvProfit)
    d["profitLoss"] = d["profitLoss"].astype('category')
    d["profitLossBB"] = d["profitLossBB"].astype('category')

    d.reset_index(drop=True, inplace=True)
    return d


def createSampleTestDataSet(interval="1d", patternsToCalc=patterns.hotPatterns):
    """
    Creates two csv files:data/testData.csv and data/sampleData.csv
     with a random sampling of last 5 years of data sampled day by day of all tickers

    Returns
    -------
    None.

    """
    lTickers = tk.getTickers()
    l1 = random.sample(lTickers, int(len(lTickers) / 3))
    l2 = [ll for ll in lTickers if not ll in l1]
    d1 = create_dataset(l1, interval=interval, patternsToCalc=patternsToCalc)
    d2 = create_dataset(l2, interval=interval, patternsToCalc=patternsToCalc)
    d1.to_csv(f"data/testData{interval}.csv")
    d2.to_csv(f"data/sampleData{interval}.csv")


def create_dataset(tickerList, period="10y", interval="1d", patternsToCalc=patterns.patternList):
    """
     creates a Dataframe invoking evaluateTickerData with  the specified params, 
         skipping etfs and btp, removing columns not useful for machine learning
     
     Parameters
     ----------
     tickerList : list of tickers to be read
         every element of the list must have at least etf,kind and bloomberg columns
     period : string, optional
         Period to consider. The default is "max".
     interval : string, optional
         Interval (frequency) to consider. The default is "1d".
     
     Returns
     -------
     d : Dataframe         
     
     """

    d = None
    endDate = datetime.today().date() - timedelta(days=1)
    for t in tickerList:
        if t["etf"] != "N":
            continue
        if t["kind"] != "E":
            continue
        dd = evaluateTickerData(t["bloomberg"], period, interval,
                                endDate=endDate,
                                patternsToCalc=patternsToCalc,
                                outDataFrame=True)
        dd.dropna(axis='index', inplace=True)

        if dd is None:
            print(f"got None reading {t['bloomberg']}")
            continue

        if d is None:
            d = dd
        else:
            d = pd.concat([d, dd])
    return d


def tryClassifyAll():
    best = None
    bestPrecision = 0

    # for _interval in ["1d","1wk","1h"]:
    #     createSampleTestDataSet(interval=_interval, patternsToCalc=patterns.patternList)

    for _interval in ["1d", "1wk", "1h"]:
        sample = pd.read_csv(f"data/sampleData{_interval}.csv")
        test = pd.read_csv(f"data/testData{_interval}.csv")
        for pt in patterns.patternList:
            for field in ["profitLossBB", "profitLossSimple", "profitLoss"]:  #"profitLoss",
                res = tryClassify(fieldProfitLoss=field,
                                  interval=_interval,
                                  patterns=[pt],
                                  plotGraph=False,
                                  sample=sample.copy(),
                                  test=test.copy()
                                  )
                if res is None:
                    continue
                if (res["results"]["precision"] > bestPrecision):
                    best = res
                    bestPrecision = res["results"]["precision"]
                    print(f"precision is {bestPrecision} with params {field} {_interval} pattern:{pt}")
                    best["set"] = f"precision is {bestPrecision} with params {field} {_interval} pattern:{pt}"
                    js = json.dumps(best)
                    text_file = open("best.txt", "wt")
                    n = text_file.write(js)
                    text_file.close()


def tryClassify(fieldProfitLoss, interval="1d",
                patterns=["CDLENGULFING"],
                plotGraph=True,
                test=None,
                sample=None
                ):
    if sample is None:
        sample = pd.read_csv(f"data/sampleData{interval}.csv")
    if test is None:
        test = pd.read_csv(f"data/testData{interval}.csv")
    print(f"tryClassify {fieldProfitLoss} {interval} {patterns}")
    return classify(sample,
                    test,
                    profitLossField=fieldProfitLoss,
                    patternsToAnalyze=patterns,
                    plotGraph=plotGraph)


def removeExtraColumns(df):
    columnsToRemove = ['prevClose', 'Open', 'High', "Low", "Close", "Adj Close", "Volume",
                       "prettyDate", "bloomberg",
                       "prevClose",
                       "OpenPerc", "diffPerc", "LowPerc", "HighPerc", "VolumePerc",
                       "RSIbought","RSImiddle","RSIoverbought","RSIoversold","RSIsold",
                       "RSIboughtprev", "RSImiddleprev","RSIoverboughtprev","RSIoversoldprev","RSIsoldprev",
                       #"BBbought","BBoverbought","BBsold", "BBoversold","BBmiddle",
                       "BBboughtprev","BBoverboughtprev", "BBoversoldprev","BBsoldprev","BBmiddleprev",
                       "anyPattern"
                       ]
    if "Date" in df.columns:
        columnsToRemove.append("Date")
    if "Datetime" in df.columns:
        columnsToRemove.append("Datetime")
    df = df.drop(columnsToRemove, axis=1)
    #print("Columns removed:")
    #print(columnsToRemove)
    return df


# https://towardsdatascience.com/my-random-forest-classifier-cheat-sheet-in-python-fedb84f8cf4f
def classify(df, dTest,
             profitLossField="profitLoss",
             patternsToAnalyze=["CDLENGULFING"],
             plotGraph=False):
    d = cleanDataFrame(df)
    d[profitLossField] = d[profitLossField].astype('int')
    d.drop(d[d[profitLossField] == 0].index, inplace=True)
    if len(d) == 0:
        return None


    def checkSomePattern(r):
        for pField in patternsToAnalyze:
            if r[pField] != 0:
                return True
        return False

    indexesToDrop = [index for index, r in d.iterrows() if not checkSomePattern(r)]
    d.drop(indexesToDrop, inplace=True)

    d[profitLossField] = d[profitLossField].astype('category')

    dTest = cleanDataFrame(dTest)
    dTest[profitLossField] = dTest[profitLossField].astype('int')
    dTest.drop(dTest[dTest[profitLossField] == 0].index, inplace=True)

    indexesToDropTest = [index for index, r in dTest.iterrows() if not checkSomePattern(r)]
    dTest.drop(indexesToDropTest, inplace=True)

    nSampleGain = len([x for x in d[profitLossField] if x > 0])  # d["profitLoss"].unique().tolist()
    nSampleLoss = len([x for x in d[profitLossField] if x < 0])  # d["profitLoss"].unique().tolist()
    if nSampleGain < 20 or nSampleLoss < 20:
        print(f"sample set not significant {nSampleGain}-{nSampleLoss}")
        return None

    nTestGain = len([x for x in dTest[profitLossField] if x > 0])  # d["profitLoss"].unique().tolist()
    nTestLoss = len([x for x in dTest[profitLossField] if x < 0])  # d["profitLoss"].unique().tolist()
    if nTestGain < 10 or nTestLoss < 10:
        print(f"test set not significant {nTestGain}-{nTestLoss}")
        return None

    print(f"sample:  {nSampleGain}-{nSampleLoss} test: {nTestGain}-{nTestLoss}")

    toRemove = [c for c in d.columns if (c.startswith("CDL") and c not in patternsToAnalyze)
                or (c.startswith("profitLoss") and c != profitLossField)
                or c in ['Dividends', 'Stock Splits', "averageVolume",
                         'overbought', 'oversold', "RSI",
                         'BBwma', 'BBdev', "BBup", "BBlow"]]
    d.drop(columns=toRemove, inplace=True)
    dTest.drop(columns=toRemove, inplace=True)

    dTest[profitLossField] = dTest[profitLossField].astype('category')

    X = removeExtraColumns(d)
    X = X.drop([profitLossField], axis=1)

    # def cvProfit(v):
    #     if v==1: return "profit"
    #     if v==-1: return "loss"
    #     return "-"

    XTest = removeExtraColumns(dTest)
    XTest = XTest.drop([profitLossField], axis=1)

    y = d[profitLossField]

    # y = y.apply(cvProfit)
    # y = y.astype('int')
    y = y.astype('category')

    X.to_csv("data/bigdata_X.csv")
    y.to_csv("data/bigdata_y.csv")
    seed = 50  # so that the result is reproducible
    search = True

    while search:
        search = False
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=seed
                                                        )
        nTrainGain = len([x for x in y_train if x > 0])  # d["profitLoss"].unique().tolist()
        nTrainLoss = len([x for x in y_train if x < 0])  # d["profitLoss"].unique().tolist()
        nTestGain = len([x for x in y_test if x > 0])  # d["profitLoss"].unique().tolist()
        nTestLoss = len([x for x in y_test if x < 0])  # d["profitLoss"].unique().tolist()
        print(f"Extracting sample Train:{nTrainGain} over {nTrainLoss}  Test: {nTestGain} over {nTestLoss}  ")
        if nTrainGain < 6 or nTrainLoss < 6:
            search = True
        if nTestGain < 3 or nTestLoss < 3:
            search = True



    # features_to_encode = X_train.columns[X_train.dtypes==object].tolist()

    # The remainder = 'passthrough' allows the constructor to ignore those variables that are not included in features_to_encode.
    # col_trans = make_column_transformer(
    #                     (OneHotEncoder(),features_to_encode),
    #                     remainder = "passthrough"
    #                     )

    # param_distributions = {'n_estimators': randint(1, 5),
    #                        'max_depth': randint(2, 5)}

    # search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
    #                          n_iter=5,
    #                          scoring="accuracy",
    #                          param_distributions=param_distributions,
    #                          random_state=0,
    #                          return_train_score=True)

    rf_classifier = RandomForestClassifier(
        # min_samples_leaf=50,
        #n_estimators=10,
        # bootstrap=True,
        # oob_score=True,
        n_jobs=-1,
        random_state=seed,
        max_features='auto')
    # pipe = make_pipeline(col_trans, rf_classifier)
    pipe = rf_classifier
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # Accuracy (fraction of correctly classified samples)
    accuracy_score(y_test, y_pred)
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 3) * 100} %")

    # Make probability predictions
    train_probs = pipe.predict_proba(X_train)[:, 1]
    probs = pipe.predict_proba(X_test)[:, 1]
    train_predictions = pipe.predict(X_train)

    # def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    #   #creating a set of all the unique classes using the actual class list
    #   unique_class = set(actual_class)
    #   roc_auc_dict = {}
    #   for per_class in unique_class:
    #     #creating a list of all the classes except the current class 
    #     other_class = [x for x in unique_class if x != per_class]

    #     #marking the current class as 1 and all other classes as 0
    #     new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    #     new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #     #using the sklearn metrics method to calculate the roc_auc_score
    #     roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    #     roc_auc_dict[per_class] = roc_auc

    #   return roc_auc_dict

    # print(f'Train ROC AUC Score: {roc_auc_score_multiclass(y_train, train_probs)}') #
    # print(f'Test ROC AUC  Score: {roc_auc_score_multiclass(y_test, probs)}') #

    # print(f"{y_train}")
    if plotGraph:
        print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_probs, labels=np.unique(y_pred))}')  #
        print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs, labels=np.unique(y_pred))}')  #

    # assuming your already have a list of actual_class and predicted_class from the logistic regression classifier
    # lr_roc_auc_multiclass = roc_auc_score_multiclass(y_test, y_train)
    # print(lr_roc_auc_multiclass)

    def evaluate_model(y_pred, probs, train_predictions, train_probs, plot):
        baseline = {}
        baseline['recall'] = recall_score(y_test,
                                          [1 for _ in range(len(y_test))],
                                          labels=np.unique(y_pred))
        baseline['precision'] = precision_score(y_test,
                                                [1 for _ in range(len(y_test))],
                                                labels=np.unique(y_pred))
        baseline['roc'] = 0.5
        results = {'recall': recall_score(y_test, y_pred),
                   'precision': precision_score(y_test, y_pred),
                   'roc': roc_auc_score(y_test, probs, labels=np.unique(y_pred))}
        train_results = {'recall': recall_score(y_train, train_predictions),
                         'precision': precision_score(y_train, train_predictions),
                         'roc': roc_auc_score(y_train, train_probs, labels=np.unique(y_pred))}
        #if plot:
        for metric in ['precision']: #['recall', 'precision', 'roc']
            print(f"""{metric.capitalize()}  Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}""")
        # Calculate false positive rates and true positive rates
        base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
        model_fpr, model_tpr, _ = roc_curve(y_test, probs)
        if plot:
            plt.figure(figsize=(8, 6))
            plt.rcParams['font.size'] = 16
            # Plot both curves
            plt.plot(base_fpr, base_tpr, 'b', label='baseline')
            plt.plot(model_fpr, model_tpr, 'r', label='model')
            plt.legend()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.show()
        return {"baseline": baseline,
                "results": results,
                "train_results": train_results
                }
    print("evaluating Training model")
    evaluation = evaluate_model(y_pred, probs, train_predictions, train_probs, plotGraph)

    def plot_confusion_matrix(cm, classes, normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Greens):  # can change color
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size=24)
        plt.colorbar(aspect=4)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size=14)
        plt.yticks(tick_marks, classes, size=14)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        # Label the plot
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     fontsize=20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            plt.grid(None)
            plt.tight_layout()
            plt.ylabel('True label', size=18)
            plt.xlabel('Predicted label', size=18)

    cm = confusion_matrix(y_test, y_pred)
    if plotGraph:
        plot_confusion_matrix(cm, classes=['0 - Stay', '1 - Exit'],
                              title='Exit_status Confusion Matrix for Sample')

        # Feature Importance
        print(rf_classifier.feature_importances_)

    # col_trans.fit_transform(X_train)
    # print(col_trans.fit_transform(X_train)[0,:])

    # def encode_and_bind(original_dataframe, features_to_encode):
    #     dummies = pd.get_dummies(original_dataframe[features_to_encode])
    #     res = pd.concat([dummies, original_dataframe], axis=1)
    #     res = res.drop(features_to_encode, axis=1)
    #     return(res)

    # X_train_encoded = encode_and_bind(X_train, features_to_encode)

    # ☺feature_importances = list(zip(X_train_encoded, rf_classifier.feature_importances_))
    feature_importances = list(zip(X_train, rf_classifier.feature_importances_))

    # Then sort the feature importances by most important first
    feature_importances_ranked = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Print out the feature and importances
    if plotGraph:
     [print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked]

    feature_names_25 = [i[0] for i in feature_importances_ranked[:25]]
    y_ticks = np.arange(0, len(feature_names_25))
    x_axis = [i[1] for i in feature_importances_ranked[:25]]
    if plotGraph:
        plt.figure(figsize=(10, 14))
        plt.barh(feature_names_25, x_axis)  # horizontal barplot
        plt.title('Random Forest Feature Importance (Top 25)',
                  fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})
        plt.xlabel('Features', fontdict={'fontsize': 16})
        plt.show()
        print('Parameters currently in use:\n')
        pprint(rf_classifier.get_params())
    #https://numpy.org/doc/stable/reference/generated/numpy.linspace.html Return evenly spaced numbers over a specified interval.
    n_estimators = [int(x) for x in np.linspace(start=3, #The starting value of the sequence. array_like
                                                stop=20, #The end value of the sequence, unless endpoint is set to False. array_like
                                                num=10  #Number of samples to generate. Default is 50. Must be non-negative.
                                                )]

    max_features = ['auto', 'log2']  # Number of features to consider at every split

    max_depth = [3,4,5,10,15] #[int(x) for x in np.linspace(2, 6, num=2)]  # Maximum number of levels in tree

    max_depth.append(None)

    min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node

    min_samples_leaf = [2, 3, 5]  # Minimum number of samples required at each leaf node

    bootstrap = [True, False]  # Method of selecting samples for training each tree

    random_grid = { # in general the more trees the less likely the algorithm is to overfit. So try increasing this.
                    'n_estimators': n_estimators, #The number of trees in the forest.

                   #This determines how many features each tree is randomly assigned. The smaller,
                   # the less likely to overfit, but too small will start to introduce under fitting.
                   'max_features': max_features,

                   #This will reduce the complexity of the learned models, lowering over fitting risk.
                   # Try starting small, say 5-10, and increasing you get the best result.
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,

                    #Try setting this to values greater than one.This has a similar effect to the max_depth parameter,
                    # it means the branch will stop splitting once the leaves have that number of samples each
                   'min_samples_leaf': min_samples_leaf,
                   'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
                   'bootstrap': bootstrap}

    # Tune the hyperparameters with RandomSearchCV

    # Create base model to tune (2 alternatives)
    # rf = RandomForestClassifier(oob_score=True)    
    # Create random search model and fit the data
    # rf_random = RandomizedSearchCV(
    #                     estimator = rf,
    #                     param_distributions = random_grid,
    #                     n_iter = 100, cv = 3,
    #                     verbose=2, random_state=seed, 
    #                     scoring='roc_auc')

    # rf_random.fit(X_train_encoded, y_train)   
    # rf_random.best_params_

    rf = RandomForestClassifier(n_jobs=-1,
                                n_estimators=3,  #The number of trees in the forest.
                                #max_depth = The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                #min_samples_split = The minimum number of samples required to split an internal node
                                criterion="entropy",
                                #oob_score=True #Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.
                                )  # oob_score=True,
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=200, #Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
        cv=2,   #cross-validation generator or an iterable, default=None
        #               integer, to specify the number of folds in a (Stratified)KFold,
        # n_iter=50, cv=3,
        verbose=1, random_state=seed,
        scoring='precision')  # roc_auc
    pipe_random = rf_random  # make_pipeline(col_trans, rf_random)
    pipe_random.fit(X_train, y_train)
    print(rf_random.best_params_)

    # Evaluate the best model
    best_model = rf_random.best_estimator_
    pipe_best_model = best_model  # make_pipeline(col_trans, best_model)
    pipe_best_model.fit(X_train, y_train)
    y_pred_best_model = pipe_best_model.predict(X_test)

    n_nodes = []
    max_depths = []
    for ind_tree in best_model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)
    if plotGraph:
        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')

    # Use the best model after tuning

    train_rf_predictions = pipe_best_model.predict(X_train)
    train_rf_probs = pipe_best_model.predict_proba(X_train)[:, 1]
    rf_probs = pipe_best_model.predict_proba(X_test)[:, 1]
    # Plot ROC curve and check scores
    print("evaluating model on Test set")
    best_model = evaluate_model(y_pred_best_model, rf_probs, train_rf_predictions, train_rf_probs, plotGraph)

    # Plot Confusion matrix
    if plotGraph:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred_best_model),
                              classes=['-1 Loss', '1 - Profit'],
                              title='Exit_status Confusion Matrix for Test')

    # test_withoutDate = XTest.copy().drop('Date', axis = 1)
    # test_withoutID = test_withoutID.fillna('na')
    final_y = pipe_best_model.predict(XTest)
    # pipe model only takes in dataframe without ID column.
    final_report = XTest
    final_report[profitLossField] = final_y
    final_report = final_report.loc[:, [profitLossField]]
    # Replace 1-0 with Yes-No to make it interpretable
    final_report = final_report.replace(1, 'Profit')
    final_report = final_report.replace(-1, 'Loss')
    final_report[profitLossField].value_counts()

    # final_report.to_csv('submissions.csv', index=False)
    return best_model

    # search.fit(X_train, y_train)
    # print (search.score(X_test, y_test))
    # return (search,X_test,y_test)


def classify2(df):
    # https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
    d = cleanDataFrame(df)

    y = d["profitLoss"].astype('category')

    X = removeExtraColumns(d)
    X = X.drop(["profitLoss"], axis=1)

    X.to_csv("data/bigdata_X.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = LogisticRegression()

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define search space
    space = dict()
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
    space['C'] = loguniform(1e-5, 100)

    search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)

    result = search.fit(X_train, y_train)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    # {'C': 0.0059892483184825155, 'penalty': 'l2', 'solver': 'newton-cg'}
    print(search.score(X_test, y_test))
    return (search, X_test, y_test)


N_FOLDS = 5
MAX_EVALS = 5


def cleanDataFrameNoCategory(df):
    d = df.copy(True)
    for col in ['RSIoverbought', 'RSIoversold', 'RSIbought', 'RSIsold', "RSImiddle",
                'BBoverbought', 'BBoversold', 'BBbought', 'BBsold', "BBmiddle"]:
        for c in [col, col + "prev"]:
            d[c] = d[c].astype('int')

    for c in ['diffPerc', 'HighPerc', 'LowPerc', 'VolumePerc']:
        d[c] = d[c] * 5
        d[c] = d[c].round(0)
        d[c] = d[c] / 5

    d["profitLoss"] = d["profitLoss"].astype('int')
    d["profitLossBB"] = d["profitLossBB"].astype('int')
    d["profitLossSimple"] = d["profitLossSimple"].astype('int')

    d.reset_index(drop=True, inplace=True)

    return d


# Grid and Random Search
# https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search
def classifyGridSearch(df):
    d = cleanDataFrameNoCategory(df)
    labels = d["profitLoss"].astype('int')
    # labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1, ))

    features = d.drop(
        ['Open', 'High', "Low", "prevClose", "Close", "Adj Close", "Volume", "prettyDate", "bloomberg", "profitLoss",
         "OpenPerc", "diffPerc", "LowPerc", "HighPerc", "VolumePerc"], axis=1)

    features.to_csv("data/bigdata_NoCat.csv")

    # Split into training and testing data
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=6000,
                                                                                random_state=50)

    print("Training features shape: ", train_features.shape)
    print("Testing features shape: ", test_features.shape)

    # Create a training and testing dataset
    train_set = lgb.Dataset(data=train_features, label=train_labels)
    test_set = lgb.Dataset(data=test_features, label=test_labels)

    # Get default hyperparameters
    model = lgb.LGBMClassifier()
    default_params = model.get_params()

    # Remove the number of estimators because we set this to 10000 in the cv call
    del default_params['n_estimators']

    # Cross validation with early stopping
    cv_results = lgb.cv(default_params, train_set, num_boost_round=10000, early_stopping_rounds=100,
                        metrics='auc', nfold=N_FOLDS, seed=42)

    print('The maximum validation ROC AUC was: {:.5f} with a standard deviation of {:.5f}.'.format(
        cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
    print('The optimal number of boosting rounds (estimators) was {}.'.format(len(cv_results['auc-mean'])))

    # Optimal number of esimators found in cv
    model.n_estimators = len(cv_results['auc-mean'])

    def objective(hyperparameters, iteration):
        """Objective function for grid and random search. Returns
           the cross validation score from a set of hyperparameters."""

        # Number of estimators will be found using early stopping
        if 'n_estimators' in hyperparameters.keys():
            del hyperparameters['n_estimators']

        myTrainSet = lgb.Dataset(data=train_features, label=train_labels)

        # Perform n_folds cross validation
        cv_results = lgb.cv(hyperparameters, myTrainSet, num_boost_round=10000, nfold=N_FOLDS,
                            early_stopping_rounds=100, metrics='auc', seed=42)

        # results to retun
        score = cv_results['auc-mean'][-1]
        estimators = len(cv_results['auc-mean'])
        hyperparameters['n_estimators'] = estimators

        return [score, hyperparameters, iteration]

    score, params, iteration = objective(default_params, 1)
    print('The cross-validation ROC AUC was {:.5f}.'.format(score))

    # Create a default model
    model = lgb.LGBMModel()
    model.get_params()

    # Hyperparameter grid
    param_grid = {
        'boosting_type': ['gbdt', 'goss', 'dart'],
        'num_leaves': list(range(20, 150)),
        'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
        'subsample_for_bin': list(range(20000, 300000, 20000)),
        'min_child_samples': list(range(20, 500, 5)),
        'reg_alpha': list(np.linspace(0, 1)),
        'reg_lambda': list(np.linspace(0, 1)),
        'colsample_bytree': list(np.linspace(0.6, 1, 10)),
        'subsample': list(np.linspace(0.5, 1, 100)),
        'is_unbalance': [True, False]
    }
    random.seed(50)
    # Randomly sample a boosting type
    boosting_type = random.sample(param_grid['boosting_type'], 1)[0]

    # Set subsample depending on boosting type
    subsample = 1.0 if boosting_type == 'goss' else random.sample(param_grid['subsample'], 1)[0]

    print('Boosting type: ', boosting_type)
    print('Subsample ratio: ', subsample)

    # %matplotlib inline

    plt.hist(param_grid['learning_rate'], bins=20, color='r', edgecolor='k')
    plt.xlabel('Learning Rate', size=14)
    plt.ylabel('Count', size=14)
    plt.title('Learning Rate Distribution', size=18)

    a = 0
    b = 0

    # Check number of values in each category
    for x in param_grid['learning_rate']:
        # Check values
        if x >= 0.005 and x < 0.05:
            a += 1
        elif x >= 0.05 and x < 0.5:
            b += 1

    print('There are {} values between 0.005 and 0.05'.format(a))
    print('There are {} values between 0.05 and 0.5'.format(b))

    # number of leaves domain
    plt.hist(param_grid['num_leaves'], color='m', edgecolor='k')
    plt.xlabel('Learning Number of Leaves', size=14)
    plt.ylabel('Count', size=14)
    plt.title('Number of Leaves Distribution', size=18)

    # Dataframes for random and grid search
    random_results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                                  index=list(range(MAX_EVALS)))

    grid_results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                                index=list(range(MAX_EVALS)))

    def grid_search(param_grid, max_evals=MAX_EVALS):
        """Grid search algorithm (with limit on max evals)"""

        # Dataframe to store results
        results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                               index=list(range(MAX_EVALS)))

        # https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
        keys, values = zip(*param_grid.items())

        i = 0

        # Iterate through every possible combination of hyperparameters
        for v in itertools.product(*values):

            # Create a hyperparameter dictionary
            hyperparameters = dict(zip(keys, v))

            # Set the subsample ratio accounting for boosting type
            hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters[
                'subsample']

            # Evalute the hyperparameters
            eval_results = objective(hyperparameters, i)

            results.loc[i, :] = eval_results

            i += 1

            # Normally would not limit iterations
            if i > MAX_EVALS:
                break

        # Sort with best score on top
        results.sort_values('score', ascending=False, inplace=True)
        results.reset_index(inplace=True)

        return results

    grid_results = grid_search(param_grid)

    print('The best validation score was {:.5f}'.format(grid_results.loc[0, 'score']))
    print('\nThe best hyperparameters were:')

    pprint.pprint(grid_results.loc[0, 'params'])

    # Get the best parameters
    grid_search_params = grid_results.loc[0, 'params']

    # Create, train, test model
    model = lgb.LGBMClassifier(**grid_search_params, random_state=42)
    model.fit(train_features, train_labels)

    preds = model.predict_proba(test_features)[:, 1]
    print('The best model from grid search scores {:.5f} ROC AUC on the test set.'.format(
        roc_auc_score(test_labels, preds)))


def categorizePerceptron(df):
    d = cleanDataFrameNoCategory(df)
    y = d["profitLoss"].astype('int')

    X = d.drop(
        ['Open', 'High', "Low", "prevClose", "Close", "Adj Close", "Volume", "prettyDate", "bloomberg", "profitLoss"],
        axis=1)

    X.to_csv("data/bigdata_Perceptron.csv")

    # Split into training and testing data

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(eta0=0.1, random_state=0, penalty="l2", tol=0.1)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test)
    diff = [c for c in y_test != y_pred if c]
    print(f"Misclassified samples {len(diff)} over {len(y_test)}")
    print(f"Accuracy is {round(1 - len(diff) / len(y_test), 2)}")

    return (ppn, X_test, y_test)


def fixedLimits(df, bloomberg, start, stop, frequency, stopLoss, takeProfit):
    """
    Starting from first row of dataframe, searches first row where prices exits from range stoplimit-profitlimit
        and returns -1 if it reaches stoplimit, or +1 if it reaches profitLimit
        Returns 0 if it's not possible to detect the answer
    Sets
    Parameters
    ----------
    df : Dataframe
    bloomberg : string
        ticker.
    start : date
        interval date start
    stop : date
        interval date stop.
    frequency : string
        frequency considered, one of 1mo 1wk 1d 
    stopLoss : float
        stop loss value
    takeProfit : TYPE
        take profit value

    Returns
    -------
    int
        1 for profit taken, -1 for stop loss taken,0 for undetermined
    string
        detail on what happened

    """
    # print(f"start={start}, stop={stop}")
    # print(f"real start = {df.index[0]}")
    # print(f"real stop = {df.index[-1]}")

    if len(df) == 0:
        return 0, "No rows left"
    first = True

    for idx, r in df.iterrows():
        if first:
            if r["Open"] <= stopLoss:
                return 0, f"Opening ({r['Open']})is less than stopLoss {stopLoss}"
            if r["Open"] >= takeProfit:
                return 0, f"Opening ({r['Open']})is greater than takeProfit {takeProfit}"
            first = False

        profit = (r["High"] >= takeProfit)
        loss = (r["Low"] <= stopLoss)
        if profit and not loss:
            return 1, f"High on {idx} cause {round(r['High'], 4)}>{round(takeProfit, 2)}, sl={round(stopLoss, 2)}"
        if loss and not profit:
            return -1, f"Low on {idx} cause {round(r['Low'], 4)}<{round(stopLoss, 2)}, tp={round(takeProfit, 2)}"
        if not loss and (r["Close"] >= takeProfit):
            return 1, f"Close on {idx} cause {r['Close']} > {round(takeProfit, 2)}, sl={round(stopLoss, 2)}"
        # if (not profit and (r["Close"]<=stopLimit)): return (1,f"Close on {idx} is {r['Close']} > {round(profitlimit,2)}, sl={round(stoplimit,2)}")

        if loss and profit:
            return 0, f"Loss and profit on {idx}  out of  ({stopLoss}) and  {takeProfit}"
            # mapFrequency = {"1mo": "1wk", "1wk": "1d", "1d": None, "1h":None}
            # newFreq = mapFrequency[frequency]
            # if newFreq == None: return (None,
            #                     f"profit and loss on {idx}: {r['High']}and {r['Low']}, "
            #                     f"sl={round(stopLoss, 2)}, "
            #                     f"tp={round(takeProfit, 2)}")
            # df = dr.getDataRange(bloomberg, start, stop, newFreq)
            # return fixedLimits(df, bloomberg, start, stop, newFreq, stopLoss, takeProfit)

    return 0, "No action"  # no take profit and not stop loss

def closeNextDay(df, rifValue):
    firstRow = df.iloc[0]
    if firstRow["Close"] > rifValue:
        return 1, f"Close {firstRow['Close']} > {rifValue}"
    return -1, f"Close {firstRow['Close']} < {rifValue}"

def bollingerLimits(df, bloomberg, start, stop, frequency):
    """
    Starting from first row of dataframe, searches first row where close price reaches a limit of bollinger band
        and returns -1 if it reaches lower limit BBlow, or +1 if it reaches high limit BBup
        Returns 0 if it's not possible to evaluate the answer
    Sets
    Parameters
    ----------
    df : Dataframe
    bloomberg : string
        ticker.
    start : date
        interval date start
    stop : date
        interval date stop.
    frequency : string
        frequency considered, one of 1mo 1wk 1d


    Returns
    -------
    int
        1 for profit taken, -1 for stop loss taken,0 for undetermined
    string
        detail on what happened

    """

    first = True
    for idx, r in df.iterrows():
        if first:
            first = False
            if r["Open"] <= r['BBlow']:
                return 0, f"Opening ({r['Open']})is less than BBlow {r['BBlow']}"
            if r["Open"] >= r['BBup']:
                return 0, f"Opening ({r['Open']})is greater than BBup {r['BBup']}"

        bbUp = r["BBup"]
        bbLow = r["BBlow"]
        profit = (r["High"] > bbUp)
        loss = (r["Low"] < bbLow)
        if profit and not loss:
            return 1, f"High on {idx} cause {round(r['High'], 4)}>{round(bbUp, 2)}, sl={round(bbLow, 2)}"
        if loss and not profit:
            return -1, f"Low on {idx} cause {round(r['Low'], 4)}<{round(bbLow, 2)}, tp={round(bbUp, 2)}"
        if not loss and (r["Close"] >= bbUp):
            return 1, f"Close on {idx} cause {r['Close']} > {round(bbUp, 2)}, sl={round(bbLow, 2)}"

        if loss and profit:
            return 0, f"Loss and profit on {idx}  out of  ({r['BBlow']}) and  {r['BBup']}"

    return 0, "No action"  # no take profit and not stop loss


reload()

print("module analyzer loaded")