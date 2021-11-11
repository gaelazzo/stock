import talib

patternList = {
    'CDL2CROWS' : 'Two Crows',
    'CDL3BLACKCROWS' : 'Three Black Crows',
    'CDL3INSIDE' : 'Three Inside Up/Down',
    'CDL3LINESTRIKE' : 'Three-Line Strike',
    'CDL3OUTSIDE' : 'Three Outside Up/Down',
    'CDL3STARSINSOUTH' : 'Three Stars In The South',
    'CDL3WHITESOLDIERS' : 'Three Advancing White Soldiers',
    'CDLABANDONEDBABY' : 'Abandoned Baby',
    'CDLADVANCEBLOCK' : 'Advance Block',
    'CDLBELTHOLD' : 'Belt-hold',
    'CDLBREAKAWAY' : 'Breakaway',
    'CDLCLOSINGMARUBOZU' : 'Closing Marubozu',
    'CDLCONCEALBABYSWALL' : 'Concealing Baby Swallow',
    'CDLCOUNTERATTACK' : 'Counterattack',
    'CDLDARKCLOUDCOVER' : 'Dark Cloud Cover',
    'CDLDOJI' : 'Doji',
    'CDLDOJISTAR' : 'Doji Star',
    'CDLDRAGONFLYDOJI' : 'Dragonfly Doji',
    'CDLENGULFING' : 'Engulfing Pattern',
    'CDLEVENINGDOJISTAR' : 'Evening Doji Star',
    'CDLEVENINGSTAR' : 'Evening Star',
    'CDLGAPSIDESIDEWHITE' : 'Up/Down-gap side-by-side white lines',
    'CDLGRAVESTONEDOJI' : 'Gravestone Doji',
    'CDLHAMMER' : 'Hammer',
    'CDLHANGINGMAN' : 'Hanging Man',
    'CDLHARAMI' : 'Harami Pattern',
    'CDLHARAMICROSS' : 'Harami Cross Pattern',
    'CDLHIGHWAVE' : 'High-Wave Candle',
    'CDLHIKKAKE' : 'Hikkake Pattern',
    'CDLHIKKAKEMOD' : 'Modified Hikkake Pattern',
    'CDLHOMINGPIGEON' : 'Homing Pigeon',
    'CDLIDENTICAL3CROWS' : 'Identical Three Crows',
    'CDLINNECK' : 'In-Neck Pattern',
    'CDLINVERTEDHAMMER' : 'Inverted Hammer',
    'CDLKICKING' : 'Kicking',
    'CDLKICKINGBYLENGTH' : 'Kicking - bull/bear determined by the longer marubozu',
    'CDLLADDERBOTTOM' : 'Ladder Bottom',
    'CDLLONGLEGGEDDOJI' : 'Long Legged Doji',
    'CDLLONGLINE' : 'Long Line Candle',
    'CDLMARUBOZU' : 'Marubozu',
    'CDLMATCHINGLOW' : 'Matching Low',
    'CDLMATHOLD' : 'Mat Hold',
    'CDLMORNINGDOJISTAR' : 'Morning Doji Star',
    'CDLMORNINGSTAR' : 'Morning Star',
    'CDLONNECK' : 'On-Neck Pattern',
    'CDLPIERCING' : 'Piercing Pattern',
    'CDLRICKSHAWMAN' : 'Rickshaw Man',
    'CDLRISEFALL3METHODS' : 'Rising/Falling Three Methods',
    'CDLSEPARATINGLINES' : 'Separating Lines',
    'CDLSHOOTINGSTAR' : 'Shooting Star',
    'CDLSHORTLINE' : 'Short Line Candle',
    'CDLSPINNINGTOP' : 'Spinning Top',
    'CDLSTALLEDPATTERN' : 'Stalled Pattern',
    'CDLSTICKSANDWICH' : 'Stick Sandwich',
    'CDLTAKURI' : 'Takuri (Dragonfly Doji with very long lower shadow)',
    'CDLTASUKIGAP' : 'Tasuki Gap',
    'CDLTHRUSTING' : 'Thrusting Pattern',
    'CDLTRISTAR' : 'Tristar Pattern',
    'CDLUNIQUE3RIVER' : 'Unique 3 River',
    'CDLUPSIDEGAP2CROWS' : 'Upside Gap Two Crows',
    'CDLXSIDEGAP3METHODS' : 'Upside/Downside Gap Three Methods'
}

hotPatterns = {
     'CDLDOJISTAR' : 'Doji Star',
     'CDL3INSIDE' : 'Three Inside Up/Down',
     'CDLDOJISTAR' : 'Doji Star',
    'CDLDRAGONFLYDOJI' : 'Dragonfly Doji',
    'CDLENGULFING' : 'Engulfing Pattern',
    'CDLEVENINGDOJISTAR' : 'Evening Doji Star',
    'CDLEVENINGSTAR' : 'Evening Star',
      'CDLGRAVESTONEDOJI' : 'Gravestone Doji',
    'CDLHAMMER' : 'Hammer',
    'CDLHANGINGMAN' : 'Hanging Man',
    'CDLHARAMI' : 'Harami Pattern',
    'CDLHARAMICROSS' : 'Harami Cross Pattern',
     'CDLSHOOTINGSTAR' : 'Shooting Star',
     'CDLMORNINGDOJISTAR' : 'Morning Doji Star',
    'CDLMORNINGSTAR' : 'Morning Star',
    }


def applyPatterns(Open,High,Low,Close,patterns=patternList):
    #returns a dictionary with:
    # bullish = dictionary(code=pattern code, name = patternName)
    # bearish = dictionary(code=pattern code, name = patternName)
    # errors = dictionary(code=pattern code, name = patternName)
    if (len(Open)==0): return dict(bullish=[], bearish=[],errors=[])
    evaluatedBullish=[[] for i in range(len(Open))]
    evaluatedBearish=[[] for i in range(len(Open))]
    evaluatedErrors=[[] for i in range(len(Open))]
    for pattern in patterns:
            patternFunction  = getattr(talib,pattern)
            try:
                result = patternFunction(Open,High,Low,Close)
                
                for i in range(len(Open)):
                    if result[i]>0: evaluatedBullish[i].append(dict(code=pattern, name=patternList[pattern]))
                    if result[i]<0: evaluatedBearish[i].append(dict(code=pattern, name=patternList[pattern]))
            except:
                evaluatedErrors[0].append(dict(code=pattern, name=patterns[pattern]))
                break;
        
    return dict(bullish=evaluatedBullish, bearish=evaluatedBearish,errors=evaluatedErrors)
