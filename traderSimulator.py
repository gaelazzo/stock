import backtrader as bt
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers



import datetime
import numpy as np
import sys

import pandas as pd


class Cross_medie(bt.Strategy):
    # Tuple of tuples containing any variable settings required by the strategy.
    params = (('QuickAvg', 20), ("SlowAvg", 50))

    def __init__(self):
        #print(f"Init class, params: {self.params.QuickAvg} {self.params.SlowAvg}")
        self.quickAvg = btind.SMA(period=self.params.QuickAvg)
        self.slowAvg = btind.SMA(period=self.params.SlowAvg)
        #print(self.quickAvg)
        #print(self.slowAvg)
        self.crossover  = btind.CrossOver(self.quickAvg, self.slowAvg)

        self.dataclose = self.datas[0].close

    def next(self):
        if self.crossover  > 0:
            #print(f"BUYING self.position.size is {self.position.size}  size is {self.getposition().size}")
            self.buy()

        elif self.crossover  < 0:
            #print(f"SELLING self.position.size is {self.position.size}  size is {self.getposition().size}")
            self.sell()


    def stop(self):
        self.value = round(self.broker.getvalue(), 2)
        self.print(f"quickAvg  {self.params.QuickAvg} slowAvg  {self.params.SlowAvg} valore finale:{self.value}")


    def print(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        #print(f'{dt.isoformat()}, {txt}')



    def notify_order(self, order):
                if order.status in [order.Submitted, order.Accepted]:
                    # ordine acquisto/vendita accettato",
                    return

                # Verifica se ordine completato
                if order.status in [order.Completed]:
                    if order.isbuy():
                  #Stampiamo dettaglio di quantità, prezzo e commissioni
                        self.print(
                            'ACQ ESEGUITO, QTY: %.2f, PREZZO: %.2f, COSTO: %.2f, COMM %.2f' %
                            (order.executed.size,
                             order.executed.price,
                             order.executed.value,
                             order.executed.comm
                             ))

                        self.buyprice = order.executed.price
                        self.buycomm = order.executed.comm
                    else:  # Vendita
                        self.print('VEND ESEGUITA, QTY: %.2f, PREZZO: %.2f, COSTO: %.2f, COMM %.2f' %
                                 (order.executed.size,
                                  order.executed.price,
                                  order.executed.value,
                                  order.executed.comm
                                  )),

                    self.bar_executed = len(self)

                if order.status in [order.Canceled, order.Margin, order.Rejected]:
                    self.print('Ordine Cancellato')

                self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.print("Profitto operazione, LORDO %.2f, NETTO %.2f" % (trade.pnl, trade.pnlcomm))

#timeframe Ticks, Seconds, Minutes, Days, Weeks, Months and Years

#https://www.backtrader.com/blog/posts/2016-07-23-sizers-smart-staking/sizers-smart-staking/
class LongOnly(bt.Sizer):
    params = (('stake', 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return self.p.stake

        # Sell situation
        position = self.strategy.getposition(data)
        if not position.size:
            return 0  # do not sell if nothing is open

        return self.p.stake

class FixedReverser(bt.Sizer):
    params = (('stake', 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)
        size = self.p.stake * (1 + (position.size != 0))
        return size

def applyStrategy(data=None,
                  strategy=Cross_medie
                  ):


    cerebro = bt.Cerebro()

    if data is None:
        data = bt.feeds.YahooFinanceCSVData(dataname="datasets/TMO_history_1d.csv",
                                        fromdate=datetime.datetime(2011, 1, 1),
                                        todate=datetime.datetime(2021, 11, 20),
                                        reverse=False)

    cerebro.adddata(data)
    cerebro.broker.setcash(100000)
    cerebro.addsizer(LongOnly, stake=1000)  # or  FixedReverser
    cerebro.broker.setcommission(commission=0.0002)
    startValue = cerebro.broker.getvalue()
    #print(f"Valore iniziale portafoglio: {cerebro.broker.getvalue()}")
    cerebro.run()
    stopValue = cerebro.broker.getvalue()
    #print(f"Valore finale portafoglio: {cerebro.broker.getvalue()}")
    return stopValue-startValue


def applyStrategyMinMax(fileName,
                          strategy=Cross_medie,
                          slowRange=np.arange(20, 50, 2),  #[20,50],
                          quickRange=np.arange(5, 10, 1),
                        ):
    cerebro= bt.Cerebro(optreturn=False)
    strats = cerebro.optstrategy(Cross_medie,
                                 SlowAvg=slowRange,
                                 QuickAvg=quickRange
                                 )
    data = bt.feeds.YahooFinanceCSVData(dataname=fileName,
                                        fromdate=datetime.datetime(2011, 1, 1),
                                        todate=datetime.datetime(2021, 11, 20),
                                        reverse=False)
    cerebro.adddata(data)
    startCash = 100000
    cerebro.broker.setcash(startCash)
    cerebro.addsizer(LongOnly, stake=1000)  # or  FixedReverser
    cerebro.broker.setcommission(commission=0.0002)
    opt_runs = cerebro.run(maxcpus=1)
    resList = []
    for run in opt_runs:
        for strategy in run:
            val = round(strategy.value, 2)
            PnL = round(val - startCash)
            m_quick = strategy.params.QuickAvg
            m_slow = strategy.params.SlowAvg
            resList.append([m_slow, m_quick, PnL])
    resData = pd.DataFrame(resList)
    resData.columns = ["slow Avg", "quick Avg", "P/L"]
    return resData.sort_values(["P/L"], ascending=False)

