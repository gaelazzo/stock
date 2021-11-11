# -*- coding: utf-8 -*-
import csv

from datetime  import datetime
 
class CommissionCalculator:
    cfg = None
    def __init__(self):
        with open('data/commissions.csv') as csv_file:
            if (CommissionCalculator.cfg is None):          
                CommissionCalculator.cfg = []
                reader = csv.DictReader(csv_file)
                for row in reader:
                    row["start"]= datetime.strptime(row["start"], "%d/%m/%Y")
                    row["min"]= float(row["min"])
                    row["max"]= float(row["max"])
                    row["num"]= float(row["num"]) if row["num"]!="" else 0
                    if (row["rate"] is not None and row["rate"]!=""):
                        row["rate"]= float(row["rate"])                    
                    CommissionCalculator.cfg.append(row)
            self.contracts = {}
    
    def reset(self):
        for d in self.contracts:
            self.contracts[d]=0
        
    def incOperation(self,date):
        dd=date
        if isinstance(dd,str):
            dd = datetime.strptime(dd, "%d/%m/%Y")
        lastValid=None;
        for d in CommissionCalculator.cfg:
            if (d["start"]<dd):
                lastValid=d["start"]
        if (lastValid is None):
            return
        
        if (lastValid not in self.contracts):
            self.contracts[lastValid]=0;
        
        self.contracts[lastValid] = self.contracts[lastValid]+1
    
    def getCommission(self, amount,date):
        lastValidDate=None;
        for d in CommissionCalculator.cfg:
            if (d["start"]<date):
                lastValidDate=d["start"]
        if (lastValidDate is None):
            return 99999
        
        contract  = [x for x in CommissionCalculator.cfg if x["start"] == lastValidDate]
        nOperations = self.contracts[lastValidDate]
        rowContract = [c for c in contract if c["num"]>=nOperations or c["num"]==0][0]
        if (rowContract["rate"]==0):
            return rowContract["min"]
        comm  = round(rowContract["rate"]*amount,2)
        if (comm<rowContract["min"]):
            comm=rowContract["min"]
        if (comm>rowContract["max"]):
            comm=rowContract["max"]
        return comm
    
    