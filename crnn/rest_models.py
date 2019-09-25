import json

class CrnnModel:
    result = str
    def __init__(self,predict,tureValue,accuracy,crnnTime):
        self.predict = predict
        self.tureValue = tureValue
        self.accuracy = accuracy
        self.crnnTime = crnnTime

    def to_json(self):
        return json.dumps(self,default=lambda o:o.__dict__,sort_keys=True,indent=4)
    def json_str(self):
        jstr = {'predict':self.predict,'tureValue':self.tureValue,'accuracy':self.accuracy,'crnnTime':self.crnnTime}
        return(json.dumps(jstr))