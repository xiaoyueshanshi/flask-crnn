import os
import sys
import time

class CrnnModel:
    def __init__(self,res_str,accuracy,timeuse):
        self.res_str = res_str
        self.accuracy = accuracy
        self.timeuse = timeuse
