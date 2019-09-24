import os
import sys

class Config:
    DEBUG = True
    URL = r'127.0.0.1'
    PORT = 8080
    THREADED = True
    #thread 和 process (多线程和多进程不可同时开启，两者只可开启一个)
    #使用：app.run(threaded=True)/app.run(processes=True)
    #PROCESSES = true




