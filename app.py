from flask import Flask

from app_config import Config as cfg

app = Flask(__name__)

@app.route('/')
def hello_world() -> str:
    return 'Hello World!'

@app.route('/crnn/<str>')
def crnn(str) -> str:
    print(str)
    return 'hello crnn-->%s'%str





if __name__ == '__main__':
    #default:localhost:5000
    #app.run()
    #指定url,port
    #debug=ture:debug模式，false:关闭debug -->开启debug改变代码，会自动部署
    #指定后如何需要运行端口需要在启动参数中加上-p用于指定：flask run -p 8080
    app.run(host=cfg.URL,port=cfg.PORT,debug=cfg.DEBUG)
