import base64
from PIL import Image
#from predict import get_predict_report
import sys
def test():
    with open(r'E:\A_1_logos\002050.jpg','rb') as image:
        base_str = base64.b64encode(image.read())
        print(base_str)

        #get_predict_report(base_str,'FRT')

if __name__=='__main__':
    test()