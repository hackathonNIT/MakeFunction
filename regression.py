import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
import os

# (0)CSVのロード
os.chdir("/content/drive/MyDrive/hackason")
data = np.loadtxt("kaiki.csv",delimiter=",", skiprows=1, usecols=(1,2,3),  dtype='float')

def WriteFunc1(X, *params):
    Y = np.zeros_like(X)
    for i,param in enumerate(params):
        print(str(param)+"*"+"sin("+str(i*i)+"x)")

def func1(X, *params):
    Y = np.zeros_like(X)
    for i,param in enumerate(params):
        #print(param+"*"+"sin("+0.1*i*i+"x)")
        Y = Y + np.array(param*np.sin( i*i*X ))
    return Y

def func(X, *params):
    Y = np.zeros_like(X)
    for i, param in enumerate(params):
        Y = Y + np.array(param * X ** i)
    return Y
def getRegressionLineND():

    D = 100
    # (1)CSVデータの1列目(とある株価)をy(目的変数)として扱う
    #y1 = data[:,1]/10.0
    # (2)x(説明変数)のデータの数だけ用意
    #x = data[:,2]/1.0

    #カージオイド
    t = np.arange(0, 10, 0.1)
    a = 1.0
    x = a * (1 + np.cos(t)) * np.cos(t)
    y1 = a * (1 + np.cos(t)) * np.sin(t)
    #



    # (3-1)グラフ作成
    plt.figure("kaji")


    plt.plot(x, y1, "o", label="kajiP")
    #for d in range(D):
    #  plt.plot(x, np.poly1d(np.polyfit(x, y1, d))(x), label="RegressionLine")
    plt.plot(x, np.poly1d(np.polyfit(x, y1, D))(x), label="RegressionLine")

    popt1, pcov1=curve_fit(func1,x, y1, p0=[1]*len(x))
    #popt2, pcov2=curve_fit(func,x, y1, p0=[1]*len(x))


    plt.plot(x,func1(x,*popt1), label="sin")
    #plt.plot(x,func(x,*popt2), label="next")

    print(popt1)
    print("要素数"+str(len(popt1)))

    WriteFunc1(x,*popt1)
    # (4-1)ラベル表示
    plt.xlabel("num")
    plt.ylabel("f")
    plt.legend()
#    print(np.poly1d(np.polyfit(x, y1, D)))
    # (4-2)グリッド表示
    plt.grid()

    # (4-3)グラフ表示
    plt.show()

if __name__=='__main__':
    getRegressionLineND()
