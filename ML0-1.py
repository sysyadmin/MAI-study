import matplotlib.pyplot as plt, numpy as np
from pandas import read_csv
from statistics import mode
file = read_csv('MSFT.csv')
title = ['Open','High','Low','Adj Close','Volume']

x = file[title].values
y = file['Close'].values
n = len(title)

for i in range(0, n):
    plt.xlabel(title[i])
    plt.ylabel('Close')
    plt.plot(file[title[i]].values, y, 'ro')
    plt.show()

#Stock price at the close of the exchange.


#Mean value is a characteristic that describes the average value in a data set. In the case of the average value of the “middle” of the dataset,
   # the arithmetic average of its values will be. The average value reflects a typical indicator in the data set.
   # If we randomly select one of the indicators, then we will most likely get a value close to the average.
    
sum = sum(file['Close'])
print(sum)
num = len(file['Close'])
print(num)
avg = sum/num
print('Средняя цена',avg)

#The median is a measure of central tendency.
    #It is needed to determine typical values in a data set, but it does not require calculations.
           
middle = len(file['Close'])/2+0.5
list_sorted=sorted(file['Close'])
mediane = list_sorted[int(middle)]
print('Median',mediane)

#Min&Max of a function are the largest and smallest value of the function
max = max(file['Close'])
print('Max',max)
min = min(file['Close'])
print('Min',min)

#Mode is defined as the value that is most commonly found in a data set.

mode = mode(file['Close'])
print('Mode',mode)

#Range of a set of data is the difference between the largest and smallest values.
   # It is the first characteristic that answers the question “How much does my data vary?”.
r = max - min
print('Range', r)

#Standard deviation is also a measure of data variation. shows how much the data differ from the arithmetic mean.
sd = np.std(file['Close'])
print('Standard deviation',sd)

#Dispersion is simply the square of the standard deviation. It reflects the measure of dispersion.
d = sd * sd
print('Dispersion',d)

