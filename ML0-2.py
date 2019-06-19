import re #for regular expressions
import string
import matplotlib.pyplot as plt


frequency = {}#чота я устала на англ писать. объявляем частотную переменную
document_text = open('t.txt', 'r')#подключаем файлиииик с всем содержимым
text_string = document_text.read().lower()#преобразовываем файл в строку в нижнем регистре
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)


for word in match_pattern:  #идем по полученному массиву слов
    count = frequency.get(word,0) #ищем вхождения слова 
    frequency[word] = count + 1
     
frequency_list = frequency.keys()
list = [] 
for words in frequency_list:
    list.append([words, frequency[words]])
    #print (words, frequency[words])


list.sort(key=lambda x: x[1])
#print(list[-5:])

s=[]
n=[]
for i in reversed(list[-5:]):
    s.append(i[1])
    n.append(i[0])

    ## s и n - это слово и его частота.
#print(s)
#print(n)

x=range(len(s))

ax = plt.gca()
ax.bar(x, s, align='edge')
ax.set_xticks(x)
ax.set_xticklabels(n)
plt.show()

