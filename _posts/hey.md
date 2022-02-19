---
layout: post
title: Data Analysis template
subtitle: My works so far
---

# Data Analysis Template

# NUMPY

→ Stored at one continuous place in memory (faster than list)

**`conda install numpy`** 

In the localhost, use `pip install numpy`

```python
#Array Basics
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
np.arange(0,20,5) #0~20 5 단위로 띄어쓰기

#listing
np.zeros((5,5))
np.ones((3,3))
np.linspace(0,10,3) #0~10 사이를 3등분해서 쪼개놓는 방법
np.linspace(0,10,50)
np.eye(4) # 4 X 4 matrix organize

#random values
np.random.rand(0,10)
np.random.rand(0,10,3) #0~10 사이 3개
np.random.randint() # ㄹㅇ 그냥 랜덤 integer
np.random.randn(10) #random no 10개
arr.reshape(5,5)

#Broadcasting
#Setting a value with index range (Broadcasting)
arr[0:5]=100

#complex operation
arr = np.arange(1,11)
bool_arr = arr>4
arr[bool_arr]

#array operations (+,-,* 는 가능, /는 불가능)
np.sqrt(), np.max(), np.sin(), np.exp(), np.log()

mat.sum(axis = 1) #return row additions
mat.sum(axis = 0) #return column additions

```

```python
import numpy as np
arr = np.array([1,2,3,4,5]) #((1,2,3)) as tuple
arr1 = np.array([[1,2],[3,4]]) # ndim = # of [] 
arr2 = np.array([1,2,3,4], ndmin = 5)
print(arr1.ndim) # = 2

print(arr1[0,1]) # same as array, it is 1st row 2nd element (2)
print(arr1[1,-1]) # = 4
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3[0,1,2]) # = 6

#slicing is the same as normal python - exclude last one
arr4 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr4[1,1:4]) #2nd dimension, from 2th~4th [7,8,9]
print(arr4[0:2,1:4]) # [[2 3 4],[7 8 9]]

arr5 = np.array([1, 2, 3, 4], dtype='S') # modify to string
arr6 = np.array([1.1, 2.1, 3.1])
arr6_i = arr.astype('int') #as integer, [1,2,3]

#copies owns the data, and views does not own the data
arr7 = np.array([1,2,3,4,5])
c = arr7.copy() #copy them
v = arr7.view() #change in original also change 'v'
arr[0] = 'false' #c = [1,2,3,4,5]     v = [0,2,3,4,5]
v[0] = 'true' # arr = ['true',2,3,4,5] v 바꿔도 arr 역으로 바뀜
print(c.base) #NONE
print(v.base) # arr = [1,2,3,4,5]

arr4.shape # (2,5) -> 1-D : 2 , 2-D : 5 each
arr2.shape # 5 dimension, # for each elements (1,1,1,1,4)

**arr8** = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr1 = arr8.reshape(4, 3) #4*3 = 12
newarr2 = arr8.reshape(2,3,2) #2*3*2 = 12
print(newarr1) #All the reshapes are '
print(newarr1.shape) #1-D : 4 elements , 2-D : 3 for each list
newarr1.base #these are all VIEWS, so NO BASE
newarr3 = arr8.reshape(2,2,-1) #for -1, it is unknown
**flatten - into 1D, rot90, flip, fliplr, flipud**

#Iteration - we have to iterate the arrays in each dimension
arr = np.array([[1, 2, 3], [4, 5, 6]])
for i in arr:
	print(i) # [1,2,3] [4,5,6]

for i in arr:
	for j in i:
		print(j) # 1,2,3,4,5,6 (all by element)

for x in np.nditer(arr): #dimension에 상관없이 모두 element화 해줌
  print(x)
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x) #return as the string (modify data type)
for idx, x in np.ndenumerate(arr):
  print(idx, x) #enumerate (0,1) 2 (0,2) 3 (1,0) 4... -> depend on dimension

```

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-20_at_6.00.51_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-20_at_6.00.51_PM.png)

Some tricks to modify the arrays

```python
#Must be a matching dimension
arr1 = np.array([[1, 2, 3],[4,5,6]])
arr2 = np.array([[4, 5, 6],[7,8,9]])
arr = np.concatenate((arr1, arr2), axis = 1) #by row, so [[1,2,3,4,5,6],[4,5,6,7,8,9]]

#JOIN
a = np.stack((arr1,arr2)) #단순히 뒤에 거 위에 앞을 쌓음
b = np.hstack((arr1,arr2)) #row - wise stacking
c = np.vstack((arr1,arr2)) # column - wise stacking
d = np.dstack((arr1,arr2)) #by 'depth' - 세로로 array 개수만큼 묶어서

#SPLIT - each dimension **must be the same**
arr3 = np.array([1, 2, 3, 4, 5, 6])
print(np.array_split(arr3,3)) #split in 3 [1,2] , [3,4] , [5,6]
print(np.array_split(arr3, 4)) # modify # of content - 1,2/3,4/5/6
arr4 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
print(np.array_split(arr4,3)) #split values are also 2D
# [array([[1, 2],[3, 4]]), array([[5, 6],[7, 8]]), array([[ 9, 10],[11, 12]])]
print(np.array_split(arr4,3,axis = 1) #by row, so arr1([1],[3],...) arr2([2],[4]..
print(np.hsplit(arr, 3)) #same as 'by row'

#SEARCH
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(조건문) # return position in array form (e.g. arr == 4, arr%2 == 0)
y = np.searchsorted(arr, 3) #binary search 후 순서상 맞는 위치를 return
y = np.searchsorted(arr, 3, side = 'right') #start from right

arr5 = np.array([1, 3, 5, 7])
x = np.searchsorted(arr5, [2, 4, 6]) #2,4,6 should be in index 1,3,5

#SORTING
arr6 = np.array([3,0,1,5]) #dimension 커지면 각 dimension이 sort됨
print(np.sort(arr6) #both numeric and alphabetic
arr7 = np.array([True, False, True])
print(np.sort(arr7)) #False > True

#FILTER - Use the Boolean index list
arr8 = np.array([41, 42, 43, 44])
x = [True, False, True, False] #normally use if-else to append T/F value to bool[]
newarr = arr8[x] #[41,43]

filter_arr = arr > 42 #direct filtering using **condition**
newarr = arr8[filter_arr] #array[condition filter]

```

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-20_at_11.35.10_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-20_at_11.35.10_PM.png)

→Example of if-else for `boolean filter`

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_10.31.51_AM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_10.31.51_AM.png)

→ Pseudo random & True Random

- Pseudo: Random numbers generated through a generation algorithm
- True: Random numbers from outside source

```python
#a
[[[1 2 3]
  [4 5 6]]
 [[4 5 6]
  [7 8 9]]]
#b
[[1 2 3 4 5 6]
 [4 5 6 7 8 9]]
#c
[[1 2 3]
 [4 5 6]
 [4 5 6]
 [7 8 9]]
#d
[[[1 4]
  [2 5]
  [3 6]]

 [[4 7]
  [5 8]
  [6 9]]]
```

```python
#RANDOM
from numpy import random
x = random.randint(100) #random int under 100
y = random.rand() #random float

x = random.randint(100, size=(3)) #array with size 3
y = random.randint(100, size=(3, 5)) #2D with size 3 rows with 5 elements
z = random.rand(3) # array of float size 3
w = random.rand(3,5) #2-D random with size 3 X 5

x = random.choice([3, 5, 7, 9]) #return any of array
x = random.choice([3, 5, 7, 9], size=(3, 5)) #2d random array 3X5 with array val
#distribution with 3,5,7,9 with P, array 3X5
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))

arr = np.array([1, 2, 3, 4, 5])
random.shuffle(arr) #randomly mix the index
print(arr)
print(random.permutation(arr)) #X affect the original, but just the permuted
print(arr) #still [1,2,3,4,5]

#NORMAL / GAUSSIAN DISTRIBUTION - Draw random samples from a bell curve
from numpy import random
x = random.normal(size=(2,3))
y = random.normal(loc=1, scale=2, size=(2, 3)) #loc - mean, scale - sd, size
#seaborn - normal dist. with mean 10, sd 1 and 1000 samples
sns.distplot(random.normal(loc = 10, scale = 1, size=1000), hist=False)

#Binomial Dist - discrete, but continuous with ㅈㄴ많은 데이터 (discrete)
x = random.binomial(n=10, p=0.5, size=10) #n = trial p - prob, size
sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False) #kde = kernel density estimate

#MULTI-nomial -> extension of binomial (n = # of outcome, pval = prob, size)
x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

#Poission Dist - event can happen in a specified time (p of eating twice/week?)
#continuous 
x = random.poisson(lam=2, size=10) #given mean & size

#UNIFORM Dist - every event has equal chances of occuring
x = random.uniform(size = (2,3))
sns.distplot(random.uniform(low = 0.0, high = 10.0,size=1000), hist=True)

#LOGISTIC Dist - growth logistic regression, neural networks
x = random.logistic(loc=1, scale=2, size=(2, 3))

#Exponential Dist - describing time till next event
x = random.exponential(scale=2, size=(2, 3)) #scale - inverse of lam, size

#Chi-squared Dist - basis to verify the hypothesis
#d.o.f = maximum number of logically independent values df = (r-1)(c-1)
x = random.chisquare(df=2, size=(2, 3))

#Rayleigh Dist - signal processing
x = random.rayleigh(scale=2, size=(2, 3)) #scale = how flat the dist is

#Pareto Dist - 80-20 distribution (20% factors cause 80% outcome)
x = random.pareto(a=2, size=(2, 3)) #a = shape parameter - 커질수록 퍼짐

#Zipf Dist - nth common term is 1/n times of the most common term
x = random.zipf(a=2, size=(2, 3)) # a = distribution parameter, 
```

Normal Dist

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.21.30_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.21.30_PM.png)

Uniform Dist

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.22.38_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.22.38_PM.png)

Logistics Dist

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.23.32_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.23.32_PM.png)

Binomial Dist

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.21.40_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.21.40_PM.png)

Poisson Dist

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.23.01_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.23.01_PM.png)

Exponential Dist

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.23.49_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_6.23.49_PM.png)

# SEABORN

**Distribution Plot**

- distplot
- jointplot
- pairplot
- rugplot
- kdeplot

**Categorical Plot**

- factorplot
- boxplot
- violinplot
- stripplot
- swarmplot
- barplot
- countplot

```python
pip install seaborn
import seaborn as sns
tips = sns.load_dataset('tips')

**#Distribution Plot**
sns.distplot(tips['total bill'])
sns.distplot(tips['total_bill'],kde=False,bins=30)

sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
#Different kind = scatter, hex, reg, resid, kde, 
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg') #regression line
sns.pairplot(tips) #tips 내의 모든 조합을 다 실험해 보는 것
sns.pairplot(tips,hue='sex',palette='coolwarm') #hue 기준으로 
sns.rugplot(tips['total_bill']) #Dashes in the corresponding bin

**#Categorical Plot**
import numpy as np
****sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std) #sex별로 total bill std?
sns.countplot(x='sex',data=tips) #count만 함 - 총합 몇인지
sns.boxplot(x="day", y="total_bill", data=tips,palette='rainbow')
sns.boxplot(data=tips,palette='rainbow',orient='h')
sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="coolwarm")
sns.stripplot(x="day", y="total_bill", data=tips,jitter=True) #jitter - distribution
sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1',split=True)
sns.swarmplot(x="day", y="total_bill",hue='sex',data=tips, palette="Set1", split=True)

**#Matrix**
flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')

#Heatmap
sns.heatmap(tips.corr()) 
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)

#pivot table (customized settings)
pvflights = flights.pivot_table(values='passengers',index='month',columns='year')
sns.heatmap(pvflights)

#Clustermap (based on similarity)
sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)

**#Grid**
sns.PairGrid(iris)
g = sns.PairGrid(iris)
g.map(plt.scatter)

# Map to upper,lower, and diagonal
g = sns.PairGrid(iris)
g.map_diag(plt.hist) #diagonal 은 histogram으로
g.map_upper(plt.scatter) # 위쪽 부분들은 map_upper
g.map_lower(sns.kdeplot) #아래쪽 부분들은 map_lower

sns.pairplot(iris,hue='species',palette='rainbow') #Species 기준으로
#This class maps a dataset onto multiple axes arrayed in a grid of 
#rows and columns that correspond to levels of variables in the dataset.
g = sns.FacetGrid(tips, col="time",  row="smoker") #TT,TF,FT,FF나눠주기
g = g.map(plt.hist, "total_bill")

#Multi-index로 설정해주기
g = sns.FacetGrid(tips, col="time",  row="smoker",hue='sex')
# Notice hwo the arguments come after plt.scatter call
g = g.map(plt.scatter, "total_bill", "tip").add_legend()

#Regression Plots
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex')
#Scatter 도 해주고 linear regression도 표시해줌
sns.lmplot(x="total_bill", y="tip", row="sex", col="time",data=tips)
#Row,Column추가 설정해서 다른 경우들이랑도 비교

```

`kde` - Rather than using discrete bins, a **KDE plot smooths the observations** with a Gaussian kernel, producing a continuous density estimate

```python
pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot([0, 1, 2, 3, 4, 5])
#Without the histogram
sns.distplot([0, 1, 2, 3, 4, 5], hist=False)
plt.show()

sns.countplot(x = 'column값', data = df)

```

```python
pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot([0, 1, 2, 3, 4, 5])
#Without the histogram
sns.distplot([0, 1, 2, 3, 4, 5], hist=False)
plt.show()

sns.countplot(x = 'column값', data = df)
sns.countplot(x = 'Day of Week', data = df, hue = 'reason')

```

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_11.21.51_AM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_11.21.51_AM.png)

```python
byMonth = df.groupby('Month').count()
byMonth['lat'].plot() #plotting the values

sns.lmplot() #이쁘게 linear fit 만들어주는 것
#여기 놓으면 column화 되어진다
sns.lmplot(x = 'Month', y = 'twp', data=byMonth.reset_index())
df.tight_layout()

Heatmap and Matrix forms

```

# PANDAS

```python
label = ['here', 'comes', 'the', 'label']
values = ['this', 'is', 'the', 'value']
pd.Series(data = values, index = label) #차원 다르면 빠꾸

#Series 거의 모든 데이터 타입을 머금을 수 있다
labels = ['a','b','c']
arr = np.array([10,20,30])
pd.Series(labels, arr)

#대략적인 information을 가지고 오는 것
ecom.info()

#Dataframe
df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
순서 -> 알맹이, x축,  y축
#new column
df['new'] = df['W'] + df['Y']
#deleting column
df.drop('new',axis=1) #axis 0 -row, axis 1 - column
#df.loc, df.iloc
df.iloc[2]
df.loc[['A','B'],['W','Y']]

#conditional
df>0 #T/F로만 나오는 테이블
df[df>0]
df[df['W']>0][['Y','X']] #x,y 축을 가져오기
df[(df['W']>0) & (df['Y'] > 1)] #satisfy both conditions

#indexing
df.reset_index() #0,1,2...새로운 index들 만들어줌
df.set_index('States') #이름 지어주기
# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

df.loc['G2'].loc[2]['B'] #table specifics

순서 -> 특정 value, 
df.xs('G1') #extract one with 'G1'
df.xs(['G1',1]) #1st row
df.xs(1,level='Num') #level Num 이 1인것들

#Finding Missing Data
df.dropna() #row 기준으로 NaN 있으면 바로 지우기
df.dropna(axis = 1) #column with null value
#thresh: thresh takes integer value which tells minimum amount of na values to drop
df.dropna(thresh = 2) #2개 NaN 까지 허용하는 것
df.fillna(value = '넣을 값')

#Using groupby
by_comp = df.groupby("Company")
by_comp.mean()
by_comp.std()
by_comp.min(), by_comp.max()
by_comp.count()
by_comp.describe()
by_comp.transpose()
by_comp.describe().transpose()['GOOG'] #for specific company

#Merge Join and Concatenate
df1, df2, df3 있을 때
df.concat([df1, df2, df3])
df.concat([df1, df2, df3], axis = 1)
#Merge - key 가 똑같을 때
pd.merge(left,right,how='inner',on='key')
how = 'left', 'right', 'outer'...
#Join syntax
left.join(right, how='outer')

#Some Operations
import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df['col2'].unique() #unique 한 것들
**df['col2'].value_counts() #몇개있는지**

**#Using Functions**
def times(n):
	return n*2
df['col1'].apply(times2)
df['col3'].apply(len) #return length of each string
#using lambda
df['col2'].apply(lambda x: x * 2)

del df['col1']
df.sort_values(by = 'col2') #뭘 기준으로 놓을 건지

df.isnull() #null 인지 boolean value
df.dropna() #NaN value 없애기

#Multiple values
df.pivot_table(values='D',index=['A', 'B'],columns=['C'])

#Boolser
boolser = df['W']>0
result = df[boolser] #return the values with

```

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-06-07_at_1.53.06_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-06-07_at_1.53.06_PM.png)

boolser 에서 True 인 것들만 추출되서 나옴

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-06-07_at_1.54.47_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-06-07_at_1.54.47_PM.png)

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-06-07_at_2.03.47_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-06-07_at_2.03.47_PM.png)

```python
pip install pandas
import pandas as pd
print(pd.__version__)
#General form - like dictionary form
myvar = {'Male': [12, 21], 'Female': [144, 423], index = ["age","bp"]} 
myvar = pd.DataFrame(mydataset)
a = pd.Series([1,2,3,4,5,6]) #index랑 자동으로 같이생성
print(a[0]) # = 1, bc 1st element
calories = {"day1": 420, "day2": 380, "day3": 390}
print(pd.Series(calories,index = ["day1", "day2"])) #pull out 2 only

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data) #calories and duration as 'column' values
print(df.loc[0]) #locating the column = calories : 420, duration : 50
print(df.loc[0,1]) #return row 1 and row 2
print(df.iloc[0]) #locating the row
df.dropna() # dropping all the null values

#idxmax() -> used to get the row label of the maximum value
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
sal.loc[sal['TotalPay'].idxmax()] #index로 찾을 때 해당 값들을 다 return
```

e.g. Fifa example & winery example

<aside>
💡 apply() 사용해서 function defined을 넣기!

</aside>

<aside>
💡 argmin() function returns the indices of the minimum value present in the input Index.

</aside>

```python
fifa_filepath = '/Users/SBMUN/Desktop/Self/fifa.csv'
reviews_filepath = '/Users/SBMUN/Desktop/Self/winemag-data-130k-v2.csv'
data = pd.read_csv(fifa_filepath, index_col = 0) #make use of 0th column as the index column
reviews = pd.read_csv(reviews_filepath, index_col = 'country')

data.head() #first 5, can modify data
data.iloc[0] #1st row [row,column] for index
data.iloc[[0,1,2],0] #same as data.iloc[:3,0] -> ARG의 값 date-ARG 3개
data.set_index("New Index Name") # set as index title

#min/max 를 판단할 때

#Additional Conditions
data.loc[(data.Date == '1993-08-08') & (data.ARG > 1.0)] #AND
data.loc[(data.Date == '1993-08-08') | (data.ARG > 5.0)] #OR
data.loc[data.ARG.**isin**(['5.0', '7.0'])] #ARG = 5.0 || 7.0

#Modify the data
data['Date'] = '2020-11-02' 
del review['description'] #Drop the column GER
reviews.price.describe() #summary of stats
reviews.taster_name.**unique()** #pull out unique
reviews.province.value_counts() #pull out frequency

#Transforming the data
**1. map**
reviews_mean = reviews.points.mean() #mean 먼저 구하기, 그리고 각각이 평균에서 얼마나 떨어진지
reviews.points.map(lambda p: p - reviews_mean) #해당 point - mean of points
**2.apply**
def point_mean(i): #
    i.points = i.points - reviews_mean
    return i
reviews.apply(point_mean, axis=1) #데이터.apply(함수)
**3. groupby**
reviews.groupby('province').province.count() #group & count by province
reviews.groupby('points').price.min() #데이터.groupby('c').항목.수행할 액션()
#first taster in each winery
reviews.groupby('winery').apply(lambda df:df.taster_name[0])
4. **agg()**
reviews.groupby(['country']).price.agg([len, min, max]).head() #len,min,max table of price by country

reviews.reset_index() #reset the index
reviewed_reset.sort_values(by='province', ascending = 'False') #sorting values
reviewed_reset.sort_values(by=['country', 'province'], ascending = True) #the front one is the priority value

```

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_10.18.24_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_10.18.24_PM.png)

Another datatypes

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_10.33.57_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-21_at_10.33.57_PM.png)

```python
**1. Data Types**
reviews.dtypes #data type 쭉 나옴
reviews.price.**astype**('float') #as type
reviews[pd.**isnull**(reviews.price)] #find price = NULL
reviews.region_2.**fillna**("Unknown") #In region_2, fill the NA with unknown
reviews.designation.**replace**("Vulkà Bianco", "Stupidland") #replace value

**2. Renaming**
reviews.rename(columns={'points': 'score'}) #change col name
reviews.rename(index={'Italy': 'firstEntry', 'Portugal': 'secondEntry'}) #change row name
#Also can reset the axis
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

**3. Combining**
#1. Import the pandas & the csv file to concatenate (have same rows)
import pandas as pd
ca_lo = '/Users/SBMUN/Desktop/Self/CAvideos.csv'
kr_youtube = pd.read_csv(ca_lo)
us_lo = '/Users/SBMUN/Desktop/Self/USvideos.csv'
us_youtube = pd.read_csv(us_lo)

pd.concat([kr_youtube,us_youtube]) #add them up
#put kr to left and us to right
left = kr_youtube.set_index(['title', 'trending_date'])
right = us_youtube.set_index(['title', 'trending_date'])
#put suffix for KR and US, add US to the right of KR
left.join(right, lsuffix='_KR', rsuffix='_US')

```

Pandas visualization library

```python
import numpy as np
import pandas as pd
%matplotlib inline

df1['A'].hist(bins = 30)
df.plot.area
df2.plot.area(alpha=0.4) #alpha - transparency
df.plot.barh
df2.plot.bar()
df2.plot.bar(stacked=True)

df.plot.density
df.plot.hist
**df.plot.line**
df1.plot.line(x=df1.index,y='B',figsize=(12,3),lw=1)

df.plot.scatter
df1.plot.scatter(x='A',y='B',c='C',cmap='coolwarm')
df1.plot.scatter(x='A',y='B',s=df1['C']*200) #s indicating size
df.plot.bar 
df.plot.box
**Useful for Bivariate Data, alternative to scatterplot:**
df.plot.hexbin
df.plot.kde
df.plot.pie

df3.ix#slicing 할 때 사용하는 것들

```

- Choropleth Maps (Geographical Plotting)
- Plotly and Cuffkins

# MATPLOTLIB

[http://localhost:8888/lab/tree/Desktop/Morgan Stanley/Jupyter Notebook/course_list/05-Data-Visualization-with-Matplotlib/01-Matplotlib Concepts Lecture.ipynb](http://localhost:8888/lab/tree/Desktop/Morgan%20Stanley/Jupyter%20Notebook/course_list/05-Data-Visualization-with-Matplotlib/01-Matplotlib%20Concepts%20Lecture.ipynb)

```python
import matplotlib.pyplot as plt
%matplotlib inline

#x,y defined
plt.plot(x,y,'r') #r = color
plt.xlabel('x value'), plt.ylabel('y value')

# plt.subplot(nrows, ncols, plot_number)
plt.subplot(1,2,1) #1 X 2 따리 중 1st
plt.subplot(1,2,2) #1 X 2 따리 중 2nd

#Object Oriented - create empty canvas
fig = plt.figure()
#add_axes(self, *args, **kwargs)
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes (left, bottom, width, height)
#10% from left, 10% from bottom

# Empty canvas of 1 by 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2) #axes -> array of axes (iterate 가능)
# Use similar to plt.figure() except use tuple unpacking to grab fig and axes
fig, axes = plt.**subplots**()
plt.tight_layout() #더 깔끔하게

# Empty canvas of 1 by 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(x,y)
axes[1].plot(x,y)

#Size Control
fig = plt.figure(figsize=(8,4), dpi=100)
fig.savefig("filename.png") #저장할 때

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend(loc = 0) #설명해주는 box를 추가해주는 방법
ax.legend(loc=1) # upper right corner
ax.legend(loc=2) # upper left corner
ax.legend(loc=3) # lower left corner
ax.legend(loc=4) # lower right corner

#Color and Transparency
ax.plot(x, x+1, color="blue", alpha=0.5, lw=3, ls='-.') # half-transparant

#Setting the range
axes[2].set_ylim([0, 60])
axes[2].set_xlim([2, 5])

#Style variation
#1. Scatter
plt.scatter(x,y)
#2. Histogram
data = sample(range(1, 1000), 100)
plt.hist(data)
#3. Boxplot
plt.boxplot(data,vert=True,patch_artist=True);

```

```python
pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

xpoint = np.array([0,10]) #x axis 0~10
ypoint = np.array([0,50]) #y axis 0 ~ 50
plt.plot(xpoints, ypoints) #y = 5x form line is drawn 
plt.plot(xpoints, ypoints, 'o') #just dot on each side
plt.show() #display the plot

x = np.array([1,3,5,7])
y = np.array([2,4,6,8]) 
plt.plot(x,y) #(1,2), (3,4), (5,6), (7,8)
plt.plot(x1, y1, x2, y2) #2 array씩 쌍, 그래서 그래프 2개 나옴

plt.title("This is the demo") #title for graph
plt.xlabel("Average Pulse") #labeling x
plt.ylabel("Following value") #labeling y

#family, color, size
font1 = {'family':'serif','color':'blue','size':20} #fontdict

ypoints = np.array([3, 8, 1, 10])
#emphasize the points with coordinates (and size) (and marker color) (and market border color)
plt.plot(ypoints, marker = 'o' ms = 20 mec = 'r', mfc = 'b') 
plt.plot(ypoints, marker = '*') #can change the shape

#Colors : r,g,b,c,m,y,k,w
plt.plot(ypoints, 'o-r') #single line
plt.plot(ypoints, 'o:r') #dotted line
plt.plot(ypoints, 'o--r') #dash line (--)
plt.plot(ypoints, 'o-.r') #mix of dash and dot

plt.plot(ypoints, linestyle = '')#dashed, dotted, dashdot, none
plt.plot(ypoints, color = 'r', linewidth = 20)

plt.plot(x,y)
plt.grid(axis = 'x') #leave x axis grid only - do y and leave y
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5) #change style

plt.suptitle("MY SHOP") #title for entire scheme
plt.subplot(1, 2, 1)
plt.plot(x,y)
#the figure has 1 row, 2 columns, and this plot is the first plot.
plt.subplot(1, 2, 2)
plt.plot(x,y)
#the figure has 1 row, 2 columns, and this plot is the second plot.

#Scatter plot
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y, color = 'hotpink') #just bunch of points
colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])
#separate colors for each dots & sizes & transparency
plt.scatter(x,y,color = colors, s = sizes, alpha = 0.5) 
plt.colorbar() #use the color bar and place next

#Bars
x = np.array(["A", "B", "C", "D"]) #labeling
y = np.array([3, 8, 1, 10]) #value for each label
c = np.array(['r','b','c','g'])
plt.bar(x,y, color = c, width = 1.2) #normal bar chart (chage color)
plt.barh(x,y, height = 0.7) #horizontal bar chart

#Histogram & bar chart
x = np.random.normal(170, 10, 250) #mean, s.d, sample size
plt.hist(x)
y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
myexplode = [0.2, 0, 0, 0] #how far exclude from it?
plt.pie(y, labels = mylabels,  startangle = 90, explode = myexplode, shadow = True)
plt.legend(title = "Four Fruits:") #add the label
plt.pie(x)

#Axis Grid
fig, axes = plt.subplots(1, 2, figsize=(10,3))

# default grid appearance
axes[0].plot(x, x**2, x, x**3, lw=2)
axes[0].grid(True)

# custom grid appearance
axes[1].plot(x, x**2, x, x**3, lw=2)
axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

#Different Plots
fig, axes = plt.subplots(1, 4, figsize=(12,3))

axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
axes[0].set_title("scatter")

axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")

axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5);
axes[3].set_title("fill_between");

```

→ do not specify the points in the x-axis, they will get the default values 0, 1, 2, 3

→ Marker reference

→ angle orientation chart (options)

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-22_at_4.44.39_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-22_at_4.44.39_PM.png)

![Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-22_at_5.56.30_PM.png](Data%20Analysis%20Template%20cab9f8e3770149a98457cb7615716985/Screen_Shot_2021-01-22_at_5.56.30_PM.png)

# Sample Project - Finance

## Reference for the finer goods

[Remote Data Access - pandas-datareader 0.9.0rc1+2.g427f658 documentation](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-tiingo)

```python
from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
%matplotlib inline

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

# Bank of America
BAC = data.DataReader("BAC", 'stooq', start, end)

# CitiGroup
C = data.DataReader("C", 'stooq', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'stooq', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'stooq', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'stooq', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'stooq', start, end)

# Could also do this for a Panel Object
**각각 어떤 것들을 가지고 와서 implement 할 지**
df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'stooq', start, end)
#list of all the tickers used
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)
#layer 추가해서 각 column 이름 넣어주기
bank_stocks.columns.names = ['Bank Ticker','Stock Info']

#EDA - 각 ticker Close() price MAX
**#Option 1 - for loop**
for i in tickers:
    print(i)
    print(bank_stocks[i]['Close'].max())
**#option 2**
#xs - cross section method
#axis = 1 so columns
#Close() price for all the banks in the ticker
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()

#Each return stock data
returns = pd.DataFrame()
#Each pct_change()
for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()

#pairplot (returning the pairwise correlation)
#returns[1:] - dropping the NaN valuesn 
import seaborn as sns
sns.pairplot(returns[1:])

**# Worst Drop** 
#반복적으로 index of the min() 사용하려면
returns.idxmin()

**# Best Single Day Gain**
# citigroup stock split in May 2011, but also JPM day after inauguration.
returns.idxmax()

#Risk Analysis - STD
#Riskness depends on the standard deviation
returns.std() # Citigroup riskiest
returns.loc['2015-01-01':'2015-12-31'].std() # Very similar risk profiles, but Morgan Stanley or BofA

#Visualization (seaborn)
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],color='blue',bins=100)
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)

#Visualization (plt in matplotlib)
import matplotlib.pyplot as plt #Matplotlib implemented
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

#trending graph using plt
for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()

bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot(figsize = (12,4))

#MA (Moving Average)
#Rolling = 30
plt.figure(figsize=(12,6))
BAC['Close'].loc['2006-01-01':'2007-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].loc['2006-01-01':'2007-01-01'].plot(label='BAC CLOSE')
plt.legend()

sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)

```