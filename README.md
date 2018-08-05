
## Dot Product

The dot product is a crucial mathematical operation that we'll be using in many algorithms going forward.  
It is defined as the sum of the products of the corresponding elements of two vectors.  

Mathematically:  
$ a = [a_1, a_2,...a_n]$  
$ b = [b_1, b_2,...b_n]$  
   
$ a \bullet b = \sum_{i=1}^{n} a_ib_i + a_2b_2 + ... + a_nb_n$


```python
import numpy as np
a = np.array(range(5))
b = np.array(range(5,10))
print('a :', a)
print('b :', b)
```

    a : [0 1 2 3 4]
    b : [5 6 7 8 9]


### 1. Write a function to calculate the dot product.


```python
def dot_product(a,b):
    return sum(a*b)
```


```python
dot_product(a,b)
```




    80



### 2. Dot Product 2
Great! The dot product of a and b can also be calculated by:

$a\bullet b = a^Tb$ 

Recall that $a^T$ is the transpose of a.

Write a second function that calculates the dot product of a and b using this alternative calculation.


```python
def dot_product2(a,b):
    return np.matmul(a.transpose(), b)
```


```python
dot_product2(a,b)
```




    80



### Polynomial Functions
Soon, we're going to expand our simple linear regression into the more generalized linear regression involving multiple variables. Instead of looking at the Gross Domestic Sales of a movie in terms of its budget alone, we'll consider more variables such as ratings and reviews to improve our predictions. 

When doing this, we will have a matrix of data where each column is a specific feature such as the budget, or the imdb review score, while each row will be an observance, one of the movies in our dataset.

$x_1\bullet w_1 + x_2\bullet w_2 + x_3\bullet w_3 + ... = y$

For example


```python
import pandas as pd
x = pd.read_excel('movie_data_detailed_with_ols.xlsx')
x = x[['budget', 'imdbRating','Metascore', 'imdbVotes']]
x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>imdbRating</th>
      <th>Metascore</th>
      <th>imdbVotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000000</td>
      <td>6.8</td>
      <td>48</td>
      <td>206513</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45658735</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000000</td>
      <td>8.1</td>
      <td>96</td>
      <td>537525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61000000</td>
      <td>6.7</td>
      <td>55</td>
      <td>173726</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40000000</td>
      <td>7.5</td>
      <td>62</td>
      <td>74170</td>
    </tr>
  </tbody>
</table>
</div>




```python
x = np.array(x)
x
```




    array([[1.3000000e+07, 6.8000000e+00, 4.8000000e+01, 2.0651300e+05],
           [4.5658735e+07, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
           [2.0000000e+07, 8.1000000e+00, 9.6000000e+01, 5.3752500e+05],
           [6.1000000e+07, 6.7000000e+00, 5.5000000e+01, 1.7372600e+05],
           [4.0000000e+07, 7.5000000e+00, 6.2000000e+01, 7.4170000e+04],
           [2.2500000e+08, 6.3000000e+00, 2.8000000e+01, 1.2876600e+05],
           [9.2000000e+07, 5.3000000e+00, 2.8000000e+01, 1.8058500e+05],
           [1.2000000e+07, 7.8000000e+00, 5.5000000e+01, 2.4008700e+05],
           [1.3000000e+07, 5.7000000e+00, 4.8000000e+01, 3.0576000e+04],
           [1.3000000e+08, 4.9000000e+00, 3.3000000e+01, 1.7436500e+05],
           [4.0000000e+07, 7.3000000e+00, 9.0000000e+01, 3.9839000e+05],
           [2.5000000e+07, 7.2000000e+00, 5.8000000e+01, 7.5884000e+04],
           [5.0000000e+07, 6.2000000e+00, 5.2000000e+01, 7.6001000e+04],
           [1.8000000e+07, 7.3000000e+00, 7.8000000e+01, 1.7098600e+05],
           [5.5000000e+07, 7.8000000e+00, 8.3000000e+01, 3.6824400e+05],
           [3.0000000e+07, 7.4000000e+00, 8.5000000e+01, 1.4232800e+05],
           [7.8000000e+07, 6.4000000e+00, 5.9000000e+01, 7.5138000e+04],
           [7.6000000e+07, 7.4000000e+00, 6.2000000e+01, 3.2466400e+05],
           [5.5000000e+06, 6.6000000e+00, 6.6000000e+01, 2.0894800e+05],
           [1.2000000e+08, 6.6000000e+00, 6.1000000e+01, 3.7813100e+05],
           [1.1000000e+08, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
           [1.0000000e+08, 6.7000000e+00, 5.2000000e+01, 9.2389000e+04],
           [4.0000000e+07, 5.9000000e+00, 3.5000000e+01, 2.2430000e+04],
           [7.0000000e+07, 6.7000000e+00, 4.9000000e+01, 1.9876700e+05],
           [1.7000000e+07, 6.5000000e+00, 5.7000000e+01, 1.3994000e+05],
           [1.6000000e+08, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
           [1.5000000e+08, 7.5000000e+00, 7.4000000e+01, 4.8355500e+05],
           [1.4000000e+08, 5.8000000e+00, 4.1000000e+01, 1.5821000e+05],
           [6.0000000e+07, 6.7000000e+00, 4.0000000e+01, 1.8884600e+05],
           [3.0000000e+07, 7.1000000e+00, 0.0000000e+00, 0.0000000e+00]])



### 3. Write a function that predicts a vector of model predictions $\hat{y}$ given a matrix of data x, and a vector of coefficient weights w.   
Mathematically:   
$x_1\bullet w_1 + x_2\bullet w_2 + x_3\bullet w_3 + ... = y$


```python
def poly_regress_predict(x,w):
    return y_hat
```


```python
def poly_regress_predict(x,w):
    y_hat = np.dot(x,w)
    return y_hat
```
