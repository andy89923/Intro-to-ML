# ML Final Project - Car Price Prediction
###### tags: `110Fall` `ML` `HW`
#### **陳琮方 0816153**




## I. Introduction
在現在每家至少一部車，各家車廠百家爭鳴的年代，每家車廠相互競爭，不僅是想要在車子上的技術提升來吸引消費者，或是在價格上、或是說 CP 值上能夠讓消費者買單。
在車廠設計並製造出來新的一台車的時候，要訂車子的價格實在是非常難的一件事情，若能夠根據車子的不同數據的關係，藉由機器學習等方法，訂出價格區間，幫助車廠訂出合理的價格。
也能夠幫助消費者在車廠公布實際售價之前，搶先一步預測出新車售價，提早開始存錢或是打消念頭 XD。




## II. Data Collection

我這裡使用 [UCI](https://archive.ics.uci.edu) 網站上的 [Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/automobile)，裡面的資料包含各種汽車資訊（詳細會在下面說明），其中包含汽車之預售價格（美金），即為我們目標的資訊。

### Dataset

1. symboling                 -3, -2, -1, 0, 1, 2, 3 (Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less), this symbol is adjusted by moving it up (or down) the scale. Actuarians call this process "symboling". A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.)
2. normalized-losses         normalized losses in use as compared to other cars.
3. Car Manufacturer          汽車經銷商 Multi class
4. fuel-type:                使用燃料種類 Binary (diesel/gas)
5. aspiration:               進氣種類：Binary (std(標準)/turbo(渦輪))
7. num-of-doors:             客用門數 Multi class 
8. body-style:               車型 Multi class
9. drive-wheels:             驅動輪 Multi class
10. engine-location:         引擎位置 Binary (front/back)
11. wheel-base               Distance between center of front and rear wheels, continuous
12. length:                  車長 continuous
13. width:                   車寬 continuous
14. height:                  車高 continuous
15. curb-weight:             全配油滿車重 continuous
16. engine-type:             引擎的種類 Multi class
17. num-of-cylinders:        汽缸數量 Multi class
18. engine-size:             引擎大小 continuous
19. fuel-system:             近油種類 Multi class
20. bore:                    引擎活塞大小 continuous
21. stroke:                  引擎行程循環 continuous
22. compression-ratio:       引擎壓縮率 continuous
23. horsepower:              馬力 continuous
24. peak-rpm:                引擎最大轉數 continuous
25. city-mpg:                城市油耗 continuous
26. highway-mpg:             公路油耗 continuous
27. price:                   價格

#### Features Distribution

* #### Car Manufacturer Distribution
可以從下圖統計圖中看到，Toyata 所佔有之車輛在這個資料集中佔比最大。
![](https://i.imgur.com/Ws57dFe.png)


* #### Car Price Distribution
![](https://i.imgur.com/HXBp2tm.png)
```
Price 𝜇  = 11445.729559748428, 𝜎 = 5859.343216339359
```
![](https://i.imgur.com/21w6Wvw.png)



* #### Features 之間的關係

這裡我算出不同 feature 之 correlation 關係，從最後一個 row 可以看到不同 feature 對於 price 的關係為何，進而在後面的 process 中選出比較好的 feature 留下，刪去關係較少低較少的（沒用）的 features。

![](https://i.imgur.com/bXKY3Jk.png)
![](https://i.imgur.com/ACgmjuj.png)
可以藉由 HeatMap 更容易看出重要的 feature，可以得到以下結論：
* Price 除了 Height, compression-ratio 這兩個 feature 之外，與其他 Numerical 都有非常明顯的正或負相關
* 我們這裡可以預估，因為大部分的 numerical features 與 price 的相關程度高，如果只使用 numerical 的 feature ，結果應該會不錯



<Note> 這裡只看 feature 為 numerical，非數字的 features 無法計算相關性。




## III. Preprocessing

根據前面分析車子價錢的分佈，我將資料的車子分成下面五類：
"0 ~ 10000", "10001 ~ 15000", "15001 ~ 20000", "20001 ~ 25000", "25001 ~ 30000", "Above 30000"

| Price | 0 ~ 10000 | 10001 ~ 15000 | 15001 ~ 20000 | 20001 ~ 25000 | 25001 ~ 30000 | Above 30000 |
|:-----:| --------- | ------------- | ------------- | ------------- | ------------- | ----------- |
| Label | 0         | 1             | 2             | 3             | 4             | 5           |

其中，包含很多 Multi class 或是 Binary 的 features ，所以我將這些 features 分成需要 oneHot encoded 跟不需要的：
```python=
oneHot = [
    'Car Manufacturer', 'fuel-type', 'aspiration', 'drive-wheels',  
    "engine-type", 'num-of-doors', 'body-style', 'engine-location', 
    'fuel-system', 'num-of-cylinders',
]

others = [
    "symboling", "normalized-losses", "wheel-base", 
    "length", "width", "height", "curb-weight",
    "engine-size", "bore", "stroke", "compression-ratio", "horsepower",
    "peak-rpm", "city-mpg", "highway-mpg"
]
```
並對 ```oneHot``` 做 oneHot encoder
```
Number of binary features: 10
Number of others features: 15
 => Sum of unique categorical: 49
=====================================
X size after one-hot: (159, 64)
```
最後將他們合在一起，可以看到 feature size 變成 64 個，對於不同的 oneHot 的 feature 共有 49 個。


## IV. Models

這裡我共實驗了三種上課教過的 Model，包含 Decision Tree, Random Forest, and Logistic Regression。

* ### Decision Tree

對於分類問題，我首先採用 Decision Tree 來實驗這個問題。
    
![](https://i.imgur.com/73qlPhE.png)
這裡我先將 PCA 降為不同的數量，並測試看看成效如何。
使用 Hold-Out Validate 的方式，相同的 trainning 資料集。最後我猜用 ```dim =  6```。
    
![](https://i.imgur.com/2QUz8xw.png)
上圖為 Decision Tree 的 Predict 路徑，可以經由數據的數值，以及在每個 node 的判斷式決定往左走還是往右走，走到葉節點決定預測的輸出。


* ### Random Forest
    
這裡與 Decision Tree 相同，藉由 Hold-Out Validate 去找到最好的 PCA Dimention，作為後面實驗的參數。這裡我使用 ```dim = 6```。
![](https://i.imgur.com/NiUD0US.png)
Random Forest 與 Decision Tree 概念相同，只是同時有很多不一樣的 Decision Tree，最後預測的結果經由 Majority Vote 去做決定，準確度理論上要高於 Decision Tree。
    

* ### Logistic Regression
這裡我嘗試使用 Logistic Regressing 用在 multi class 的分類上面。
這個模型有測試包含與不包含 OneHot feature 的 Input 做測試。

    
這裡與另外兩種模型做的事相同，藉由 Hold-Out Validate 去找到最好的 PCA Dimention，作為後面實驗的參數。這裡我使用 ```dim = 10```。
![](https://i.imgur.com/92cYPZh.png)
Logistic Regression 會藉由演算法，將不通的參數給予不同的 weights，最後輸出與 Linear 不同的就是 Logistic 會是離散的，對於分類來說也是可以使用。

### <Note> 
* PCA 只針對 numerical 的 features，不會受到 OneHot 的 features 影響。
* Logistic Regression 的參數也包含 OneHot（品牌等 OneHot feature 也會對 price 造成影響)
    

## V. Results

以下是不同模型的實驗數據，這裡都採用 3 lebel K-fold 平均後的結果：

* ### Decision Tree

| Model                  | Accuracy | Recall | Precision | Confusion Matrix                     |
|:---------------------- | -------- | ------ |:---------:|:------------------------------------ |
| Decision Tree          | 73.5833  | 0.4133 |  0.3933   | ![](https://i.imgur.com/9TXTecU.png) |
| Decision Tree with PCA | 68.5533  | 0.4400 |  0.4367   | ![](https://i.imgur.com/rjZSF6c.png) |

Decision Tree 在加上 PCA 降低維度之後，準確率明顯的下降，原因可能為降低 feature 會導致在走 decision tree 的時候，會有更多模糊、或是不明確的影響（降維後變得不明顯)，所以準確率下降。
DT 在跑在 Trainning set 的時候，準確率非常的高，有點造成 Overfit，所以這也會造成在 Testing set 表現不夠好的原因之一。

* ### Random Forest

| Model            | Accuracy | Recall | Precision | Confusion Matrix                     |
|:---------------- |:-------- | ------ |:---------:|:------------------------------------ |
| RF-20 Binary     | 74.8433  | 0.5300 |  0.4900   | ![](https://i.imgur.com/hYHkOGW.png) |
| RF-20 Binary PCA | 72.3267  | 0.5600 |  0.5500   | ![](https://i.imgur.com/ryZgvxo.png) |
| RF-40 Binary     | 76.1033  | 0.5700 |  0.4967   | ![](https://i.imgur.com/kZPgpCR.png) |
| RF-40 Binary PCA | 77.9867  | 0.5367 |  0.4367   | ![](https://i.imgur.com/FN5N0qj.png) |

我測試了 20 棵樹與 40 棵樹，也分別使用 PCA 降低 features 的 dimention 。可以從實驗數據知道，40 棵樹的結果都比 20 棵樹還要好(淺而易見 越多棵樹理論上錯誤機會越低)。
對於 PCA 的使用，沒有造成太大的影響。


* ### Logistic Regression

* #### With OneHot Features
    
| Model                        | Accuracy | Recall | Precision | Confusion Matrix                     |
|:---------------------------- | -------- | ------ |:---------:|:------------------------------------ |
| Logistic Regression          | 72.3267  | 0.3767 |  0.3833   | ![](https://i.imgur.com/rpN3gVM.png) |
| Logistic Regression with PCA | 70.4400  | 0.4367 |  0.4233   | ![](https://i.imgur.com/OKn9S4V.png) |

* #### Without OneHot Features
      
| Model               | Accuracy | Recall | Precision | Confusion Matrix                     |
|:------------------- | -------- | ------ |:---------:|:------------------------------------ |
| Logistic Regression | 69.8067  | 0.4433 |  0.3433   | ![](https://i.imgur.com/9mMOUnp.png) |

經過測試，只有 Numerical features 在 Logistic Regression 不會有更高的 Accuracy ，與前面的猜測違背。我推測的原因為因為車價還是會受到「品牌」、「引擎種類」等 OneHot features 會對 Price 造成影響，所以拔掉這些 features 會造成準確率下降的原因。


## VI. Conclusion

這次期末作業總體來說，資料集有點舊、資料量有點少，但經由不同的模型之間的比較，可以看到如果資料能夠引進新一點的資料（最近幾年的車），以及更多的 features ，可能包含車子的配備，可以升級的等級，或是將電動車的規格也納入進來，一定可以有更好的利用空間。
    
    
#### Review
這次的 Final Project 為修完 ML 這門課的期末作業，要從 0 到 100 完整規劃整個機器學習的流程，不像是作業一樣，一項項的打勾，而是要有前面分析的結果，帶入後面模型選擇、或是資料預處理時需要做的事情，都要完整規劃好，才是一個好的機器學習流程。
    
