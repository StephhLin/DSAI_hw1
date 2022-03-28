# DSAI_hw1
>使用Keras LSTM 模型預測台電備轉容量

## 資料預處理

```sh 
!gdown --id '1YDXZHB4zuSMxRh5wTF-Q3NNTdXeUzx5o' --output operating_reserve.csv <br>
!ls <br>
```
將圖片從google雲端硬碟下載到colab <br>
```sh 
dataframe = read_csv('operating_reserve.csv', usecols=[1], engine='python', skipfooter=3) <br>
dataset = dataframe.values <br>
dataset = dataset.astype('float32') <br>
```
為了適合神經網路建模，我們將數據加載成pandas數據格式，且將數據從整數變成浮點數型別。<br>
```sh
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
```
使用scikit-learn 庫中的MinMaxScaler預處理來標準化數據集，範圍0~1之間。
```sh
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```
我們使用的是將有序數據集拆分為訓練數據集和測試數據集。其中 50% 的觀察可以用來訓練我們的模型，剩下的 50% 用於測試模型。
```sh
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
  
  
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
```
現在我們以一個函數creat_dataset，來創建一個新的數據集。<br>
其中有兩個參數分別為:<br>
* dataset是我們想要轉換為數據集的 NumPy 數組。<br>
* look_back用來當作預測下一個時間的input，我們選擇用1。<br>

其中，創造兩個資料集， X 是給定時間 (t) 的各個備轉容量，Y 是下一次 (t + 1) 的各個備轉容量。<br>

```sh
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
```
使用numpy.reshape()，將準備好的訓練和測試輸入數據轉換為適合LSTM 網絡的輸入數據。<br>
輸入格式為[samples, time steps, features]<br>


## 開始建立及訓練LSTM模型<br>
```sh
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
```
>其中有 1個 input 到 visible layer，hidden layer 有4個神經元， output layer 輸出單一值的預測資料。<br>
* epochs = 100<br>
* batch_size = 1<br>
##### 以下為Keras導入模塊 : <br>
* Sequential() 用於初始化神經網路<br>
* LSTM(4, input_shape=(1, look_back)) 用於添加長短期記憶層<br>
* Dense(1) 用於添加密集連接的神經網路 <br>
* 用adam優化器編譯我們模型
* 用mean_squared_error計算誤差

## 預測
```sh
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
```
## 結果 
```sh
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
```
回復預測資料值為原始數據的規模。<br>
```sh
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
```
使用root mean squared error計算均方根誤差。<br>
```sh
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
```
作圖<br>
* 藍色顯示原始數據集。
* 橘色顯示訓練數據集的預測。
* 綠色顯示未見過的測試數據集的預測。

