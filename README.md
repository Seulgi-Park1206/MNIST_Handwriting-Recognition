### MNIST_Handwriting-Recognition
* Python을 이용한 필기체 글자 인식 프로그램(마우스로 숫자를 그려서 입력)입니다.

#### Result
![result](https://user-images.githubusercontent.com/57548434/102866686-03c76f00-447b-11eb-80c0-4f6948431421.png)
#### Training
* MNIST data를 normalize=True, one_hot_label=True로 업로드하여 입력으로 하고, hidden_size = 500, batch_size = 100으로 하는 미니 배치 학습을 통해 얻은 가중치, bias, network 구조를 pickle 파일로 저장한다.
#### Test
* 마우스로 입력할 숫자를 키보드로 입력하고, 마우스로 그 숫자를 그리면 그린 숫자는 test data가 되고, 키보드로 입력한 숫자는 정답이 된다.
* Training 단계에서 얻은 pickle 파일을 통해 정답에 가까운 인덱스를 얻어 결과를 예측한다.

#### 보고서 [PDF](/HWR.pdf)