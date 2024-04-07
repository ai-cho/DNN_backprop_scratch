# Homework#1: Implementing Fully-connected Neural Network
### 이름: 조정환

### 학번: 2022314309
---
### Before start

    import numpy as np
    import torch
    from torch import nn
    import torch.optim as optim 
    
    np.__version__, torch.__version__ #check version -> ('1.25.2', '2.2.1+cu121')
    torch.manual_seed(42)
  
## Task 1
* Using Numpy

      # prepare inputs, weights using numpy
  
      x1_n = np.array([1.0, 2.0, 3.0])
      x2_n = np.array([4.0, 5.0, 6.0])
      
      w1_n = np.array([[0.1, 0.2, 0.3, 0.4],
                     [0.5, 0.6, 0.7, 0.8],
                     [0.9, 1.0, 1.1, 1.2]])
      w2_n = np.array([[0.2, 0.3],
                      [0.4, 0.5],
                      [0.6, 0.7],
                      [0.8, 0.9]])
  
      # make ReLU function
      def ReLU(x): # define ReLU using numpy
        return np.maximum(x, 0)

      # make Softmax function
      def Softmax(x): # define Softmax using numpy
        return np.exp(x)/sum(np.exp(x))
      
      # define neural network
      def NN_np(input, dropout = False, p = 0, random_seed = 42):
      # random_seed 설정 이유 (뒷부분도 동일)
      # dropout시 backpropagation을 하는 과정에서 NN_np 모델을 여러 번 호출하도록 구현되어있음. 만약 랜덤시드가 설정되어 있지 않다면 호출마다 dropout 노드가 달라지게 된다.
      # 한번의 epoch 동안은 랜덤시드 동일. epoch 달라지면 랜덤시드도 변경되도록 세팅되어 있음. (Task3에서 사용됨)
  
        hidden_nodes = np.dot(input, w1_n) # before first activation function
        hidden_nodes_act = ReLU(hidden_nodes) # after first activation function
        if dropout == True: # check whether to apply dropout or not (by NN_np model's parameter)
          np.random.seed(random_seed) 
          drop = np.random.rand(4) # 히든 노드의 각 dropout 확률 설정 (만약 p보다 크면 dropout)
          drop = drop < p # dropout될 노드들은 False로 표시
          hidden_nodes_act = hidden_nodes_act * drop # dropout된 노드들은 0으로 변경됨.
          hidden_nodes_act /= (1-p) # 남아있는 노드들의 값 변경 h/(1-p)
        output_nodes_np = np.dot(hidden_nodes_act, w2_n) # before second activation function
        output_nodes_act = Softmax(output_nodes_np) # after second activation function
        return hidden_nodes, hidden_nodes_act, output_nodes_np, output_nodes_act
      
      print('when input is x1: ', np.round(NN_np(x1_n)[3], 4)) # input x1에 대해 output_nodes_act 반환 후 반올림
      print('when input is x2: ', np.round(NN_np(x2_n)[3], 4)) # input x2에 대해 output_nodes_act 반환 후 반올림

  ## output
       when input is x1:  [0.1324 0.8676]
       when input is x2:  [0.0145 0.9855]  
  
* Using Pytorch

      # prepare inputs, weights using pytorch
  
        x1_t = torch.tensor([1.0, 2.0, 3.0])
        x2_t = torch.tensor([4.0, 5.0, 6.0])
        
        w1_t = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8],
                           [0.9, 1.0, 1.1, 1.2]]) 
        w2_t = torch.tensor([[0.2, 0.3],
                           [0.4, 0.5],
                           [0.6, 0.7],
                           [0.8, 0.9]]) 

        # define neural network
        class NN(nn.Module):
          def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 4, bias = False) # no bias
            self.linear1.weight.data = w1_t.T # set first weigth w1_t
            self.act1 = nn.ReLU()
            self.dropout = nn.Dropout(p=0.4) # dropout 시 사용 예정
            self.linear2 = nn.Linear(4, 2, bias = False)
            self.linear2.weight.data = w2_t.T # set first weight w2_t
            self.act2 = nn.Softmax(dim = -1)
        
          def forward(self, x, dropout = False): 
            x = self.linear1(x)
            x = self.act1(x)
            if dropout == True: # check whether to apply dropout or not (by NN model's parameter)
              x = self.dropout(x)
            x = self.linear2(x)
            x = self.act2(x)
            return x
        
        NN_torch = NN()
        print('when input is x1: ', NN_torch(x1_t))
        print('when input is x2: ', NN_torch(x2_t))

    ## output
      when input is x1:  tensor([0.1324, 0.8676], grad_fn=<SoftmaxBackward0>)
      when input is x2:  tensor([0.0145, 0.9855], grad_fn=<SoftmaxBackward0>)
---
## Task 2
* Using Numpy

        # weight, target 설정 (by numpy)
        w1_n = np.array([[0.1, 0.2, 0.3, 0.4],
                       [0.5, 0.6, 0.7, 0.8],
                       [0.9, 1.0, 1.1, 1.2]])
        w2_n = np.array([[0.2, 0.3],
                        [0.4, 0.5],
                        [0.6, 0.7],
                        [0.8, 0.9]])
        y1_n = np.array([0, 1])
        y2_n = np.array([1, 0])
  
        def CEE_np(y_pred, y_target):
          y_pred = Softmax(y_pred) # 위에서 만든 Softmax 함수 적용
          return -np.sum(y_target*np.log(y_pred)) # Cross Entropy Loss: - sum(target*log(Softmax(y_pred)))
          # output을 softmax한 다음에 대입해야함.
        
        def gradient_w1(input, target, i, j, dropout = False, p = 0, random_seed = 42): # i_1: 출발 노드 번호, j_1: 도착 노드 번호
          if dropout == True:
            output = NN_np(input, True, p, random_seed)[-1] # dropout 적용된 아웃풋으로 가져옴
          else:
            output = NN_np(input)[-1] # dropout 적용 안된 아웃풋
          dL_do = Softmax(output) - target # Cross Entropy Loss를 O에 대해 미분 
          do_dy = np.zeros((2,2)) # Softmax(y)를 y에 대해 미분
          for i_1 in range(2):
            for j_1 in range(2):
              if i_1 == j_1:
                do_dy[i_1][j_1] = output[0]*output[1] # 같은 번호의 노드인 경우
              else:
                do_dy[i_1][j_1] = -output[0]*output[1] # 다른 번호의 노드인 경우
          dy_dr_j = w2_n[j].T # y를 r[j] 에 대해 미분
          dr_j_dh = np.array([1 if NN_np(input)[0][j] > 0 else 0]) # ReLU(h)를 h에 대해 미분. h 값이 0보다 크면 1, 아니면 0
          dh_dw_ij = input[i] # h를 w에 대해 미분. 그냥 input 값

          # chain-rule 적용하여 계산 (dL/dw)
          result = np.dot(dL_do, do_dy)
          result = np.dot(result, dy_dr_j)
          result = np.dot(result, dr_j_dh)
          result = np.dot(result, dh_dw_ij)
          return round(result.item(), 4)
        
        def get_weight1_grad_np(input, target, dropout = False, p = 0, random_seed = 42):
          weight_grad = np.zeros((3,4)) # weight gradient 저장할 곳
          for start in range(3):
            for end in range(4):
              if dropout == True:
                weight_grad[start][end] = gradient_w1(input, target, start, end, dropout, p, random_seed = 42)
                idx = np.where(NN_np(input, True, p, random_seed)[1] == 0)[0] # dropout된 노드 인덱스 찾기
                for i in idx: # dropout된 노드로 도착하는 weight는 0으로 초기화
                  weight_grad[:, i] = 0.0
              else:
                weight_grad[start][end] = gradient_w1(input, target, start, end)
        
          return weight_grad
        
        print('when input is x1: \n', get_weight1_grad_np(x1_n, y1_n))
        print()
        print('when input is x2: \n', get_weight1_grad_np(x2_n, y2_n))

    ## output
        when input is x1: 
         [[-0.0074 -0.0074 -0.0074 -0.0074]
         [-0.0149 -0.0149 -0.0149 -0.0149]
         [-0.0223 -0.0223 -0.0223 -0.0223]]
        
        when input is x2: 
         [[0.0083 0.0083 0.0083 0.0083]
         [0.0104 0.0104 0.0104 0.0104]
         [0.0124 0.0124 0.0124 0.0124]]
  
* Using Pytorch
  
        # weight, target 설정 (by pytorch)
        w1_t = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8],
                           [0.9, 1.0, 1.1, 1.2]]) # (3, 4)
        w2_t = torch.tensor([[0.2, 0.3],
                           [0.4, 0.5],
                           [0.6, 0.7],
                           [0.8, 0.9]])  # (4, 2)
        y1_t = torch.tensor([0, 1], dtype = torch.float32) # float형으로 설정
        y2_t = torch.tensor([1, 0], dtype = torch.float32)
        
        def get_weight_grad_t(input, target):
          NN_torch = NN()
          loss_fn = nn.CrossEntropyLoss() # Loss 함수 설정
          pred = NN_torch(input)

          # backward
          loss = loss_fn(pred, target)
          loss.backward()
          result = NN_torch.linear1.weight.grad # w1의 gradient 얻기
          return result.T
        
        print('when input is x1:\n', get_weight_grad_t(x1_t, y1_t)) # input x1에 대한 w1 gradient
        print()
        print('when input is x2:\n', get_weight_grad_t(x2_t, y2_t)) # input x2에 대한 w1 gradient

    ## output
        when input is x1:
         tensor([[-0.0074, -0.0074, -0.0074, -0.0074],
                [-0.0149, -0.0149, -0.0149, -0.0149],
                [-0.0223, -0.0223, -0.0223, -0.0223]])
        
        when input is x2:
         tensor([[0.0083, 0.0083, 0.0083, 0.0083],
                [0.0104, 0.0104, 0.0104, 0.0104],
                [0.0124, 0.0124, 0.0124, 0.0124]])


## Task 3

* Using Numpy
  
        # w2 gradient 구하는 부분. 기본적인 형태는 Task2의 w1 gradient와 동일
  
        def gradient_w2(input, target, i, j, dropout = False, p = 0, random_seed = 42): # i_1: start, j_1: end
          if dropout == True:
            output = NN_np(input, dropout, p, random_seed)[-1]
          else:
            output = NN_np(input)[-1]
          dL_do = Softmax(output) - target # Cross Entropy Loss를 O에 대하여 미분 
          do_dy = np.zeros((2,2)) # softmax(y)를 y에 대하여 미분
          for i_1 in range(2):
            for j_1 in range(2):
              if i_1 == j_1:
                do_dy[i_1][j_1] = output[0]*output[1] # 같은 번호의 노드인 경우
              else:
                do_dy[i_1][j_1] = -output[0]*output[1] # 다른 번호의 노드인 경우
          dy_dw_j = NN_np(input)[1][i] # y를 w에 대해 미분. 그냥 h 값
          result = np.dot(dL_do, do_dy)[j]
          result = result * dy_dw_j
          return round(result.item(), 4)
        
        
        
        def get_weight2_grad_np(input, target, dropout = False, p = 0, random_seed = 42):
          weight_grad = np.zeros((4,2))
          for start in range(4):
            for end in range(2):
              if dropout == True:
                weight_grad[start][end] = gradient_w2(input, target, start, end, dropout, p, random_seed)
                idx = np.where(NN_np(input, True, p, random_seed)[1] == 0)[0] # 00인 지점은 가중치 0으로 세팅 #idx가 고정되는 error 발생
                for i in idx:
                  weight_grad[i, :] = 0.0
              else:
                weight_grad[start][end] = gradient_w2(input, target, start, end)
          return weight_grad
        # 여기까지가 w2 gradient 구하는 부분

        # input x1일 때의 updated weights
        w1_n = np.array([[0.1, 0.2, 0.3, 0.4],
                       [0.5, 0.6, 0.7, 0.8],
                       [0.9, 1.0, 1.1, 1.2]])
        w2_n = np.array([[0.2, 0.3],
                        [0.4, 0.5],
                        [0.6, 0.7],
                        [0.8, 0.9]])
        
        for i in range(100): # 100 epochs
          random_seed = i+40 # 랜덤시드 각 epoch 마다 바꿈.
          w1_grad = get_weight1_grad_np(x1_n, y1_n, True, 0.4, random_seed)
          w2_grad = get_weight2_grad_np(x1_n, y1_n, True, 0.4, random_seed)
          # 위에서 구한 각 weight의 gradient를 lr(학습률) 곱해서 뺴기
          # gradient descent
          w1_n -= 0.01*w1_grad
          w2_n -= 0.01*w2_grad
        print('input: x1')
        print()
        print('w1: ', np.round(w1_n, 4))
        print()
        print('w2: ', np.round(w2_n, 4))

        # input x2일 때의 updated weights
        # 위의 과정에서 w1_n, w2_n 의 값들이 update 됨.
        # w1_n, w2_n 초기화하고 진행 (위 과정과 동일)
        w1_n = np.array([[0.1, 0.2, 0.3, 0.4],
                       [0.5, 0.6, 0.7, 0.8],
                       [0.9, 1.0, 1.1, 1.2]])
        w2_n = np.array([[0.2, 0.3],
                        [0.4, 0.5],
                        [0.6, 0.7],
                        [0.8, 0.9]])
        
        for i in range(100): # 100 epochs
          random_seed = i+40
          w1_grad = get_weight1_grad_np(x2_n, y2_n, True, 0.4, random_seed)
          w2_grad = get_weight2_grad_np(x2_n, y2_n, True, 0.4, random_seed)
          w1_n -= 0.01*w1_grad
          w2_n -= 0.01*w2_grad
        print('input: x2')
        print()
        print('w1: ', np.round(w1_n, 4))
        print()
        print('w2: ', np.round(w2_n, 4))
  
  ## output
        input: x1
        
        w1:  [[0.1107 0.2103 0.3085 0.4107]
         [0.5214 0.6207 0.7169 0.8213]
         [0.9322 1.031  1.1254 1.232 ]]
        
        w2:  [[0.1191 0.3809]
         [0.3274 0.5726]
         [0.5234 0.7766]
         [0.7254 0.9746]]
        ===================================
  
        input: x2
        
        w1:  [[0.1104 0.2102 0.3097 0.4128]
         [0.513  0.6127 0.7122 0.8161]
         [0.9156 1.0152 1.1146 1.2193]]
        
        w2:  [[0.3463 0.1537]
         [0.5362 0.3638]
         [0.7412 0.5588]
         [0.9364 0.7636]]
  
* Using Pytorch

        # input x1일 떄의 updated weights
        w1_t = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8],
                           [0.9, 1.0, 1.1, 1.2]]) # (3, 4)
        w2_t = torch.tensor([[0.2, 0.3],
                           [0.4, 0.5],
                           [0.6, 0.7],
                           [0.8, 0.9]])  # (4, 2)
        
        NN_torch1 = NN()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(NN_torch1.parameters(), lr = 0.01)
        
        for epoch in range(100): # num epoch
          pred = NN_torch1(x1_t, True)
          loss = loss_fn(pred, y1_t)
          # backward
          optimizer.zero_grad() #gradient 누적안되게 각 epoch마다 초기화
          loss.backward()
          optimizer.step()
        print('input: x1\n')
        print('w1: ', NN_torch1.linear1.weight.T) # w1
        print('w2: ', NN_torch1.linear2.weight.T) # w2

        # input x2일 때의 updated weights
        # 위 과정에서 w1_t, w2_t update 됨.
        # 각 weight 다시 선언함으로써 초기화 후 진행
        # 위 과정과 동일
        w1_t = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8],
                           [0.9, 1.0, 1.1, 1.2]]) # (3, 4)
        w2_t = torch.tensor([[0.2, 0.3],
                           [0.4, 0.5],
                           [0.6, 0.7],
                           [0.8, 0.9]])  # (4, 2)
        
        NN_torch1 = NN()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(NN_torch1.parameters(), lr = 0.01)
        
        # input: x2
        for epoch in range(100):
          pred = NN_torch1(x2_t, True)
          loss = loss_fn(pred, y2_t)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        print('input: x2\n')
        print('w1: ', NN_torch1.linear1.weight.T)
        print('w2: ', NN_torch1.linear2.weight.T)

  ## output
        input: x1
        
        w1:  tensor([[0.1032, 0.2033, 0.3034, 0.4019],
                [0.5063, 0.6066, 0.7069, 0.8039],
                [0.9095, 1.0099, 1.1103, 1.2058]], grad_fn=<PermuteBackward0>)
        w2:  tensor([[0.1279, 0.3721],
                [0.3180, 0.5820],
                [0.5080, 0.7920],
                [0.7332, 0.9668]], grad_fn=<PermuteBackward0>)
        ===================================

        input: x2
        
        w1:  tensor([[0.1031, 0.2021, 0.3004, 0.4001],
                [0.5039, 0.6027, 0.7004, 0.8002],
                [0.9046, 1.0032, 1.1005, 1.2002]], grad_fn=<PermuteBackward0>)
        w2:  tensor([[0.3592, 0.1408],
                [0.5493, 0.3507],
                [0.7356, 0.5644],
                [0.9324, 0.7676]], grad_fn=<PermuteBackward0>)
