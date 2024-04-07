# Homework#1: Implementing Fully-connected Neural Network
### 이름: 조정환

### 학번: 2022314309
---
### Before start

    np.__version__, torch.__version__ #check version -> ('1.25.2', '2.2.1+cu121')
    torch.manual_seed(42)
  
## Task 1
* Using Numpy

      # prepare inputs, weights
  
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
        hidden_nodes = np.dot(input, w1_n) # before first activation function
        hidden_nodes_act = ReLU(hidden_nodes) # after first activation function
        if dropout == True: # check whether to apply dropout or not (by NN_np model's parameter)
          np.random.seed(random_seed) 
          drop = np.random.rand(4) # 히든 노드의 각 dropout 확률 설정 (만약 p보다 크면 dropout)
          drop = drop < p # dropout될 노드들은 False로 표시
          hidden_nodes_act = hidden_nodes_act * drop # dropout된 노드들은 0으로 변경됨.
          hidden_nodes_act /= (1-p) # 남아있는 노드들의 값 변경
        output_nodes_np = np.dot(hidden_nodes_act, w2_n) # before second activation function
        output_nodes_act = Softmax(output_nodes_np) # after second activation function
        return hidden_nodes, hidden_nodes_act, output_nodes_np, output_nodes_act
      
      print('when input is x1: ', np.round(NN_np(x1_n)[3], 4)) # input x1에 대해 output_nodes_act 반환 후 반올림
      print('when input is x2: ', np.round(NN_np(x2_n)[3], 4)) # input x2에 대해 output_nodes_act 반환 후 반올림

      # when input is x1:  [0.1324 0.8676]
      # when input is x2:  [0.0145 0.9855]
  
* Using Pytorch

      
