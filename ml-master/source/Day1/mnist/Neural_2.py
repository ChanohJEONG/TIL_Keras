import numpy
import scipy.special
import matplotlib.pyplot as plt

class NerualNetwork:
    # 신경망 초기화
    def __init__(self, inputnodes, hiddennodes, outputdones, learnrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputdones

        # 가중치 행렬 wih, who
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))                                        

        self.lr = learnrate

        # 활성화 함수 지정 -> sigmoid 
        self.activation_function = lambda x: scipy.special.expit(x)
        pass


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    
    # 신경망 학습
    def train(self, inputs_list, targets_list):
        
        pass

    # 신경망 테스트(질의)
    def query(self, inputs_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 은닉계층으로 들어오는 값 계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 은닉계층으로 나가는 값 계산
        hidden_outputs = self.activation_function(hidden_inputs)
        # 최종 출력 계층으로 들어오는 값 계산
        final_inptus = numpy.dot(self.who, hidden_outputs)
        # 최종 출력 계층으로 나가는 값 계산
        final_outputs = self.activation_function(final_inptus)

        return final_outputs

input_nodes = 784
hidden_nodes =100
output_nodes = 10
learning_rate = 0.3

n = NerualNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print(n.query([1.0, 0.5, -1.5]))
