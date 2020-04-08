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

    # 신경망 학습
    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원로 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 은닉계층으로 들어오는 값 계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 은닉계층으로 나가는 값 계산
        hidden_outputs = self.sigmoid(hidden_inputs)

        # 최종계층으로 들어오는 값 계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 최종계층으로 나가는 값 계산
        final_outputs = self.activation_function(final_inputs)

        # 오차는 (실제 값 - 계산 값)
        output_errors = targets - final_outputs

        # 은닉계층의 오차는 가중치에 의해 나뉜 출력 계층의
        # 오차들의 재조합 
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 은닉계층과 출력계층 간 가중치 업데이트
        self.who += self.lr * numpy.dot(
                    (output_errors * final_outputs * (1.0 - final_outputs)),
                    numpy.transpose(hidden_outputs))
        
        # 입력계층과 은닉계층 간 가중치 업데이트
        self.wih += self.lr * numpy.dot(
                    (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                    numpy.transpose(inputs))

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
