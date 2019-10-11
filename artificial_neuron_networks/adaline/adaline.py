import os

import numpy as np

from ..artificial_neuron import ArtificialNeuron

MAXITER = 2000


class Adaline(ArtificialNeuron):
    """
       Classe Adaline de uma camada
       > adaline possui saidas bipolares
       > seus pesos sao adaptados em funcao de uma saida linear:
            y = somatorio w[i]x[i] antes da aplicacao da funcao de ativacao

       > tenta minimizar o erro das saidas em relacao aos valores desejados d[i] do conjunto de treinamento
    """

    def __init__(self, meta, previsores):
        super().__init__(meta, previsores)

    def ActivationFunction(self, sum):

        y = np.tanh(sum * 10)
        return y

    def Train(self):
        """
            seja um par de treinamento (x, yd) ===> y desejado
            erro quadratico = e² = (yd = w.x)²

            > devemos encontrar o w que leve a menor e²
            > logo desejamos encontrar o mínimo da superfície correspondente à soma das superfícies de erro de todos os
                dados de treinamento
            > funcao de erro(custo):
                    j = 1/n somatorio de i = 1:n(y desejado(i) - w.x(i))²

            - Pelo gradiente da funcao de custo no ponto
                > gradiente possui a mesma direcao da maior variacao de erro, o ajuste deve ser na direcao contraria
                > deltaW(t)eta-deltaJ
                > gradiente = deltaw(t) * eta * e * x(i) => deltaW(t) = nex { para n = eta

            - equacao de ajuste: (ja explicado)
                > w(t+1) = w(t) + eta * e * x(t)

            - erro quadratico:
                > E(W) = 1/n somatorio de k=1 ate p (d[k] - (wT.x[k] - bias))²
        """
        epoch = 0
        MSE = 0.0  # atual
        while (epoch < MAXITER):
            os.system('clear')
            LMSE = MSE  # last mse
            sqError = 0.0  # erro quadratico total
            for i in range(len(self.target)):
                reg = np.array(
                    self.previsores[i])  # registro da iteracao a ser calculado - pega uma linha da matriz de entrada
                sum = self.Sum(reg)
                error = self.target[i] - sum  # erro = (saidaEsperada - saidaCalculada) ===== d - u
                sqError += np.square(error)
                self.UpdateWeigth(error, reg)

            MSE = sqError / 4  # atual
            epoch += 1

            if ((abs(MSE - LMSE)) <= self.precisionTax):  # o quao errado esta, para simular um do-while
                break

        print('--')
        self.ShowWeights()
        print("Treinado com " + str(epoch) + " epocas!")
