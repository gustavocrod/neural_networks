import abc
import numpy as np
import math


class ArtificialNeuron(object):
    __metaclass__ = abc.ABCMeta

    """
    Classe abstrata
        Adaline:
            equacao de ajuste obtida para a saida linear

        Perceptron: 
            equacao de ajuste obtida para a saida do nodo apos a aplicacao da funcao de ativacao
    """

    def __init__(self, meta, previsores):
        self.weights = np.array([2 * np.random.random() - 1, 2 * np.random.random() - 1,
                                 2 * np.random.random() - 1, 2 * np.random.random() - 1])
        self.target = np.array(meta)  # saida esperada
        self.learnTax = 0.0025  # taxa de aprendizado - eta
        self.precisionTax = 0.000001  # taxa de precisao
        self.previsores = np.array(previsores)  # vetores de entradas com x[0] = bias

    @abc.abstractmethod
    def ActivationFunction(self, sum):
        """
        perceptron -> f(x){ 1 se w . x + b >= 0
                         { -1 senao

        adaline -> regra delta para gradiente descendente - minimos quadrados
                -> tanh(x)

        :param sum: recebe o valor do somatorio - saida calculada
        :return: retorna o valor de ativacao
        """
        pass

    @abc.abstractmethod
    def Train(self):
        pass

    def ShowWeights(self):
        """
        mostra o vetor de pesos atual
        :return:
        """
        print("Pesos: ", self.weights)

    def UpdateWeigth(self, error, previsores):
        """
        atualiza os n pesos sinapticos  do neuronio (weights)

        1. peso(k+1) = peso(k) + (taxaDeAprendizagem * erro * input[i])) ==== perceptron
        2. peso(k+1) = peso(k) + (taxaDeAprendizagem * (valorDesejado - sum) * input[i] ) ==== adaline

        """
        for i in range(len(self.weights)):
            self.weights[i] += self.learnTax * error * previsores[i]

    def Sum(self, reg):
        """
        representa o somatorio da formula
        :param reg: um registro da matriz, e.g (1, -1)
        :return: retorna o somatorio de p(i) * w(i) + bias
        """
        return reg.dot(self.weights)  # produto escalar

