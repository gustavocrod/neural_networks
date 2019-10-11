# -*- coding: utf-8 -*-
import numpy as np

from ..artificial_neuron import ArtificialNeuron

MAXITER = 99999


class Perceptron(ArtificialNeuron):
    """
    Algoritmo perceptron:
    <1> Obter o conjunto de amostras de treinamento {x(k)};
    <2> Associar a saida desejada {d{k}};
    <3> Iniciar o vetor w com valores aleatorios pequenos;
    <4> Especificar a taxa de aprendizagem {eta};
    <5> Iniciar o contador denumero de epocas {epoca <- 0};
    <6> Repetir as instrucoes:
        <6.1> erro <- inexistente;
        <6.2> Para todas as amostras de treinamento {x(k), d(k)}, fazer:
            <6.2.1> u <- w^T . x(k);
            <6.2.2> y <- sinal(u);
            <6.2.3> Se y != d(k):
                <6.2.3.1> Entao { w <- w + eta . (d(k) - y) . x(k);
                                { erro <- existe;
        <6.3> epoca <- epoca + 1;
        ate que: erro <- inexistente
    Fim {Algoritmo perceptron}

    """

    def __init__(self, meta, previsores):
        super().__init__(meta, previsores)

    def ActivationFunction(self, sum):
        """
        implementa a ActivationFunction da perceptron - stepFuncion - Degrau Bipolar

        w' . x' >= limiar de ativacao, onde w' eh o vetor de pesos e x' eh o vetor de entrada

        """
        if sum >= 0:
            return 1
        if sum < 0:
            return -1

    def Train(self):
        """
               funcao de treino da perceptron de uma camada
               considerando o par de treinamento (x, d) { x = entrada
                                                        { d - saida esperada
                   > saida da rede sera y
                   > Erro: e = d - y
                   > y pertence {-1, 1} e d pertence {-1, 1}
                   > e != 0: d=1 e y = -1  OU   d = -1 e y = 1
                   > equacao dos pesos = w(t+1) = w(t) + nex(t) ==> nex: eta * error * entrada(t)
               :return:
               """
        totalError = 1
        epoch = 0
        while ((totalError != 0.0) and epoch < MAXITER):
            totalError = 0  # reset no erro
            for i in range(len(self.target)):
                reg = np.array(self.previsores[i])  # pega uma linha da matriz de previsores
                output = self.ActivationFunction(self.Sum(reg))  # saida calculada (prediccao)
                # e = (y * (1-y) * (d-y))
                if (output != self.target[i]):
                    error = self.target[i] - output  # erro = (saidaEsperada - saidaCalculada) ===== d - u
                    totalError += abs(error)
                    self.UpdateWeigth(error, reg)

            epoch += 1
        # os.system('clear')
        print('--')
        self.ShowWeights()
        print("Treinado com " + str(epoch) + " epocas!")
