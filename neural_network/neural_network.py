import random
import math
from neural_network.constants import WHITE_INT, BLACK_INT


def sigmoid(n: float) -> float:
    """     Returns values 0 - 1 Squashing function"""
    return 1 / (1 + math.exp(-n))


class InputNeuron:
    __slots__ = {'input_'}

    def __init__(self, input_: int = None):
        self.input_ = input_


class HiddenNeuron:
    __slots__ = {'summation', 'output'}

    def __init__(self):
        self.summation = None     # Summation of Weight times values + bias
        self.output = 0       # Output of Activation Func

    def activation_function(self):
        """     Activation / Transfer / Squashing Function   """
        self.output = sigmoid(n=self.summation)


class OutputNeuron:
    __slots__ = {'summation', 'output', 'category'}

    def __init__(self, category: int):
        self.summation = None     # Summation of Weight times values + bias
        self.output = 0       # Output of Activation Func
        self.category = category  # 0 or 1

    def activation_function(self):
        """     Activation / Transfer / Squashing Function   """
        self.output = sigmoid(n=self.summation)


class NeuronUnDirectedGraph:

    graph = dict()

    def add_neuron(self, neuron):
        self.graph[neuron] = dict()

    def add_weight(self, src_neuron, dst_neuron, weight: float):
        self.graph[src_neuron][dst_neuron] = weight
        self.graph[dst_neuron][src_neuron] = weight

    def update_weight(self, src_neuron, dst_neuron, weight: float):
        self.graph[src_neuron][dst_neuron] = weight
        self.graph[dst_neuron][src_neuron] = weight

    def get_weight(self, src_neuron, dst_neuron):
        return self.graph[src_neuron][dst_neuron]


class ANN:

    """
        Architecture:
            5-2-2
    """

    graph = NeuronUnDirectedGraph()
    LEARN_RATE = 0.1

    def __init__(self):
        # -- Input Layer -- #
        self.neuron_input_a = InputNeuron()
        self.neuron_input_b = InputNeuron()
        self.neuron_input_c = InputNeuron()
        self.neuron_input_d = InputNeuron()
        self.neuron_input_bias = InputNeuron(input_=1)

        # -- Hidden Layer -- #
        self.neuron_hidden_a = HiddenNeuron()
        self.neuron_hidden_b = HiddenNeuron()

        # -- Output Layer -- #
        self.neuron_output_a = OutputNeuron(category=WHITE_INT)
        self.neuron_output_b = OutputNeuron(category=BLACK_INT)

        # -- Add Neurons to Graph -- #
        for neuron in [
            self.neuron_input_a, self.neuron_input_b, self.neuron_input_c, self.neuron_input_d, self.neuron_input_bias,
            self.neuron_hidden_a, self.neuron_hidden_b,
            self.neuron_output_a, self.neuron_output_b,
        ]:
            self.graph.add_neuron(neuron)

        # -- Input to Hidden Layer Connections -- #
        self.graph.add_weight(self.neuron_input_a, self.neuron_hidden_a, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_a, self.neuron_hidden_b, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_b, self.neuron_hidden_a, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_b, self.neuron_hidden_b, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_c, self.neuron_hidden_a, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_c, self.neuron_hidden_b, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_d, self.neuron_hidden_a, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_d, self.neuron_hidden_b, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_bias, self.neuron_hidden_a, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_input_bias, self.neuron_hidden_b, weight=self.generate_random_weight())

        # -- Hidden to Output Layer Connections -- #
        self.graph.add_weight(self.neuron_hidden_a, self.neuron_output_a, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_hidden_a, self.neuron_output_b, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_hidden_b, self.neuron_output_a, weight=self.generate_random_weight())
        self.graph.add_weight(self.neuron_hidden_b, self.neuron_output_b, weight=self.generate_random_weight())

    def train(self, a, b, c, d, expected_answer) -> bool:
        """
            Takes a single set of inputs, performs a
            feedforward to obtain the guess, does a back
            propagation to train the NN, and finally
            return boolean indicating where the guess
            matched the expected answer.
        """
        self.neuron_input_a.input_ = a
        self.neuron_input_b.input_ = b
        self.neuron_input_c.input_ = c
        self.neuron_input_d.input_ = d
        solution_neuron = self.feed_forward()
        self.back_propagation(expected_answer)
        return solution_neuron.category == expected_answer

    def test(self, a, b, c, d) -> int:
        """

        """
        self.neuron_input_a.input_ = a
        self.neuron_input_b.input_ = b
        self.neuron_input_c.input_ = c
        self.neuron_input_d.input_ = d
        solution_neuron = self.feed_forward()
        return solution_neuron.category

    def feed_forward(self):

        # -------------------------------- #
        # -- HIDDEN Layer Calculations  -- #
        # -------------------------------- #

        self.neuron_hidden_a.summation = sum([
            (self.graph.get_weight(self.neuron_hidden_a, self.neuron_input_a) * self.neuron_input_a.input_),
            (self.graph.get_weight(self.neuron_hidden_a, self.neuron_input_b) * self.neuron_input_b.input_),
            (self.graph.get_weight(self.neuron_hidden_a, self.neuron_input_c) * self.neuron_input_c.input_),
            (self.graph.get_weight(self.neuron_hidden_a, self.neuron_input_d) * self.neuron_input_d.input_),
            (self.graph.get_weight(self.neuron_hidden_a, self.neuron_input_bias) * self.neuron_input_bias.input_),
        ])
        self.neuron_hidden_a.activation_function()

        self.neuron_hidden_b.summation = sum([
            (self.graph.get_weight(self.neuron_hidden_b, self.neuron_input_a) * self.neuron_input_a.input_),
            (self.graph.get_weight(self.neuron_hidden_b, self.neuron_input_b) * self.neuron_input_b.input_),
            (self.graph.get_weight(self.neuron_hidden_b, self.neuron_input_c) * self.neuron_input_c.input_),
            (self.graph.get_weight(self.neuron_hidden_b, self.neuron_input_d) * self.neuron_input_d.input_),
            (self.graph.get_weight(self.neuron_hidden_b, self.neuron_input_bias) * self.neuron_input_bias.input_),
        ])
        self.neuron_hidden_b.activation_function()

        # -------------------------------- #
        # -- OUTPUT Layer Calculations  -- #
        # -------------------------------- #

        self.neuron_output_a.summation = sum([
            (self.graph.get_weight(self.neuron_output_a, self.neuron_hidden_a) * self.neuron_hidden_a.output),
            (self.graph.get_weight(self.neuron_output_a, self.neuron_hidden_b) * self.neuron_hidden_b.output),
        ])
        self.neuron_output_a.activation_function()

        self.neuron_output_b.summation = sum([
            (self.graph.get_weight(self.neuron_output_b, self.neuron_hidden_a) * self.neuron_hidden_a.output),
            (self.graph.get_weight(self.neuron_output_b, self.neuron_hidden_b) * self.neuron_hidden_b.output),
        ])
        self.neuron_output_b.activation_function()

        # -------------- #
        # -- Decision -- #
        # -------------- #

        if self.neuron_output_a.output >= self.neuron_output_b.output:
            return self.neuron_output_a
        else:
            return self.neuron_output_b

    def back_propagation(self, expected_answer: int):

        # --------------------------------------- #
        # -- Determine categorization which is -- #
        # -- dependent on expected answer      -- #
        # --------------------------------------- #

        if expected_answer == WHITE_INT:
            t_1_e = 1
            t_2_e = 0
        else:
            t_1_e = 0
            t_2_e = 1

        # --------------------------------------- #
        # -- Calculate Error Values for Output -- #
        # --------------------------------------- #

        o_1_e = self.neuron_output_a.output
        s_o_1 = o_1_e * (1 - o_1_e) * (t_1_e - o_1_e)

        o_2_e = self.neuron_output_b.output
        s_o_2 = o_2_e * (1 - o_2_e) * (t_2_e - o_2_e)

        # --------------------------------------- #
        # -- Pull Values from activation function
        # -- on Hidden 1 and Hidden 2.
        # --------------------------------------- #

        h_1_e = self.neuron_hidden_a.output
        h_2_e = self.neuron_hidden_b.output

        # --------------------------------------- #
        # -- Calculate Error Values for Hidden -- #
        # --------------------------------------- #

        s_h_1 = (
            (
                (self.graph.get_weight(self.neuron_hidden_a, self.neuron_output_a) * s_o_1) +
                (self.graph.get_weight(self.neuron_hidden_a, self.neuron_output_b) * s_o_2)
            ) * h_1_e * (1 - h_1_e)
        )

        s_h_2 = (
            (
                (self.graph.get_weight(self.neuron_hidden_b, self.neuron_output_a) * s_o_1) +
                (self.graph.get_weight(self.neuron_hidden_b, self.neuron_output_b) * s_o_2)
            ) * h_2_e * (1 - h_2_e)
        )

        # ---------------------------------- #
        # -- Update Input <> Hidden units -- #
        # ---------------------------------- #

        self.adjust_weights_input_hidden(self.neuron_input_a, self.neuron_hidden_a, s_h_k=s_h_1)
        self.adjust_weights_input_hidden(self.neuron_input_b, self.neuron_hidden_a, s_h_k=s_h_1)
        self.adjust_weights_input_hidden(self.neuron_input_c, self.neuron_hidden_a, s_h_k=s_h_1)
        self.adjust_weights_input_hidden(self.neuron_input_d, self.neuron_hidden_a, s_h_k=s_h_1)
        self.adjust_weights_input_hidden(self.neuron_input_bias, self.neuron_hidden_a, s_h_k=s_h_1)
        self.adjust_weights_input_hidden(self.neuron_input_a, self.neuron_hidden_b, s_h_k=s_h_2)
        self.adjust_weights_input_hidden(self.neuron_input_b, self.neuron_hidden_b, s_h_k=s_h_2)
        self.adjust_weights_input_hidden(self.neuron_input_c, self.neuron_hidden_b, s_h_k=s_h_2)
        self.adjust_weights_input_hidden(self.neuron_input_d, self.neuron_hidden_b, s_h_k=s_h_2)
        self.adjust_weights_input_hidden(self.neuron_input_bias, self.neuron_hidden_b, s_h_k=s_h_2)

        # ----------------------------------- #
        # -- Update Hidden <> Output units -- #
        # ----------------------------------- #

        self.adjust_weights_hidden_output(self.neuron_hidden_a, self.neuron_output_a, s_o_k=s_o_1, h_k_e=h_1_e)
        self.adjust_weights_hidden_output(self.neuron_hidden_a, self.neuron_output_b, s_o_k=s_o_2, h_k_e=h_1_e)
        self.adjust_weights_hidden_output(self.neuron_hidden_b, self.neuron_output_a, s_o_k=s_o_1, h_k_e=h_2_e)
        self.adjust_weights_hidden_output(self.neuron_hidden_b, self.neuron_output_b, s_o_k=s_o_2, h_k_e=h_2_e)

    def adjust_weights_input_hidden(self, src, dst, s_h_k):
        """   Weight Calculations for Input <> Hidden   """
        w = self.graph.get_weight(src, dst)
        w_delta = self.LEARN_RATE * s_h_k * src.input_
        self.graph.update_weight(src, dst, weight=w + w_delta)

    def adjust_weights_hidden_output(self, src, dst, s_o_k, h_k_e):
        """   Weight Calculations for Hidden <> Output   """
        w = self.graph.get_weight(src, dst)
        w_delta = self.LEARN_RATE * s_o_k * h_k_e
        self.graph.update_weight(src, dst, weight=w + w_delta)

    @staticmethod
    def generate_random_weight() -> float:
        """     Professor recommendation for randomly assigned weights  """
        return round(
            random.uniform(-0.5, 0.5),
            ndigits=2,
        )

    def to_dict(self) -> dict:
        """
            Output function to a Application/User Interface.
        """
        ndigits = 4
        d = {
            'i1_text': self.neuron_input_a.input_,
            'i2_text': self.neuron_input_b.input_,
            'i3_text': self.neuron_input_c.input_,
            'i4_text': self.neuron_input_d.input_,
            'i5_text': f'bias {self.neuron_input_bias.input_}',
            'h1_text': round(self.neuron_hidden_a.output, ndigits=ndigits),
            'h2_text': round(self.neuron_hidden_b.output, ndigits=ndigits),
            'o1_text': round(self.neuron_output_a.output, ndigits=ndigits),
            'o2_text': round(self.neuron_output_b.output, ndigits=ndigits),
            'i1toh1_text': round(self.graph.get_weight(self.neuron_input_a, self.neuron_hidden_a), ndigits=ndigits),
            'i2toh1_text': round(self.graph.get_weight(self.neuron_input_b, self.neuron_hidden_a), ndigits=ndigits),
            'i3toh1_text': round(self.graph.get_weight(self.neuron_input_c, self.neuron_hidden_a), ndigits=ndigits),
            'i4toh1_text': round(self.graph.get_weight(self.neuron_input_d, self.neuron_hidden_a), ndigits=ndigits),
            'i5toh1_text': round(self.graph.get_weight(self.neuron_input_bias, self.neuron_hidden_a), ndigits=ndigits),
            'i1toh2_text': round(self.graph.get_weight(self.neuron_input_a, self.neuron_hidden_b), ndigits=ndigits),
            'i2toh2_text': round(self.graph.get_weight(self.neuron_input_b, self.neuron_hidden_b), ndigits=ndigits),
            'i3toh2_text': round(self.graph.get_weight(self.neuron_input_c, self.neuron_hidden_b), ndigits=ndigits),
            'i4toh2_text': round(self.graph.get_weight(self.neuron_input_d, self.neuron_hidden_b), ndigits=ndigits),
            'i5toh2_text': round(self.graph.get_weight(self.neuron_input_bias, self.neuron_hidden_b), ndigits=ndigits),
            'h1too1_text': round(self.graph.get_weight(self.neuron_hidden_a, self.neuron_output_a), ndigits=ndigits),
            'h2too1_text': round(self.graph.get_weight(self.neuron_hidden_b, self.neuron_output_a), ndigits=ndigits),
            'h1too2_text': round(self.graph.get_weight(self.neuron_hidden_a, self.neuron_output_b), ndigits=ndigits),
            'h2too2_text': round(self.graph.get_weight(self.neuron_hidden_b, self.neuron_output_b), ndigits=ndigits),
        }
        return d
