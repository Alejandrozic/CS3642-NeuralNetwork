import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from neural_network.neural_network import ANN
from neural_network.create_combinations import create_combinations
from neural_network.constants import WHITE_INT, BLACK_INT


class NeuralNetworkApp(tk.Canvas):
    # -- Instantiate tkinter object -- #
    window = tk.Tk()

    # -- Title -- #
    TITLE = 'Neural Network'

    # -- Size of Puzzle -- #
    COLUMNS = 8
    ROWS = 12

    # -- Constants related to PUZZLE CANVAS -- #
    TILE_WIDTH = 90
    TILE_HEIGHT = 90
    TILE_BORDER = 3
    HEIGHT = TILE_HEIGHT * COLUMNS
    WIDTH = TILE_WIDTH * ROWS
    text_output = None

    i1_color = None
    i2_color = None
    i3_color = None
    i4_color = None
    i1_text = None
    i2_text = None
    i3_text = None
    i4_text = None
    i5_text = None
    h1_text = None
    h2_text = None
    o1_text = None
    o2_text = None
    i1toh1_text = None
    i2toh1_text = None
    i3toh1_text = None
    i4toh1_text = None
    i5toh1_text = None
    i1toh2_text = None
    i2toh2_text = None
    i3toh2_text = None
    i4toh2_text = None
    i5toh2_text = None
    h1too1_text = None
    h2too1_text = None
    h1too2_text = None
    h2too2_text = None

    neural_network = ANN()
    combinations = create_combinations()

    def __init__(self):
        # -- Add Title -- #
        self.window.title(self.TITLE)

        # -- Initialize Options Frame -- #
        self.init_options_frame()

        # -- Initialize Neural Network Drawing -- #
        self.init_neural_network_frame()

        # -- Initialize Text Output Frame -- #
        self.init_text_output_frame()

        # -- Initialize NN -- #
        self.update_nn(self.neural_network.to_dict())

        # -- Start Window -- #
        self.window.mainloop()

    def init_options_frame(self):
        """     These are configuration options      """
        options_frame_label = tk.LabelFrame(self.window, text="Options")
        options_frame_label.pack(side='top', anchor=tk.W, expand="yes")
        options_frame = tk.Frame(options_frame_label, width=200, height=400)
        options_frame.pack(fill='both', padx=10, pady=5, expand=True)

        new_btn = tk.Button(options_frame, text="RESET Neural Network", command=self.reset_nn, bg="cornsilk3")
        new_btn.pack(side='top')

        new_btn = tk.Button(options_frame, text="Train Neural Network", command=self.train_nn, bg="cornsilk3")
        new_btn.pack(side='top')

        reset_btn = tk.Button(options_frame, text="Test Neural Network", command=self.test_nn, bg="cornsilk3")
        reset_btn.pack(side='bottom')

    def init_neural_network_frame(self):
        frame_label = tk.LabelFrame(self.window, text="Neural Network")
        frame_label.pack(side='left', fill="both", expand="yes")
        frame = tk.Frame(frame_label)
        frame.pack(side='left', fill='both', padx=10, pady=5, expand=True)
        tk.Canvas.__init__(self, frame, bg="powder blue", height=self.HEIGHT, width=self.WIDTH)
        x = 0
        y = 0
        self.i1_color = self.create_oval(*self.__find_coordinates__(x, y), fill='khaki', tags="i1")
        self.i1_text = self.create_text(self.find_center_circle(*self.coords("i1")), text='')
        self.i2_color = self.create_oval(*self.__find_coordinates__(x, y + 1.7), fill='khaki', tags="i2")
        self.i2_text = self.create_text(self.find_center_circle(*self.coords("i2")), text='')
        self.i3_color = self.create_oval(*self.__find_coordinates__(x, y + 3.4), fill='khaki', tags="i3")
        self.i3_text = self.create_text(self.find_center_circle(*self.coords("i3")), text='')
        self.i4_color = self.create_oval(*self.__find_coordinates__(x, y + 5.1), fill='khaki', tags="i4")
        self.i4_text = self.create_text(self.find_center_circle(*self.coords("i4")), text='')
        self.create_oval(*self.__find_coordinates__(x, y + 6.8), fill='khaki', tags="i5")
        self.i5_text = self.create_text(self.find_center_circle(*self.coords("i5")), text='')

        self.create_oval(*self.__find_coordinates__(x + 4, y + 2.2), fill='dark khaki', tags="h1")
        self.h1_text = self.create_text(self.find_center_circle(*self.coords("h1")), text='')
        self.create_oval(*self.__find_coordinates__(x + 4, y + 5.0), fill='dark khaki', tags="h2")
        self.h2_text = self.create_text(self.find_center_circle(*self.coords("h2")), text='')

        self.create_oval(*self.__find_coordinates__(x + 8, y + 2.4), fill='white', tags="o1")
        self.o1_text = self.create_text(self.find_center_circle(*self.coords("o1")), text='')
        self.create_oval(*self.__find_coordinates__(x + 8, y + 4.8), fill='MistyRose4', tags="o2")
        self.o2_text = self.create_text(self.find_center_circle(*self.coords("o2")), text='')

        # def aa_line(event):
        #     self.create_line(0, 0, event.x, event.y)
        #     print(event.x, event.y)
        # def bind_line():
        #     self.window.bind("<Button-1>", aa_line)
        # button_line = tk.Button(text="line", command=bind_line).pack()

        self.create_line(87, 47, 364, 241, fill="black", tags="i1 to h1")
        self.i5toh1_text = self.create_text(90, 605, angle=50, text='')
        self.create_line(87, 204, 364, 241, fill="black", tags="i2 to h1")
        self.i4toh1_text = self.create_text(105, 479, angle=45, text='')
        self.create_line(87, 356, 364, 241, fill="black", tags="i3 to h1")
        self.i3toh1_text = self.create_text(110, 335, angle=30, text='')
        self.create_line(87, 507, 364, 241, fill="black", tags="i4 to h1")
        self.i2toh1_text = self.create_text(115, 195, angle=-9, text='')
        self.create_line(87, 658, 364, 241, fill="black", tags="i5 to h1")
        self.i1toh1_text = self.create_text(115, 48, angle=-20, text='')

        self.create_line(87, 47, 362, 498, fill="black")
        self.i5toh2_text = self.create_text(115, 671, angle=25, text='')
        self.create_line(87, 204, 362, 498, fill="black")
        self.i4toh2_text = self.create_text(115, 524, angle=0, text='')
        self.create_line(87, 356, 362, 498, fill="black")
        self.i3toh2_text = self.create_text(115, 380, angle=-30, text='')
        self.create_line(87, 507, 362, 498, fill="black")
        self.i2toh2_text = self.create_text(100, 248, angle=-45, text='')
        self.create_line(87, 658, 362, 498, fill="black")
        self.i1toh2_text = self.create_text(100, 85, angle=-60, text='')

        self.create_line(446, 248, 723, 260, fill="black")
        self.h1too1_text = self.create_text(657, 242, angle=-5, text='')
        self.create_line(446, 248, 723, 480,  fill="black")
        self.h2too1_text = self.create_text(637, 311, angle=45, text='')

        self.create_line(446, 500, 723, 260, fill="black")
        self.h1too2_text = self.create_text(675, 420, angle=-45, text='')
        self.create_line(446, 500, 723, 480, fill="black")
        self.h2too2_text = self.create_text(638, 501, angle=5, text='')

        self.pack()

    @staticmethod
    def find_center_circle(x1, y1, x2, y2) -> tuple:
        return (x1 + x2)/2, (y1+y2)/2

    def init_text_output_frame(self):
        frame_label = tk.LabelFrame(self.window, text="Output")
        frame_label.pack(side='right', fill="both", expand="yes")
        frame = tk.Frame(frame_label)
        frame.pack(side='top', fill='both', padx=10, pady=5, expand=True)
        self.text_output = tk.scrolledtext.ScrolledText(frame, width=30, height=30)
        self.text_output.pack(side='right')

    def reset_nn(self):
        self.neural_network = ANN()
        self.combinations = create_combinations()
        self.update_nn(self.neural_network.to_dict())

    def train_nn(self):
        result = dict()
        all_success = False
        iteration = 0
        training_set = self.combinations[3:]
        for i in training_set:
            self.text_output.insert(tk.INSERT, f'{i}\n')
        while all_success is False:
            iteration += 1
            for i in training_set:
                expected_answer = WHITE_INT if i.count(WHITE_INT) >= 2 else BLACK_INT
                _bool = self.neural_network.train(*i, expected_answer=expected_answer)
                result[i] = _bool

            if all(a for i, a in result.items()):
                all_success = True

            if iteration % 100 == 0:
                self.text_output.insert(tk.INSERT, f'Epoch(s) {iteration}.\n')
                self.text_output.insert(tk.INSERT, f'Correctly Guessed: {list(a for i, a in result.items()).count(True)}/{len(result)}\n')
                self.update_nn(self.neural_network.to_dict())
                self.__waiting__(1)

        self.text_output.insert(tk.INSERT, f'Epoch(s) {iteration}.\n')
        self.text_output.insert(tk.INSERT, f'Correctly Guessed: {list(a for i, a in result.items()).count(True)}/{len(result)}\n')
        self.update_nn(self.neural_network.to_dict())

    def update_nn(self, state: dict):
        self.itemconfig(self.i1_text, text=state['i1_text'])
        self.itemconfig(self.i2_text, text=state['i2_text'])
        self.itemconfig(self.i3_text, text=state['i3_text'])
        self.itemconfig(self.i4_text, text=state['i4_text'])
        self.itemconfig(self.i5_text, text=state['i5_text'])
        self.itemconfig(self.h1_text, text=state['h1_text'])
        self.itemconfig(self.h2_text, text=state['h2_text'])
        self.itemconfig(self.o1_text, text=state['o1_text'])
        self.itemconfig(self.o2_text, text=state['o2_text'])

        self.itemconfig(self.i1toh1_text, text=state['i1toh1_text'])
        self.itemconfig(self.i2toh1_text, text=state['i2toh1_text'])
        self.itemconfig(self.i3toh1_text, text=state['i3toh1_text'])
        self.itemconfig(self.i4toh1_text, text=state['i4toh1_text'])
        self.itemconfig(self.i5toh1_text, text=state['i5toh1_text'])
        self.itemconfig(self.i1toh2_text, text=state['i1toh2_text'])
        self.itemconfig(self.i2toh2_text, text=state['i2toh2_text'])
        self.itemconfig(self.i3toh2_text, text=state['i3toh2_text'])
        self.itemconfig(self.i4toh2_text, text=state['i4toh2_text'])
        self.itemconfig(self.i5toh2_text, text=state['i5toh2_text'])

        self.itemconfig(self.h1too1_text, text=state['h1too1_text'])
        self.itemconfig(self.h2too1_text, text=state['h2too1_text'])
        self.itemconfig(self.h1too2_text, text=state['h1too2_text'])
        self.itemconfig(self.h2too2_text, text=state['h2too2_text'])

    def test_nn(self):
        test_set = self.combinations[:3]
        for i in test_set:
            # -- Set Values and color effects -- #
            self.itemconfig(self.i1_text, text=i[0])
            self.itemconfig(self.i1_color, fill='indian red',)
            self.__waiting__(1)
            self.itemconfig(self.i1_color, fill='khaki', )

            self.itemconfig(self.i2_text, text=i[1])
            self.itemconfig(self.i2_color, fill='indian red', )
            self.__waiting__(1)
            self.itemconfig(self.i2_color, fill='khaki', )

            self.itemconfig(self.i3_text, text=i[2])
            self.itemconfig(self.i3_color, fill='indian red', )
            self.__waiting__(1)
            self.itemconfig(self.i3_color, fill='khaki', )

            self.itemconfig(self.i4_text, text=i[3])
            self.itemconfig(self.i4_color, fill='indian red', )
            self.__waiting__(1)
            self.itemconfig(self.i4_color, fill='khaki', )

            self.__waiting__(3)
            answer = self.neural_network.test(*i)
            expected_answer = WHITE_INT if i.count(WHITE_INT) >= 2 else BLACK_INT
            color = 'green' if expected_answer == answer else 'red'
            if answer == WHITE_INT:
                line = self.create_line(947, 257, 849, 257, fill=color, arrow=tk.LAST, width=3)
            else:
                line = self.create_line(947, 480, 849, 480, fill=color, arrow=tk.LAST, width=3)
            self.__waiting__(10)
            self.delete(line)
            self.__waiting__(1)

    def __find_coordinates__(self, x, y) -> list:

        """     Takes one dimensional pointer and converts it to two dimensional   """

        return [
            (x * self.TILE_WIDTH) + self.TILE_BORDER,  # x1
            (y * self.TILE_HEIGHT) + self.TILE_BORDER,  # y1
            ((x + 1) * self.TILE_WIDTH) - self.TILE_BORDER,  # x2
            ((y + 1) * self.TILE_HEIGHT) - self.TILE_BORDER,  # y2
        ]

    def __waiting__(self, seconds: int):

        """     Sleep Function for Tkiner   """

        var = tk.IntVar()
        self.window.after(seconds * 1000, var.set, 1)
        self.window.wait_variable(var)
