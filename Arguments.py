class Arguments:

    def __init__(self):
        self.n_qubits = 4
        self.layer_repo = 5
        self.a_insize = 74
        self.v_insize = 35
        self.t_insize = 300
        self.a_hidsize = 3  #6
        self.v_hidsize = 3  #3
        self.t_hidsize = 6  #12
        self.device = 'cpu'
        self.clr = 0.005
        self.qlr = 0.05
        self.epochs = 1
        self.batch_size = 32
        self.num_layers = 400
