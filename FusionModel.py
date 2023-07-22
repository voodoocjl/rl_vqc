import pennylane as qml
import torch
import torch.nn as nn
from math import pi 


class QuantumLayer(torch.nn.Module):

    def __init__(self, arguments, design):
        super(QuantumLayer, self).__init__()
        self.args = arguments
        self.design = design
        self.params = torch.nn.Parameter(torch.randn(self.args["num_layers"]))
    
    def forward(self, input):
        # q_out = torch.Tensor(0, self.args["num_qubits"])
        q_out = torch.Tensor(0)

        # q_out = q_out.to(self.args.device)
        for elem in input:
            q_out_elem = self.vqc(elem, self.params,
                                 design=self.design).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return q_out    
    
    def vqc(self, cla_input, q_weight, **kwargs):
        n_qubits = self.args["num_qubits"]
        n_layers = self.args["num_layers"]

        qml.disable_return()
        dev = qml.device("lightning.qubit", wires=n_qubits)
        @qml.qnode(dev, interface="torch", diff_method="adjoint")    

        def circuit():
            current_design = kwargs['design']
            q_input_features = cla_input.reshape(n_qubits, 3)
            q_weight_features = q_weight
            
            for i in range(n_qubits):  # data-reuploading
                qml.Rot(*q_input_features[i], wires=i)

            # qml.broadcast(unitary=qml.CNOT, pattern="ring", wires=[i for i in range(n_qubits)])
                
            for i in range(n_layers):        
                angle = q_weight_features[i]
                if current_design[0][i].item() != n_qubits:
                    qml.IsingZZ(angle,
                                wires=[
                                    int(current_design[0][i].item()),
                                    int(current_design[1][i].item())
                                ])  #[control，target]

                elif current_design[2][i].item() != n_qubits:  # rotation gates
                    axis = current_design[3][i].item()
                    if axis == 'X' or axis == 'x' or axis == 1:
                        qml.RX(angle, wires=int(current_design[2][i].item()))
                    elif axis == 'Y' or axis == 'y' or axis == 2:
                        qml.RY(angle, wires=int(current_design[2][i].item()))
                    elif axis == 'Z' or axis == 'z' or axis == 3:
                        j = int(current_design[2][i].item())
                        qml.Rot(*q_input_features[j], wires=j)
                    else:
                        print("Wrong gate")            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            # return qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))

        return circuit()    


class QNet(torch.nn.Module):

    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        self.ClassicalLayer_a = nn.RNN(self.args["a_insize"], self.args["a_hidsize"])
        self.ClassicalLayer_v = nn.RNN(self.args["v_insize"], self.args["v_hidsize"])        
        self.ClassicalLayer_t = nn.RNN(self.args["t_insize"], self.args["t_hidsize"])        
        self.QuantumLayer = QuantumLayer(self.args, self.design)
        self.Regressor = nn.Linear(self.args["num_qubits"], 1)

    def forward(self, x_a, x_v, x_t):
        x_a = torch.permute(x_a, (1, 0, 2))
        x_v = torch.permute(x_v, (1, 0, 2))
        x_t = torch.permute(x_t, (1, 0, 2))
        a_h = self.ClassicalLayer_a(x_a)[0][-1]
        v_h = self.ClassicalLayer_v(x_v)[0][-1]
        t_h = self.ClassicalLayer_t(x_t)[0][-1]
        a_o = (a_h + 1)/2  * pi 
        v_o = (v_h + 1)/2  * pi
        t_o = (t_h + 1)/2  * pi
        x_p = torch.cat((a_o, v_o, t_o), 1)
        exp_val = self.QuantumLayer(x_p)
        # output = exp_val * 3
        output = torch.tanh(self.Regressor(exp_val).squeeze(dim=1)) * 3

        return output
