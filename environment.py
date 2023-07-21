import torch
from utils import *
from sys import stdout
from itertools import product
import numpy as np
import copy
import curricula
import pennylane as qml
from Arguments import Arguments
import torch.optim as optim
from schemes import Scheme, evaluate
from FusionModel import QNet
from datasets import MOSIDataLoaders


# args = Arguments()


class CircuitEnv():

    def __init__(self, conf, device):

        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']

        self.ham = conf['problem']['ham_type']
        self.geometry = conf['problem']['geometry'].replace(" ", "_")

        self.fake_min_energy = conf[
            'env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys(
            ) else None
        self.fn_type = conf['env']['fn_type']

        # If you want to run agent from scratch without *any* curriculum just use the setting with
        # normal curriculum and set config[episodes] = [1000000]
                
        min_eig = conf['env']['fake_min_energy']
        self.curriculum_dict = {}
        self.curriculum_dict[self.ham] = curricula.__dict__[
            conf['env']['curriculum_type']](conf['env'], target_energy=min_eig)

        self.model = conf["architecture"]      
        self.device = device
        self.done_threshold = conf['env']['accept_err']

        stdout.flush()
        self.state_size = 5 * self.num_layers
        self.actual_layer = -1
        self.prev_energy = None
        self.energy = 0

        self.action_size = (self.num_qubits * (self.num_qubits + 2))
        self.train_loader, self.val_loader, self.test_loader = MOSIDataLoaders(self.model)


        if 'non_local_opt' in conf.keys():
            self.global_iters = conf['non_local_opt']['global_iters']
            self.optim_method = conf['non_local_opt']["method"]

            if conf['non_local_opt']["method"] in [
                    "Rotosolve_local_end", "Rotosolve_local_end_only_rot",
                    "scipy_local_end"
            ]:
                self.local_opt_size = conf['non_local_opt']["local_size"]
            if "optim_alg" in conf['non_local_opt'].keys():
                self.optim_alg = conf['non_local_opt']["optim_alg"]

        else:
            self.global_iters = 0
            self.optim_method = None

    def step(self, action, train_flag=True):
        """
        Action is performed on the first empty layer.
        Variable 'actual_layer' points last non-empty layer.
        """
        next_state = self.state.clone()
        self.actual_layer += 1
        """
        First two elements of the 'action' vector describes position of the CNOT gate.
        Position of rotation gate and its axis are described by action[2] and action[3].
        When action[0] == num_qubits, then there is no CNOT gate.
        When action[2] == num_qubits, then there is no Rotation gate.
        """

        next_state[0][self.actual_layer] = action[0]
        next_state[1][self.actual_layer] = (action[0] +
                                            action[1]) % self.num_qubits

        ## state[2] corresponds to number of qubit for rotation gate
        next_state[2][self.actual_layer] = action[2]
        next_state[3][self.actual_layer] = action[3]
        next_state[4][self.actual_layer] = torch.zeros(1)

        self.state = next_state.clone()

        _, report = Scheme(self)
        # self.energy = report['metrics']
        energy = report['metrics']['mae']

        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)

        self.error = float(abs(self.min_eig - energy))

        rwd = self.reward_fn(energy)
        self.prev_energy = np.copy(energy)

        energy_done = int(self.error < self.done_threshold)
        layers_done = self.actual_layer == (self.num_layers - 1)
        done = int(energy_done or layers_done)

        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(
                self.current_bond_distance)] = copy.deepcopy(self.curriculum)

        return next_state.view(-1).to(self.device), torch.tensor(
            rwd, dtype=torch.float32, device=self.device), done

    def reset(self):
        """
        Returns randomly initialized state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with control gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with not gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """
        ## state_per_layer: (Control_qubit, NOT_qubit, R_qubit, R_axis, R_angle)
        controls = self.num_qubits * torch.ones(self.num_layers)
        nots = torch.zeros(self.num_layers)
        rotats = self.num_qubits * torch.ones(self.num_layers)
        generatos = torch.zeros(self.num_layers)
        angles = torch.zeros(self.num_layers)

        state = torch.stack((controls.float(), nots.float(), rotats.float(),
                             generatos.float(), angles))
        self.state = state
        self.actual_layer = -1

        self.current_bond_distance = self.ham
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(
            self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(
            self.curriculum.get_current_threshold())        

        self.min_eig = self.fake_min_energy
        
        model = QNet(self.model, state).to(self.device)        
        metrics = evaluate(model, self.test_loader,self.device)

        self.prev_energy = metrics['mae']

        return state.view(-1).to(self.device)

    def reward_fn(self, energy):
        if self.fn_type == "staircase":
            return (0.2 * (self.error < 15 * self.done_threshold) + 0.4 *
                    (self.error < 10 * self.done_threshold) + 0.6 *
                    (self.error < 5 * self.done_threshold) + 1.0 *
                    (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "two_step":
            return (0.001 * (self.error < 5 * self.done_threshold) + 1.0 *
                    (self.error < self.done_threshold)) / 1.001
        elif self.fn_type == "two_step_end":
            max_depth = self.actual_layer == (self.num_layers - 1)
            if ((self.error < self.done_threshold) or max_depth):
                return (0.001 * (self.error < 5 * self.done_threshold) + 1.0 *
                        (self.error < self.done_threshold)) / 1.001
            else:
                return 0.0
        elif self.fn_type == "naive":
            return 0. + 1. * (self.error < self.done_threshold)
        elif self.fn_type == "incremental":
            return (self.prev_energy - energy) / abs(self.prev_energy -
                                                     self.min_eig)
        elif self.fn_type == "incremental_clipped":
            return np.clip((self.prev_energy - energy) /
                           abs(self.prev_energy - self.min_eig), -1, 1)
        elif self.fn_type == "nive_fives":
            max_depth = self.actual_layer == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = 0.
            return rwd
        elif self.fn_type == "incremental_with_fixed_ends":
            max_depth = self.actual_layer == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy) /
                              abs(self.prev_energy - self.min_eig), -1, 1)
            return rwd
        elif self.fn_type == "log":
            return -np.log(1 - (energy / self.min_eig))
        elif self.fn_type == "log_neg_punish":
            return -np.log(1 - (energy / self.min_eig)) - 5
        # elif self.fn_type == "end_energy":
        #     max_depth = self.actual_layer == (self.num_layers - 1)
        #     if ((self.error < self.done_threshold) or max_depth):
        #         rwd = (self.max_eig - energy) / (abs(self.min_eig) +
        #                                          abs(self.max_eig))
        #     else:
        #         rwd = 0.0
            return rwd


if __name__ == "__main__":
    pass