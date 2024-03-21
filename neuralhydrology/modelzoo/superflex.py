from typing import Tuple
import re

import torch
from torch.nn.functional import softplus
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.inputlayer import InputLayer

ELEMENTS_INPUTS = {
    'SnowReservoir': ['static_inputs', 'precip', 'tmin', 'tmax'],
    #'ThresholdReservoir': ['x_in','x_out'],
    'ThresholdReservoir': ['x_in', 'pe'],
    'RoutingReservoir': ['x_in'],
    'FluxPartition': ['flux', 'x'],
    'LagFunction': ['x_in'],
    'Inputs': [],
    'CatFunction': ['x'],
    'Transparent': ['x'],
    'NegNode': ['x'],
    'BiasNode': ['x'],
    'ScaleNode': ['x'],
    'Splitter': ['x'],
    'FullyConnected': ['x']
}
ELEMENTS_OUTPUTS = {
    'SnowReservoir': ['miss_flux', 'snowmelt'],
    'ThresholdReservoir': ['overflow'],
    'RoutingReservoir': ['outflow'],
    'FluxPartition': [],
    'LagFunction': ['output'],
    'CatFunction': ['output'],
    'Transparent': ['output'],
    'NegNode': ['output'],
    'BiasNode': ['output'],
    'ScaleNode': ['output'],
    'Splitter': ['output'],
    'FullyConnected': ['output']
}  #overrided
ELEMENTS_NB_PARMETERS = {
    'SnowReservoir': 1,
    #'ThresholdReservoir': 1, for euler
    'ThresholdReservoir': 2,  # for analytic
    'RoutingReservoir': 1,
    'FluxPartition': 0,
    'LagFunction': 0,  #overrided
    'Inputs': 0,
    'CatFunction': 0,
    'Transparent': 0,
    'NegNode': 0,
    'BiasNode': 1,
    'ScaleNode': 1,
    'Splitter': 0,  #overrided
    'FullyConnected': 0
}


class _DirectedLabelledGraph:
    """
    This class exclusively used by the 'parser' function to represent the model under construction.
    """

    def __init__(self) -> None:
        self.nodes = []
        self.layer = -1

    def __repr__(self) -> str:
        res = ""
        for i, n in enumerate(self.nodes):
            res += str(i) + ' ' + str(n) + '\n'
        return res

    def add_node(self,
                 name: str,
                 type_node: str,
                 param: int = 0,
                 outputs: list = [],
                 static_inputs_size: int = 0) -> None:
        """
        Add a new node in the graph.
        This method is called by the 'parser' function while parsing the 'Nodes' section
        of the model_description file.

        Parameters
        ----------
        name : str
            name of the node in the description file.
        type_node : str
            type of the node, one in ELEMENTS_INPUTS.keys()
        param : int, optional
            used for LagFunction only, corresponds to 'timesteps, by default 0.
        outputs : list, optional
            used for Inputs only, list of inputs names, by default []
        static_inputs_size : int, optional
            used for Inputs only, size of the static_inputs, by default 0.
        """
        if self.find_node(name) != None:
            raise ValueError(name + ' declared multiple times.')
        if type_node == 'SnowReservoir':
            n = _SnowNode(name, type_node)
        elif type_node == 'ThresholdReservoir':
            n = _ThresholdNode(name, type_node)
        elif type_node == 'RoutingReservoir':
            n = _RoutingNode(name, type_node)
        elif type_node == 'FluxPartition':
            n = _FluxNode(name, type_node)
        elif type_node == 'LagFunction':
            n = _LagNode(name, type_node, param)
        elif type_node == 'Inputs':
            n = _InputNode(name, type_node, outputs, static_inputs_size)
        elif type_node == 'CatFunction':
            n = _CatNode(name, type_node)
        elif type_node == 'Transparent':
            n = _TransparentNode(name, type_node)
        elif type_node == 'NegNode':
            n = _NegNode(name, type_node)
        elif type_node == 'BiasNode':
            n = _BiasNode(name, type_node)
        elif type_node == 'ScaleNode':
            n = _ScaleNode(name, type_node)
        elif type_node == 'Splitter':
            n = _SplitterNode(name, type_node)
        elif type_node == 'FullyConnected':
            n = _FullyConnectedLayerNode(name, type_node, param)
        else:
            raise ValueError(type_node + ' is not a valid element type.')
        self.nodes.append(n)

    def find_node(self, name: str):
        """
        Return the index of the node with name `name` in `self.nodes`.
        If there is no node with this name, returns `None`.

        Parameters
        ----------
        name : str
            name of the node to look for.

        Returns
        -------
        int 
            index of this node.
        """
        for i, n in enumerate(self.nodes):
            if n.name == name:
                return i
        return None

    def add_connection(self, output_node: str, output_data: str, input_node: str, input_data: str) -> None:
        """
        Add an edge connecting node ``output_node``'s output channel ``output_data`` to node ``input_node``'s
        channel ``input_data``.

        Parameters
        ----------
        output_node : str
            name of the source node.
        output_data : str
            name of the source node's output channel.
        input_node : str
            name of the destination node.
        input_data : str
            name of the destination node's input channel.
        """
        error_msg = f'Cannot add an edge between {output_node} and {input_node}:'
        output_node_idx = self.find_node(output_node)
        if output_node_idx is None:
            raise SyntaxError(error_msg + f'{output_node} not declared.')
        input_node_idx = self.find_node(input_node)
        if input_node_idx is None:
            raise SyntaxError(error_msg + f'{input_node_idx} not declared.')
        output_data = self.nodes[output_node_idx].add_output(input_node_idx, output_data)
        data_size = self.nodes[output_node_idx].outputs_sizes[output_data]
        self.nodes[input_node_idx].add_input(input_data, output_node_idx, output_data, data_size)

    def check(self) -> None:
        """
        Check  1) if all the input channels of all the nodes are connected to at least one output channel, 
        and 2) if the graph is acyclic.

        If one of these two conditions if not verified, returns a ValueError.
        """
        # 1 - check completness
        for n in self.nodes:
            n.is_complete()
        # 2 - check DAG
        if not self.is_dag():
            raise ValueError("The model described is not acyclic.")

    def compute_layers(self) -> None:
        """
        Compute the layer value for each node and sets its ``layer`` to this value.
        The layer value of a node is the length of the longest path from the root to itself.
        """
        self.nodes[0].set_layer(0)
        queue = [self.nodes[0]]
        while queue:
            node = queue[0]
            queue = queue[1:]
            for son in node.sons:
                if son != None:
                    son = self.nodes[son]
                    son.set_layer(max(son.layer, node.layer + 1))
                    queue.append(son)

    def export(self, cfg):
        self.compute_layers()
        sorted_nodes = sorted(self.nodes, key=lambda node: node.layer)
        layers = []  #list of list of nodes
        nodes_inputs = [None for _ in self.nodes]
        nodes_outputs = [[None for _ in ELEMENTS_OUTPUTS[n.type_node]] for n in self.nodes]
        c = -1
        nb_parameters = 0
        for n in sorted_nodes:
            if n.layer != c:  # we know that n.layer == c+1, easy to prove
                c += 1
                layers.append([])
            node_idx = self.find_node(n.name)
            new_component = n.return_instance(cfg)
            layers[-1].append((new_component, node_idx))
            if n.type_node != "Inputs":
                nb_parameters += new_component.number_of_parameters

            if n.type_node == '_FullyConnectedLayerNode':
                nodes_outputs[node_idx] = []

            nodes_inputs[node_idx] = []
            for input_type in ELEMENTS_INPUTS[n.type_node]:
                nodes_inputs[node_idx].append([(nn, d) for nn, d in n.inputs[input_type]])

        if len(layers[-1]) != 1 or len(nodes_outputs[layers[-1][0][1]]) != 1:
            raise ValueError("Currently, bucket models only support a single target variable.")
        return layers, nodes_inputs, nodes_outputs, nb_parameters

    def _is_acyclic_dfs(self, node, visited, recursion_stack):
        visited[node.name] = True
        recursion_stack[node.name] = True
        for son in node.sons:
            son = self.nodes[son]
            if not visited[son.name]:
                if self._is_acyclic_dfs(son, visited, recursion_stack):
                    return True
            elif recursion_stack[son.name]:
                return True

        recursion_stack[node.name] = False
        return False

    def is_dag(self):
        visited = {node.name: False for node in self.nodes}
        recursion_stack = {node.name: False for node in self.nodes}

        for node in self.nodes:
            if not visited[node.name]:
                if self._is_acyclic_dfs(node, visited, recursion_stack):
                    return False

        return True


class _Node:

    def __init__(self, name, type_node) -> None:
        self.name = name
        self.type_node = type_node
        self.inputs = {i: [] for i in ELEMENTS_INPUTS[type_node]}
        self.sons = []
        self.nb_parameters = ELEMENTS_NB_PARMETERS[type_node]
        self.layer = -1
        self.outputs_sizes = [0 for _ in ELEMENTS_OUTPUTS[type_node]]

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name + ' ' + str(self.inputs) + ' ' + str(self.sons)

    def add_output(self, son_node, data):
        if data == False:
            if len(ELEMENTS_OUTPUTS[self.type_node]) > 1:
                raise ValueError(self.name + "has several outputs but none where chosen.")
            else:
                data = 0
        else:
            if data not in ELEMENTS_OUTPUTS[self.type_node]:
                raise ValueError(f"Node {self.name} of type {self.type_node} doesn't have any output channel {data}")
            data = ELEMENTS_OUTPUTS[self.type_node].index(data)
        if not son_node in self.sons:
            self.sons.append(son_node)
        return data

    def add_input(self, input_data, father_node, father_data):
        if not input_data:
            if len(self.inputs.keys()) == 1:
                input_data = next(iter(self.inputs))
            else:
                raise ValueError(self.name + " has several inputs and none was specified.")
        if not input_data in self.inputs:
            raise ValueError(f"Node {self.name} of type {self.type_node} doesn't have any input channel {input_data}")
        self.inputs[input_data].append((father_node, father_data))
        return input_data

    def is_complete(self):
        for i in self.inputs:
            if len(self.inputs[i]) == 0:
                raise ValueError(self.name + ' input ' + i + ' is not set.')

    def set_layer(self, layer):
        self.layer = layer


class _RoutingNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _RoutingReservoir(cfg)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:
            self.outputs_sizes[0] = data_size


class _ThresholdNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _ThresholdReservoir(cfg)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:
            self.outputs_sizes[0] = data_size


class _SnowNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _SnowReservoir(cfg)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if input_data == ELEMENTS_INPUTS['SnowReservoir'].index('precip'):
            if len(self.inputs[input_data]) == 1:
                self.outputs_sizes[0] = data_size
                self.outputs_sizes[1] = data_size


class _FluxNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)
        self.input_size_x = 0

    def return_instance(self, cfg):
        return _FluxPartition(cfg, num_inputs=self.input_size_x, num_outputs=len(self.sons), activation='sigmoid')

    def add_output(self, son_node, data):
        data = len(self.sons)
        if not son_node in self.sons:
            self.sons.append(son_node)
            if len(self.outputs_sizes) == 0:
                self.outputs_sizes.append(None)
            else:
                self.outputs_sizes.append(self.outputs_sizes[-1])
        return data

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if input_data == ELEMENTS_INPUTS['FluxPartition'].index('flux'):
            if len(self.inputs[input_data]) == 1:
                self.outputs_sizes = [data_size for _ in self.sons]
        else:
            if len(self.inputs[input_data]) == 1:
                self.input_size_x = data_size


class _LagNode(_Node):

    def __init__(self, name, type_node, time_steps) -> None:
        super().__init__(name, type_node)
        try:
            time_steps = int(time_steps)
        except TypeError:
            raise TypeError("The parameter of the LagFunction must be an integer.")
        if not time_steps:
            raise ValueError("A LagFunction must be declared with the number of time_steps (ex.: LagFunction(5)")
        self.nb_parameters = time_steps
        self.time_steps = time_steps

    def return_instance(self, cfg):
        return _LagFunction(cfg, self.time_steps)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:
            self.outputs_sizes[0] = data_size


class _InputNode(_Node):

    def __init__(self, name, type_node, outputs, static_inputs_size) -> None:
        ELEMENTS_OUTPUTS['Inputs'] = outputs
        super().__init__(name, type_node)
        self.outputs_sizes = [1 for _ in outputs]
        self.outputs_sizes[outputs.index('static_inputs')] = static_inputs_size

    def return_instance(self, cfg):
        return None


class _CatNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _CatFunction(cfg)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        self.outputs_sizes[0] += data_size


class _TransparentNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _Transparent(cfg)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:
            self.outputs_sizes[0] = data_size


class _NegNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _Neg(cfg)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:
            self.outputs_sizes[0] = data_size


class _BiasNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _Bias(cfg)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:
            self.outputs_sizes[0] = data_size


class _ScaleNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _Scale(cfg)

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:
            self.outputs_sizes[0] = data_size


class _SplitterNode(_Node):

    def __init__(self, name, type_node) -> None:
        super().__init__(name, type_node)

    def return_instance(self, cfg):
        return _Splitter(cfg, num_outputs=len(self.sons))

    def add_output(self, son_node, data) -> int:
        data = len(self.sons)
        if not son_node in self.sons:
            self.sons.append(son_node)
            if len(self.sons) > 1:
                self.outputs_sizes.append(self.outputs_sizes[-1])
        return data

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:  #the first time we add an input
            self.outputs_sizes = [data_size for _ in range(max(1, len(self.sons)))]  #we set the output sizes


class _FullyConnectedLayerNode(_Node):

    def __init__(self, name, type_node, param) -> None:
        super().__init__(name, type_node)
        param = param.lower()
        if param not in ['sigmoid', 'tanh']:
            raise ValueError(param + " is not a valid activation function. 'tanh', and 'sigmoid' are supported.")
        self.activation_function = param

    def return_instance(self, cfg):
        return _FullyConnected(cfg, len(self.inputs['x']), self.activation_function, len(self.sons))

    def add_output(self, son_node, data) -> int:
        data = len(self.sons)
        if not son_node in self.sons:
            self.sons.append(son_node)
            if len(self.sons) > 1:
                self.outputs_sizes.append(self.outputs_sizes[-1])
        return data

    def add_input(self, input_data, father_node, father_data, data_size):
        input_data = super().add_input(input_data, father_node, father_data)
        if len(self.inputs[input_data]) == 1:
            self.outputs_sizes = [data_size for _ in range(max(1, len(self.sons)))]


def parser(input_file, cfg, static_inputs_size: int) -> Tuple:
    """
    Parse the model description file.

    Parameters
    ----------
    input_file: _io.TextIOWrapper
        The model description file.
        input_file = open(<nameofthefile>,'r')
    
    cfg: Config
        The run configuration
    
    static_inputs_size : int
        sum of the length of the static inputs.

    Returns
    -------
        Tuple
            A tuple containing five elements:
                - layers: a list of lists of tuples containing one element (SnowReservoir, Splitter... instance) and an int
                          layers[i][j][0] contains the element instance of the jth node in the ith layer
                          layers[i][j][0] contains the index of the jth node in the ith layer. This index will be used in
                          nodes_inputs and nodes_outputs just below.
                - nodes_inputs: a list of lists of lists of tuples containing two integers
                                nodes_inputs[i][j] = [(3,0),(4,1)] => the the jth inputs of the ith nodes is a
                                                                      combination of the 3rd node first input and
                                                                      4th node 2nd input.
                - nodes_outputs: list of lists of None
                                 nodes_inputs[i][j] will contain the jth output of the ith node. Initialized at None.
                - total_parameters: int
                                    total number of model parameter to optimized.
                - inputs: list of strings
                          contains the ordered list of inputs names.
    """
    categories = ["Inputs", "Nodes", "Edges"]
    graph = _DirectedLabelledGraph()
    input_names = []

    def is_category(line, categories):
        pattern = r'^(' + '|'.join(map(re.escape, categories)) + r')\s*$'
        match = re.match(pattern, line)
        return match is not None

    def remove_blank(input_text):
        return input_text.replace(" ", "").replace("\t", "").replace("\n", "")

    def match_words(input_text):
        pattern = r'\b\w+\b'
        matches = re.findall(pattern, input_text)
        return matches

    def is_empty_line(line):
        if '#' in line:
            line = line[:line.index('#')]
        empty_line_pattern = r'^\s*$'
        return re.match(empty_line_pattern, line) is not None

    def parse_input(line):
        input_names.append(match_words(line)[0])

    def parse_node(line):
        l = line.split(':')
        if len(l) == 0:
            raise SyntaxError(f'Line {line_counter}: exepected ":" somewhere in the line, got {original_line}')

        name = remove_blank(l[0])
        pattern = r'(\w+)(\[\d+\])'
        match = re.search(pattern, name)
        names = [name]
        if match:  # line of form  "name[x] : typenode"
            name = match.group(1)
            try:
                number = int(match.group(2)[1:-1])
                assert number > 0
            except:
                raise SyntaxError(
                    f'Line {line_counter}: the value between brackets must be a strictly positive integer, got {match.group(2)[1:-1]}'
                )
            names = [name + str(i + 1) for i in range(number)]

        typ = remove_blank(l[1])
        pattern = r'(\w+)(\(\w+\))'
        match = re.search(pattern, typ)
        param = None
        if match:  # line of form "name : typenode(x)"
            typ = match.group(1)
            if typ != 'LagFunction' and typ != 'FullyConnected':
                print(
                    f'WARN: line {line_counter}: the value between parenthesis will be ignored, since this node is neither a LagFunction or a FullyConnected.'
                )
            else:
                param = match.group(2)[1:-1]

        for name in names:
            graph.add_node(name, typ, param)

    def parse_edge(line):
        line = remove_blank(line)
        line = line.split('-')
        if len(line) == 0:
            raise SyntaxError(f'Line {line_counter}: exepected "-" somewhere in the line, got {original_line}')
        f = line[0]
        to = len(line) - 1
        # if line of form "from - (input_type) - to" => input_data = 1
        # if line of form "from - to" => input_data = False
        input_data = 1 if to == 2 else False
        if input_data:
            input_data = line[input_data][1:-1]
        to = line[to]

        pattern = r'(\w+)(\[\w+\])'
        match = re.search(pattern, f)
        # if line of form "from[smth] - to" => output_data = smth
        # if line of form "from - to" => output_data = False
        if match:
            f = match.group(1)
            output_data = match.group(2)[1:-1]
        else:
            output_data = False
        graph.add_connection(f, output_data, to, input_data)

    status = None
    line = input_file.readline()
    line_counter = 1
    original_line = line
    while line:
        if not is_empty_line(line):
            cat = is_category(line, categories)
            if cat:  # changing category
                if status == 'I':
                    # if the current category is Input we must the first
                    # node in the graph that represents the inputs.
                    if len(input_names) == 0:
                        raise SyntaxError(f'Line {line_counter}: no input given.')
                    graph.add_node('Inputs', 'Inputs', outputs=input_names, static_inputs_size=static_inputs_size)
                status = line[0]
                categories = categories[1:]
            else:
                if line[-1] == '\n':
                    line = line[:-1]
                if status == 'I':
                    parse_input(line)
                elif status == 'N':
                    parse_node(line)
                elif status == 'E':
                    parse_edge(line)
                else:
                    raise SyntaxError(f'Line {line_counter}: expected "Inputs", got {original_line}.')
        line = input_file.readline()
        line_counter += 1

    graph.check()
    layers, nodes_inputs, nodes_outputs, total_parameters = graph.export(cfg)
    return layers, nodes_inputs, nodes_outputs, total_parameters, input_names


class Superflex(BaseModel):
    """Superflex model class.

    This class allows the user to create a model similar to a SuperflexPy unit.
    A SuperflexPy unit is a collection of multiple connected elements (reservoirs, lag functions, splitters,...).
    A unit can be used to represent a lumped catchment model.
    The structure of the unit is described into file. The path to this file must be given in the 'model_description'
    field of the configuration file.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(Superflex, self).__init__(cfg=cfg)
        if len(cfg.target_variables) > 1:
            raise ValueError("Currently, superflex models only support a single target variable.")

        # The universal embedding network. This network can't transform the dynamic inputs
        # because that would change units of the conserved inputs (P & E).
        # This embedding network could be re-designed so that it acts on just statics, or on
        # statics and auxilary inputs. Auxilary inputs are dynamic inputs other than the mass-
        # conserved inputs.
        if cfg.dynamics_embedding is not None:
            raise ValueError("Embedding for dynamic inputs is not supported with the current bucket-style models.")

        self.embedding_net = InputLayer(cfg)

        # Build the parameterization network. This takes static catchment attributes as inputs and estimates
        # all of the parameters for all of the different model components at once. Parameters will be extracted
        # from the output vector of this model sequentially.
        static_inputs_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        # If the user requests using one-hot encoding to identify basins, this will be added to static inputs.
        if cfg.use_basin_id_encoding:
            static_inputs_size += cfg.number_of_basins

        if cfg.model_description is None:
            raise ValueError(
                "The model desctiption file must be provided in the configuration file 'model_description' field.")
        # Parse the model_description file and check if the model described is valid.
        with open(cfg.model_description, 'r') as input_file:
            self.layers, self.nodes_inputs, self.nodes_outputs, total_parameters, self.inputs = parser(
                input_file, cfg, static_inputs_size)
        # self.layers: list of elements (SnowReservoir, Splitter, LagFunction,etc ... instances)
        # self.nodes_inputs:  list of list of list of tuples containing two integers
        #                     self.nodes_inputs[i][j] = [(3,0),(4,1)] => the the jth inputs of the ith nodes is a
        #                                                               combination of the 3rd node first input and
        #                                                               4th node 2nd input.
        # self.nodes_outputs: list of list of None
        #                     self.nodes_inputs[i][j] will contain the jth output of the ith node. Initialized at None.
        # total_parameters  : int.
        #                     total number of model parameter to optimized.
        # self.inputs       : list of strings
        #                     contains the ordered list of inputs names.

        if static_inputs_size == 0:
            raise ValueError("At least one static input must be provided.")  # TODO

        #if cfg.static_embedding is None:
        #    hidden_sizes = cfg.static_embedding['hiddens'] if cfg.static_embedding['hiddens'] is not None else [20]
        #    dpout = cfg.static_embedding['dropout'] if cfg.static_embedding['dropout'] is not None else 0.0
        #    activ = cfg.static_embedding['activation'] if cfg.static_embedding['activation'] is not None else "tanh"

        self.parameterization = FC(input_size=static_inputs_size,
                                   hidden_sizes=[20, total_parameters],
                                   dropout=0.,
                                   activation="sigmoid")

    def _execute_graph(self, parameters: torch.Tensor, inputs: list[torch.Tensor]) -> torch.Tensor:
        #load the inputs
        self.nodes_outputs[0] = inputs

        parameters_count = 0
        # for each layer
        for i in range(1, len(self.layers)):
            # for each node in the layer
            for j in range(len(self.layers[i])):
                node, node_idx = self.layers[i][j]
                node_inputs = []
                # we first deal with the inputs:
                # for each input of this node

                for k in self.nodes_inputs[node_idx]:
                    node_inputs.append([])
                    # add all sources for this input in node_inputs[-1]
                    for l in k:
                        node_inputs[-1].append(self.nodes_outputs[l[0]][l[1]])
                    # if the node is a CatFunction the sources are concatenated
                    if type(node) == _CatFunction or type(node) == _FullyConnected:
                        node_inputs[-1] = torch.cat(node_inputs[-1], dim=-1)
                    # otherwhise they are element-wise sumed
                    else:
                        stacked_tensor = torch.stack(node_inputs[-1], dim=0)
                        node_inputs[-1] = torch.sum(stacked_tensor, dim=0)

                # then we deal with the parameters
                if node.number_of_parameters == 0:
                    p = None
                elif node.number_of_parameters == 1:
                    p = parameters[:, parameters_count]
                else:
                    p = parameters[:, parameters_count:parameters_count + node.number_of_parameters]
                parameters_count += node.number_of_parameters

                # finally we execute the node and put the outputs in nodes_outputs
                self.nodes_outputs[node_idx] = node(inputs=node_inputs, parameters=p)

        last_node_idx = self.layers[-1][0][1]
        # the output of the model is the output of the node in the last layer.
        # We know that there is exactly one node in the last layer and it has exactly one output:
        # we have checked it in _DirectedLabelledGraph.export
        return self.nodes_outputs[last_node_idx][0]

    def forward(
        self,
        data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass on the Superflex model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
        """
        # Prepare inputs through any universal embedding layer that might exist.
        # Since we don't allow embedding transforms in the current version, really
        # all this is doing is separating the dynamic and static inputs in a
        # way that conforms to NeuralHydrology convention.
        #  - x_d are dynamic inputs.
        #  - x_s are static inputs.
        x_d, x_s = self.embedding_net(data, concatenate_output=False)
        # Dimensions.
        time_steps, batch_size, _ = x_d.shape

        # Estimate model parameters.
        if self.parameterization:
            parameters = self.parameterization(x_s)
        # Initialize storage in all model components.
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j][0].initialize_bucket(batch_size=batch_size, device=x_d.device)
        # Execute time loop.
        output = []
        for t in range(time_steps):
            inputs = [x_s]  #WARNING !!!!! STATIC INPUTS MUST ALWAYS BE FIRST IN THE MODEL DESCRIPTION
            for i in range(len(self.inputs) - 1):
                inputs.append(torch.unsqueeze(x_d[t, :, i], -1))
            output.append(self._execute_graph(parameters=parameters, inputs=inputs))
        return {'y_hat': torch.stack(output, 1)}


class _RoutingReservoir(torch.nn.Module):
    """Initialize a routing bucket node.

    A routing bucket is a bucket with infinite height and a drain.
    It has one parameter: outflow rate. The parameter is treated dynamically, instead of
    as a fixed parameter, which allows the parameter to be either learned or estimated with
    an external parameterization network. The routing bucket only has a prescribed source
    flux.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(self, cfg: Config):
        super(_RoutingReservoir, self).__init__()
        self.number_of_parameters = 1

    def initialize_bucket(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        """
        # Initialization must happen at runtime so that we know the batch size and device.
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)

    def forward_explicit_euler(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass for a routing reservoir."""
        # Account for the source flux.
        x_in = inputs[0]
        rate = torch.unsqueeze(parameters, dim=-1)
        self.storage = self.storage.clone() + x_in

        # Ensure that the bucket rate parameter is in (0, 1).
        rate = torch.sigmoid(rate)

        # Outflow from leaky bucket.
        outflow = rate * self.storage
        self.storage = self.storage - outflow
        return [outflow]

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass for a routing reservoir."""
        # Account for the source flux.
        x_in = inputs[0]
        rate = torch.sigmoid(torch.unsqueeze(parameters, dim=-1))
        q = x_in + self.storage - (x_in * torch.exp(rate) + rate * self.storage - x_in) / (rate * torch.exp(rate))
        self.storage = self.storage.clone() + x_in - q
        return [q]


class _SnowReservoir(torch.nn.Module):
    """Initialize a snow bucket node.

    A snow bucket is a bucket with infinite height and a drain, where the input is partitioned
    into one flux that goes into the bucket and one flux that misses the bucket. This bucket has
    two parameters: flux partition and outflow rate. The parameters are treated dynamically, instead
    of being fixed, which allows them to be either learned or estimated with an external
    parameterization network. The snow bucket only has a prescribed source flux.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(self, cfg: Config):
        super(_SnowReservoir, self).__init__()
        self.number_of_parameters = 1

        # Network for converting temperature, precip, and static attributes into a partitioning
        # coefficient between liquid and solid precip.
        static_inputs_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            static_inputs_size += cfg.number_of_basins

        self.precip_partitioning = _FluxPartition(
            cfg=cfg,
            num_inputs=3 + static_inputs_size,
            num_outputs=2,
        )

    def initialize_bucket(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass for a snow reservoir."""
        # Partition the in-flux.
        static_inputs, precip, tmin, tmax = inputs
        partition = self.precip_partitioning(inputs=[precip,
                                                     torch.cat([tmin, tmax, precip, static_inputs], dim=-1)],
                                             parameters=None)

        miss_flux = partition[0]
        self.storage = self.storage.clone() + partition[1]

        # Outflow from leaky bucket is snowmelt.
        # The rate parameter is in (0, 1).
        rate = torch.unsqueeze(torch.sigmoid(parameters), dim=-1)
        snowmelt = rate * self.storage
        self.storage = self.storage.clone() - snowmelt
        return [miss_flux, snowmelt]


class _LagFunction(torch.nn.Module):
    """Initialize a lag function.

    A generic lag function as a convolution. We use a storage vector to make this easy to plug into
    a model graph that operates over a single timestep.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(self, cfg: Config, timesteps: int):
        super(_LagFunction, self).__init__()
        # The number of timesteps in a lag function must be set.
        self.timesteps = timesteps
        self.number_of_parameters = timesteps

    def initialize_bucket(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size, convolution width,
        and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, self.timesteps], device=device)

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass for a lag function."""
        weights = parameters
        x_in = inputs[0]
        if weights.shape[-1] != self.timesteps:
            raise ValueError(f'Convolution weights must be the same dimenions as the conv filter. '
                             f'Expected {self.timesteps}, received {weights.shape[-1]}.')

        # Add to the storage in the filter.
        self.storage = self.storage.clone() + x_in * weights

        # Shift the filter.
        outflow = torch.unsqueeze(self.storage[:, -1], dim=-1)
        self.storage[:, 1:] = self.storage[:, :-1].clone()
        self.storage[:, 0] = self.storage.new_zeros([self.storage.shape[0]])
        return [outflow]


class _FluxPartition(torch.nn.Module):
    """Fully connected layer with N normalized outputs.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as number of basins).
    num_inputs: Number of inputs to the fully connected layer.
    num_outputs: Number of normalized outputs.

    Raises
    ------
    ValueError
        Given an unsupported ativation function.
    ----------
    """

    def __init__(self, cfg: Config, num_inputs: int, num_outputs: int, activation: str = 'sigmoid'):
        super(_FluxPartition, self).__init__()

        self.fc = torch.nn.Linear(in_features=num_inputs, out_features=num_outputs)
        self.number_of_parameters = 0

        if activation.lower() == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation.lower() == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise ValueError(f'Flux partitioning currently only works with sigmoid and relu activation functions. '
                             f'Got {activation}. This is necessary becasue the activations must always be positive.')

        self._reset_parameters()

    def initialize_bucket(self, batch_size: int, device: torch.device) -> None:
        """Do nothing. Just for compatiblity reasons.
        """
        pass

    def _reset_parameters(self):
        torch.nn.init.orthogonal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Perform forward pass through the normalized gate"""
        flux, x = inputs
        weigths = self.activation(self.fc(x))
        normalized_weights = torch.nn.functional.normalize(weigths, p=1, dim=-1)

        if flux.shape[-1] != 1:
            raise ValueError('FluxPartition network can only partition a scaler.')
        outputs = normalized_weights * flux
        outputs = torch.split(outputs, 1, dim=1)
        return outputs


class _ThresholdReservoir(torch.nn.Module):
    """Initialize a threshold bucket node.

    A threshold bucket is a bucket with finite height and no drain. Outflow is from overflow.
    It has one parameter: bucket height. The parameter is treated dynamically, instead of
    as a fixed parameter, which allows the parameter to be either learned or estimated with
    an external parameterization network. The threshold bucket has prescribed source and sink
    fluxes.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(self, cfg: Config):
        super(_ThresholdReservoir, self).__init__()
        #self.number_of_parameters = 1 #for euler
        self.number_of_parameters = 2  #for analytic

    def initialize_bucket(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)

        # Initialize a tensor of zeros to use in the threshold calculation.
        self._zeros = torch.zeros_like(self.storage)

    def forward_euler(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass for a threshold reservoir."""
        # Account for prescribed fluxes (e.g., E, P)
        x_in, x_out = inputs
        height = parameters  #+ 20.0
        height = torch.unsqueeze(height, dim=-1)
        self.storage = self.storage.clone() + x_in
        self.storage = self.storage.clone() - torch.minimum(x_out, self.storage)
        # Ensure that the bucket height parameter is positive.
        height = torch.abs(height)
        # Calculate bucket overflow.
        overflow = torch.maximum(self.storage - height, self._zeros)
        self.storage = self.storage.clone() - overflow
        return [overflow]

    def forward(self, inputs: list[torch.Tensor], parameters: list[torch.Tensor]) -> list[torch.Tensor]:
        """Analytic forward pass for a threshold reservoir."""
        # Account for prescribed fluxes (e.g., E, P)
        p, ep = inputs
        smax, k = parameters.T
        k = torch.unsqueeze(k, dim=-1)
        smax = softplus(smax)
        smax = torch.unsqueeze(smax, dim=-1)
        ep = torch.abs(ep)

        condition = (1 + torch.tanh(k * (p - ep))) / 2
        q = condition * (p - smax * torch.tanh(p / smax) * (1 - (self.storage / smax)**2) /
                         (1 + (self.storage / smax) * torch.tanh(p / smax)))
        e = (1 - condition) * (p + ep * self.storage / (ep + smax / (2 - self.storage / smax)))
        self.storage = self.storage + p - q - e
        return [q]

    def forward_analytic(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Previous version, leads to parameters = NaN"""
        p, ep = inputs
        b, c, smax, k = parameters.T
        b = softplus(b)
        c = torch.sigmoid(c)
        b = torch.unsqueeze(b, dim=-1)
        c = torch.unsqueeze(c, dim=-1)
        k = torch.unsqueeze(k, dim=-1)
        smax = softplus(smax)  #+20.0
        smax = torch.unsqueeze(smax, dim=-1)

        condition = (1 + torch.tanh(k * (p - ep))) / 2

        q = condition * ((p - ep) + self.storage - smax + smax * softplus((1 - (self.storage / smax))**(1 / (b + 1)) -
                                                                          (p - ep) / ((b + 1) * smax))**(b + 1))
        e = condition * ep + (1 - condition) * (p + self.storage + smax * softplus(
            (p - ep) * (1 - c) / smax + (self.storage / smax)**(1 - c))**(1 / (1 - c)))
        assert not torch.isnan(q).any().item()
        assert not torch.isnan(e).any().item()
        #assert (q>=0).all().item()
        #assert (e>=0).all().item()
        self.storage = self.storage.clone() + p - e - q
        print('new storage', self.storage)
        #assert (self.storage>=0).all().item()
        #assert (smax>=self.storage).all().item()
        print()

        return [q]


class _CatFunction(torch.nn.Module):

    def __init__(self, cfg: Config):
        super(_CatFunction, self).__init__()
        self.number_of_parameters = 0

    def initialize_bucket(self, batch_size: int, device: torch.device) -> None:
        pass

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        return inputs


class _Transparent(torch.nn.Module):

    def __init__(self, cfg: Config):
        super(_Transparent, self).__init__()
        self.number_of_parameters = 0

    def initialize_bucket(self, batch_size: int, device: torch.device) -> None:
        pass

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        return inputs


class _Neg(torch.nn.Module):

    def __init__(self, cfg: Config):
        super(_Neg, self).__init__()
        self.number_of_parameters = 0

    def initialize_bucket(self, batch_size: int, device: torch.device) -> None:
        pass

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        return [-1 * inputs[0]]


class _Bias(torch.nn.Module):

    def __init__(self, cfg: Config):
        super(_Bias, self).__init__()
        self.number_of_parameters = 1

    def initialize_bucket(self, batch_size: int, device: torch.device) -> None:
        pass

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        x_in = inputs[0]
        rate = torch.unsqueeze(parameters, dim=-1)
        outflow = rate + x_in
        return [outflow]


class _Scale(torch.nn.Module):

    def __init__(self, cfg: Config):
        super(_Bias, self).__init__()
        self.number_of_parameters = 1

    def initialize_bucket(self, batch_size: int, device: torch.device) -> None:
        pass

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        x_in = inputs[0]
        rate = torch.unsqueeze(parameters, dim=-1)
        outflow = rate * x_in
        return [outflow]


class _FullyConnected(torch.nn.Module):
    """Fully connected layer with N normalized outputs.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as number of basins).
    num_outputs: Number of normalized outputs.
    ----------
    """

    def __init__(self, cfg: Config, num_inputs: int, activation_function: str, num_outputs: int):
        super(_FullyConnected, self).__init__()
        self.number_of_parameters = 0
        self.fc = FC(input_size=num_inputs, hidden_sizes=[num_outputs], dropout=0.0, activation=activation_function)

    def initialize_bucket(self, batch_size: int, device: torch.device) -> None:
        """Do nothing. Just for compatiblity reasons.
        """
        pass

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Perform forward pass through the normalized gate"""
        x = inputs[0]
        return self.fc(x)


class _Splitter(torch.nn.Module):
    """Fully connected layer with N normalized outputs.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as number of basins).
    num_outputs: Number of normalized outputs.
    ----------
    """

    def __init__(self, cfg: Config, num_outputs: int):
        super(_Splitter, self).__init__()
        self.number_of_parameters = num_outputs
        self.device = cfg.device

    def initialize_bucket(self, batch_size: int, device: torch.device) -> None:
        """Do nothing. Just for compatiblity reasons.
        """
        pass

    def forward(self, inputs: list[torch.Tensor], parameters: torch.Tensor) -> list[torch.Tensor]:
        """Perform forward pass through the normalized gate"""
        x = inputs[0]
        weights = torch.sigmoid(parameters)
        row_sums = weights.sum(dim=1, keepdim=True)  # Calculate row sums
        normalized_weights = weights / row_sums
        if x.shape[-1] != 1:
            raise ValueError('Splitter network can only partition a scaler.')
        outputs = normalized_weights * x
        outputs = torch.split(outputs, 1, dim=1)
        return outputs
