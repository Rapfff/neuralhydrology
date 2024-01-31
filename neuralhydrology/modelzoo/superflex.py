from typing import Tuple
import re 

import torch
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.inputlayer import InputLayer

ELEMENTS_INPUTS = {'SnowReservoir': ['static_input', 'precip','tmin','tmax'],
                   'ThresholdReservoir': ['x_in','x_out'],
                   'RoutingReservoir':['x_in'],
                   'FluxPartition':['flux','x','static_input'],
                   'LagFunction':['x_in'],
                   'Inputs':[],
                   'CatFunction':['x'],
                   'Transparent':['x'],
                   'Splitter':['x']}
ELEMENTS_OUTPUTS = {'SnowReservoir': ['miss_flux', 'snowmelt'],
                   'ThresholdReservoir': ['overflow'],
                   'RoutingReservoir':['outflow'],
                   'FluxPartition':[],
                   'LagFunction':['output'],
                   'CatFunction':['output'],
                   'Transparent':['output'],
                   'Splitter'   :['output']}
ELEMENTS_NB_PARMETERS = {'SnowReservoir': 1,
                         'ThresholdReservoir': 1,
                         'RoutingReservoir':1,
                         'FluxPartition':0,
                         'LagFunction':0, #overrided
                         'Inputs':0,
                         'CatFunction':0,
                         'Transparent':0,
                         'Splitter' :0 #overrided
                         }

class _DirectedLabelledGraph:
    def __init__(self) -> None:
        self.nodes = []
        self.layer = -1

    def __repr__(self) -> str:
        res = ""
        for i,n in enumerate(self.nodes):
            res += str(i)+' '+str(n)+'\n'
        return res

    def add_node(self,name,type_node,param=False,outputs=None):
        if self.find_node(name) != None:
            raise ValueError(name+' declared multiple times.')
        if type_node == 'SnowReservoir':
            n = _SnowNode(name,type_node)
        elif type_node == 'ThresholdReservoir':
            n = _ThresholdNode(name,type_node)
        elif type_node == 'RoutingReservoir':
            n = _RoutingNode(name,type_node)
        elif type_node == 'FluxPartition':
            n = _FluxNode(name,type_node)
        elif type_node == 'LagFunction':
            n = _LagNode(name,type_node,param)
        elif type_node == 'Inputs':
            n = _InputNode(name,type_node,outputs)
        elif type_node == 'CatFunction':
            n = _CatNode(name,type_node)
        elif type_node == 'Transparent':
            n = _TransparentNode(name,type_node)
        elif type_node == 'Splitter':
            n = _SplitterNode(name, type_node)
        else:
            raise ValueError(type_node + ' is not a valid element type.')
        self.nodes.append(n)
    
    def find_node(self,name):
        for i,n in enumerate(self.nodes):
            if n.name == name:
                return i
        return None
    
    def add_connection(self, output_node, output_data, input_node, input_data):
        output_node_idx = self.find_node(output_node)
        input_node_idx  = self.find_node(input_node )
        output_data = self.nodes[output_node_idx].add_output(input_node_idx, output_data)
        self.nodes[input_node_idx].add_input(input_data, output_node_idx, output_data)
    
    def check(self):
        # 1 - check completness
        for n in self.nodes:
            if not n.is_complete():
                return False
        # 2 - check DAG
        if not self.is_dag():
            raise ValueError("The model described is not acyclic.")
    
    def compute_layers(self):
        self.nodes[0].set_layer(0)
        queue = [self.nodes[0]]
        while queue:
            node = queue[0]
            queue = queue[1:]
            for son in node.sons:
                if son != None:
                    son = self.nodes[son]
                    son.set_layer(max(son.layer, node.layer+1))
                    queue.append(son)

    def export(self, cfg, statics_input_size):
        self.compute_layers()
        sorted_nodes = sorted(self.nodes, key=lambda node: node.layer)
        layers = [] #list of list of nodes
        nodes_inputs = [None for _ in self.nodes]
        nodes_outputs = [[None for _ in ELEMENTS_OUTPUTS[n.type_node]] for n in self.nodes]
        c = -1
        nb_parameters = 0
        for n in sorted_nodes:
            nb_parameters += n.nb_parameters
            if n.layer != c: # we know that n.layer == c+1, easy to prove
                c+=1
                layers.append([])
            node_idx = self.find_node(n.name)
            layers[-1].append((n.return_instance(cfg, statics_input_size),node_idx))

            nodes_inputs[node_idx] = []
            for input_type in ELEMENTS_INPUTS[n.type_node]:
                nodes_inputs[node_idx].append([(n, d) for n, d in n.inputs[input_type]])

        
        if len(layers[-1]) != 1 or len(nodes_outputs[layers[-1][0][1]]) != 1:
            raise ValueError("Currently, bucket models only support a single target variable.")
        return layers, nodes_inputs, nodes_outputs, nb_parameters
        
    def is_acyclic_dfs(self, node, visited, recursion_stack):
        visited[node.name] = True
        recursion_stack[node.name] = True
        for son in node.sons:
            son = self.nodes[son]
            if not visited[son.name]:
                if self.is_acyclic_dfs(son, visited, recursion_stack):
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
                if self.is_acyclic_dfs(node, visited, recursion_stack):
                    return False

        return True
    
class _Node:
    def __init__(self, name, type_node) -> None:
        self.name = name
        self.type_node = type_node
        self.inputs = {i : [] for i in ELEMENTS_INPUTS[type_node]}
        self.sons = []
        self.nb_parameters = ELEMENTS_NB_PARMETERS[type_node]
        self.layer = -1

    def __repr__(self) -> str:
        return self.name
    
    def __str__(self) -> str:
        return self.name +' '+ str(self.inputs) +' '+ str(self.sons)

    def add_output(self, son_node, data):
        if data == False:
            if len(ELEMENTS_OUTPUTS[self.type_node]) > 1:
                raise ValueError(self.name + "has several outputs but none where chosen.")
            else:
                data = 0
        else:
            data = ELEMENTS_OUTPUTS[self.type_node].index(data)
        if not son_node in self.sons:
            self.sons.append(son_node)
        return data
    
    def add_input(self,input_data, father_node, father_data):
        if not input_data:
            if len(self.inputs.keys()) == 1:
                input_data = next(iter(self.inputs))
            else:
                raise ValueError(self.name+ " has several inputs and none was specified.")
        self.inputs[input_data].append((father_node,father_data))
    
    def is_complete(self):
        for i in self.inputs:
            if len(self.inputs[i]) == 0:
                raise ValueError(self.name+' input '+i+' is not set.')
        return True

    def set_layer(self,layer):
        self.layer = layer

class _RoutingNode(_Node):
    def __init__(self,name, type_node) -> None:
        super().__init__(name, type_node)
    def return_instance(self, cfg, *args):
        return RoutingReservoir(cfg)
    
class _ThresholdNode(_Node):
    def __init__(self,name, type_node) -> None:
        super().__init__(name, type_node)
    def return_instance(self, cfg, *args):
        return ThresholdReservoir(cfg)
    
class _SnowNode(_Node):
    def __init__(self,name, type_node) -> None:
        super().__init__(name, type_node)
    def return_instance(self, cfg, *args):
        return SnowReservoir(cfg)
    
class _FluxNode(_Node):
    def __init__(self,name, type_node) -> None:
        super().__init__(name, type_node)
    def return_instance(self, cfg, static_input_size):
        num_inputs = static_input_size + 4 #TODO replace 4 by dynamic value
        return FluxPartition(cfg, num_inputs=num_inputs, num_outputs=len(self.sons), activation='sigmoid')
    def add_output(self, son_node, data):
        data = len(self.sons)
        if not son_node in self.sons:
            self.sons.append(son_node)
        return data

class _LagNode(_Node):
    def __init__(self,name, type_node,time_steps) -> None:
        super().__init__(name, type_node)
        if not time_steps:
            raise ValueError("A LagFunction must be declared with the number of time_steps (ex.: LagFunction(5)")
        self.nb_parameters = time_steps
        self.time_steps = time_steps
    def return_instance(self, cfg, *args):
        return LagFunction(cfg,self.time_steps)

class _InputNode(_Node):
    def __init__(self,name, type_node,outputs) -> None:
        super().__init__(name, type_node)
        ELEMENTS_OUTPUTS['Inputs'] = outputs
    def return_instance(self, cfg, *args):
        return None

class _CatNode(_Node):
    def __init__(self,name, type_node) -> None:
        super().__init__(name, type_node)
    def return_instance(self, cfg, *args):
        return CatFunction(cfg)

class _TransparentNode(_Node):
    def __init__(self,name, type_node) -> None:
        super().__init__(name, type_node)
    def return_instance(self, cfg, *args):
        return Transparent(cfg)

class _SplitterNode(_Node):
    def __init__(self,name, type_node) -> None:
        super().__init__(name, type_node)
    def return_instance(self, cfg, *args):
        return Splitter(cfg, num_outputs=len(self.sons))
    def add_output(self, son_node, data) -> int:
        data = len(self.sons)
        if not son_node in self.sons:
            self.sons.append(son_node)
        return data

def parser(input_file,cfg,statics_input_size):
    categories = ["Inputs","Nodes","Edges"]
    graph = _DirectedLabelledGraph()
    input_names = []
    
    def is_category(line,categories):
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

        name = remove_blank(l[0])
        pattern = r'(\w+)(\[\d+\])'
        match = re.search(pattern, name)
        names = [name]
        if match:
            name = match.group(1)
            number = int(match.group(2)[1:-1])
            names = [name+str(i+1) for i in range(number)]
        
        typ  = remove_blank(l[1])
        pattern = r'(\w+)(\(\d+\))'
        match = re.search(pattern, typ)
        param = None
        if match:
            typ = match.group(1)
            param = int(match.group(2)[1:-1])
        
        for name in names:
            graph.add_node(name, typ, param)

    def parse_edge(line):
        line = remove_blank(line)
        line = line.split('-')
        f = line[0]
        to = len(line)-1
        input_data = 1 if to == 2 else False
        if input_data:
            input_data = line[input_data][1:-1]
        to = line[to]

        
        pattern = r'(\w+)(\[\w+\])'
        match = re.search(pattern, f)
        if match:
            f = match.group(1)
            output_data = match.group(2)[1:-1]
        else:
            output_data = False
        graph.add_connection(f,output_data,to,input_data)

    status = None
    line = input_file.readline()    
    while line:

        if not is_empty_line(line):
            cat = is_category(line, categories)
            if cat:
                if status == 'I':
                    graph.add_node('Inputs','Inputs',outputs=input_names)
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
        line = input_file.readline()

    graph.check()
    layers, nodes_inputs, nodes_outputs, total_parameters = graph.export(cfg, statics_input_size)
    return layers, nodes_inputs, nodes_outputs, total_parameters, input_names

class SuperFlex(BaseModel):
    """..."""

    def __init__(
        self,
        cfg: Config
    ):
        super(SuperFlex, self).__init__(cfg=cfg)

        if len(cfg.target_variables) > 1:
            raise ValueError("Currently, bucket models only support a single target variable.")

        self._n_mass_vars = len(cfg.mass_inputs)

        # The universal embedding network. This network can't transform the dynamic inputs
        # because that would change units of the conserved inputs (P & E).
        # This embedding network could be re-designed so that it acts on just statics, or on
        # statics and auxilary inputs. Auxilary inputs are dynamic inputs other than the mass-
        # conserved inputs.
        if cfg.dynamics_embedding is not None:
            raise ValueError("Embedding for dynamic inputs is not supported with the current bucket-style models.")

        self.embedding_net = InputLayer(cfg)

        # Build the parameterization network. This takes static catchment attributes as inputs and estiamtes
        # all of the parameters for all of the different model components at once. Parameters will be extracted
        # from the output vector of this model sequentially.
        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        # If the user requests using one-hot encoding to identify basins, this will be added to static inputs.
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        with open(cfg.model_description,'r') as input_file:
           self.layers, self.nodes_inputs, self.nodes_outputs, total_parameters, self.inputs = parser(input_file, cfg, statics_input_size)
        
        if statics_input_size == 0:
            raise ValueError("0 Static input given.")
        
        hidden_sizes = [20, total_parameters] # TODO Why 20 ?
        self.parameterization = FC(
        input_size=statics_input_size,
        hidden_sizes=hidden_sizes,
        dropout=0.,
        )

    def _execute_graph(
        self,
        parameters: torch.Tensor,
        inputs : list[torch.Tensor]
    ) -> torch.Tensor:
        """..."""
        self.nodes_outputs[0] = inputs
        parameters_count = 0
        for i in range(1,len(self.layers)):
            for j in range(len(self.layers[i])):
                node, node_idx = self.layers[i][j]
                node_inputs = []
                for k in self.nodes_inputs[node_idx]:
                    node_inputs.append([])
                    for l in k:
                        node_inputs[-1].append(self.nodes_outputs[l[0]][l[1]])
                    if type(node) == CatFunction:
                        node_inputs[-1] = torch.cat(node_inputs[-1], dim=-1)
                    else:
                        stacked_tensor = torch.stack(node_inputs[-1], dim=0)
                        node_inputs[-1] = torch.sum(stacked_tensor, dim=0)
                if node.number_of_parameters == 0:
                    p = None
                elif node.number_of_parameters == 1:
                    p = parameters[:, parameters_count]
                else:
                    p = parameters[:, parameters_count:parameters_count + node.number_of_parameters]
                parameters_count += node.number_of_parameters
                self.nodes_outputs[node_idx] = node(inputs=node_inputs,parameters=p)
        last_node_idx = self.layers[-1][0][1]
        return self.nodes_outputs[last_node_idx][0]

    def forward(
        self,
        data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """..."""
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
            parameters = self.parameterization(x=x_s)
        # Initialize storage in all model components.
        for i in range(1,len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j][0].initialize_bucket(batch_size=batch_size, device=x_d.device)
        # Execute time loop.
        output = []
        for t in range(time_steps):
            inputs = [x_s]
            for i in range(len(self.inputs)):
                inputs.append(torch.unsqueeze(x_d[t, :, i], -1))

            output.append(
                self._execute_graph(
                    parameters=parameters,
                    inputs=inputs
                )
            )
        return {'y_hat': torch.stack(output, 1)}

class RoutingReservoir(torch.nn.Module):
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

    def __init__(
        self,
        cfg: Config
    ):
        super(RoutingReservoir, self).__init__()
        self.number_of_parameters = 1

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size,1], device=device)

    def forward(
        self,
        inputs: list[torch.Tensor],
        parameters: torch.Tensor
    ) -> list[torch.Tensor]:
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

class SnowReservoir(torch.nn.Module):
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

    def __init__(
        self,
        cfg: Config
    ):
        super(SnowReservoir, self).__init__()
        self.number_of_parameters = 1

        # Network for converting temperature, precip, and static attributes into a partitioning
        # coefficient between liquid and solid precip.
        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        self.precip_partitioning = FluxPartition(
            cfg=cfg,
            num_inputs=3+statics_input_size,
            num_outputs=2,
        )

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)

    def forward(
        self,
        inputs: list[torch.Tensor],
        parameters: torch.Tensor
    ) -> list[torch.Tensor]:
        """Forward pass for a snow reservoir."""
        # Partition the in-flux.
        static_input, precip, tmin, tmax = inputs
        rate = parameters
        partition = self.precip_partitioning(inputs=[precip,torch.cat([tmin, tmax, precip], dim=-1),static_input], parameters=None)

        miss_flux = partition[0]
        self.storage = self.storage.clone() + partition[1]

        # Outflow from leaky bucket is snowmelt.
        # The rate parameter is in (0, 1).
        rate = torch.unsqueeze(torch.sigmoid(rate), dim=-1)       
        snowmelt = rate * self.storage
        self.storage = self.storage.clone() - snowmelt
        return [miss_flux, snowmelt]
    
class LagFunction(torch.nn.Module):
    """Initialize a lag function.

    A generic lag function as a convolution. We use a storage vector to make this easy to plug into
    a model graph that operates over a single timestep.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as
        number of basins).
    """

    def __init__(
        self,
        cfg: Config,
        timesteps: int
    ):
        super(LagFunction, self).__init__()
        # The number of timesteps in a lag function must be set.
        self.timesteps = timesteps
        self.number_of_parameters = timesteps

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size, convolution width,
        and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, self.timesteps], device=device)

    def forward(
        self,
        inputs: list[torch.Tensor],
        parameters: torch.Tensor
    ) -> list[torch.Tensor]:
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

class FluxPartition(torch.nn.Module):
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
    def __init__(
        self,
        cfg: Config,
        num_inputs: int,
        num_outputs: int,
        activation: str = 'sigmoid'
    ):
        super(FluxPartition, self).__init__()
        
        self.fc = torch.nn.Linear(in_features=num_inputs, out_features=num_outputs)
        self.number_of_parameters = 0

        if activation.lower() == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation.lower() == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise ValueError(
                f'Flux partitioning currently only works with sigmoid and relu activation functions. '
                f'Got {activation}. This is necessary becasue the activations must always be positive.'
            )

        self._reset_parameters()

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> None:
        """Do nothing. Just for compatiblity reasons.
        """
        pass

    def _reset_parameters(self):
        torch.nn.init.orthogonal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        inputs: list[torch.Tensor],
        parameters: torch.Tensor
    ) -> list[torch.Tensor]:
        """Perform forward pass through the normalized gate"""
        flux, x, static_input = inputs
        x = torch.cat([x,static_input], dim=-1)
        weigths = self.activation(self.fc(x))
        normalized_weights = torch.nn.functional.normalize(weigths, p=1, dim=-1)

        if flux.shape[-1] != 1:
            raise ValueError('FluxPartition network can only partition a scaler.')
        outputs = normalized_weights * flux
        outputs = torch.split(outputs, 1, dim=1)
        return outputs

class ThresholdReservoir(torch.nn.Module):
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

    def __init__(
        self,
        cfg: Config
    ):
        super(ThresholdReservoir, self).__init__()
        self.number_of_parameters = 1

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Initializes a bucket during runtime.
        
        Initialization must happen at runtime so that we know the batch size and device.
        """
        # We can do this however we want, but for now let's just start with every bucket being empty.
        self.storage = torch.zeros([batch_size, 1], device=device)
        
        # Initialize a tensor of zeros to use in the threshold calculation.
        self._zeros = torch.zeros_like(self.storage)

    def forward(
        self,
        inputs: list[torch.Tensor],
        parameters: torch.Tensor
    ) -> list[torch.Tensor]:
        """Forward pass for a threshold reservoir."""
        # Account for prescribed fluxes (e.g., E, P)
        x_in, x_out = inputs
        height = torch.unsqueeze(parameters,dim=-1)
        self.storage = self.storage.clone() + x_in
        self.storage = self.storage.clone() - torch.minimum(x_out, self.storage)
        # Ensure that the bucket height parameter is positive.
        height = torch.abs(height)
        # Calculate bucket overflow.
        overflow = torch.maximum(self.storage - height, self._zeros)
        self.storage = self.storage.clone() - overflow
        return [overflow]

class CatFunction(torch.nn.Module):

    def __init__(
        self,
        cfg: Config
    ):
        super(CatFunction, self).__init__()
        self.number_of_parameters = 0

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> None:
        pass

    def forward(
        self,
        inputs: list[torch.Tensor],
        parameters: torch.Tensor
    ) -> list[torch.Tensor]:
        return inputs

class Transparent(torch.nn.Module):

    def __init__(
        self,
        cfg: Config
    ):
        super(Transparent, self).__init__()
        self.number_of_parameters = 0

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> None:
        pass

    def forward(
        self,
        inputs: list[torch.Tensor],
        parameters: torch.Tensor
    ) -> list[torch.Tensor]:
        return inputs

class Splitter(torch.nn.Module):
    """Fully connected layer with N normalized outputs.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as number of basins).
    num_outputs: Number of normalized outputs.
    ----------
    """
    def __init__(
        self,
        cfg: Config,
        num_outputs: int
    ):
        super(Splitter, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(num_outputs, requires_grad=True))
        self.number_of_parameters = 0
        self._reset_parameters()

    def initialize_bucket(
        self,
        batch_size: int,
        device: torch.device
    ) -> None:
        """Do nothing. Just for compatiblity reasons.
        """
        pass

    def _reset_parameters(self):
        torch.nn.init.ones_(self.weights)

    def forward(
        self,
        inputs: list[torch.Tensor],
        parameters: torch.Tensor
    ) -> list[torch.Tensor]:
        """Perform forward pass through the normalized gate"""
        x = inputs[0]
        normalized_weights = torch.softmax(self.weights, dim=0)
        if x.shape[-1] != 1:
            raise ValueError('Splitter network can only partition a scaler.')
        outputs = normalized_weights * x
        outputs = torch.split(outputs, 1, dim=1)
        return outputs