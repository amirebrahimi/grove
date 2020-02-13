##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

"""
Finding a maximum cut by QAOA.
"""
import random as random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pyquil.api import get_qc
from pyquil.paulis import PauliTerm, PauliSum
from scipy.optimize import minimize

from grove.pyqaoa.qaoa import QAOA


def maxcut_qaoa(graph, steps=1, rand_seed=None, connection=None, samples=None,
                initial_beta=None, initial_gamma=None, minimizer_kwargs=None,
                vqe_option=None):
    """
    Max cut set up method

    :param graph: Graph definition. Either networkx or list of tuples
    :param steps: (Optional. Default=1) Trotterization order for the QAOA algorithm.
    :param rand_seed: (Optional. Default=None) random seed when beta and gamma angles
        are not provided.
    :param connection: (Optional) connection to the QVM. Default is None.
    :param samples: (Optional. Default=None) VQE option. Number of samples
        (circuit preparation and measurement) to use in operator averaging.
    :param initial_beta: (Optional. Default=None) Initial guess for beta parameters.
    :param initial_gamma: (Optional. Default=None) Initial guess for gamma parameters.
    :param minimizer_kwargs: (Optional. Default=None). Minimizer optional arguments.  If None set to
        ``{'method': 'Nelder-Mead', 'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2, 'disp': False}``
    :param vqe_option: (Optional. Default=None). VQE optional arguments.  If None set to
        ``vqe_option = {'disp': print_fun, 'return_all': True, 'samples': samples}``

    """
    if not isinstance(graph, nx.Graph) and isinstance(graph, list):
        maxcut_graph = nx.Graph()
        for edge in graph:
            maxcut_graph.add_edge(edge[0], edge[1], weight=edge[2])
        graph = maxcut_graph.copy()

    cost_operators = []
    driver_operators = []
    for i, j, d in graph.edges(data=True):
        # According to https://grove-docs.readthedocs.io/en/latest/qaoa.html#quantum-approximate-optimization
        # the hamiltonian for the cost function for the edge returns a 1 (or the weight of the edge), so this
        # is where we introduce the edge weight
        coeff = 0.5 * d['weight']
        cost_operators.append(PauliTerm("Z", i, coeff) * PauliTerm("Z", j) + PauliTerm("I", 0, -coeff))
    for i in graph.nodes():
        driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))

    if connection is None:
        connection = get_qc(f"{len(graph.nodes)}q-qvm")

    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                        'disp': False}}
    if vqe_option is None:
        vqe_option = {'disp': print, 'return_all': True,
                      'samples': samples}

    qaoa_inst = QAOA(connection, list(graph.nodes()), steps=steps, cost_ham=cost_operators,
                     ref_ham=driver_operators, store_basis=True,
                     rand_seed=rand_seed,
                     init_betas=initial_beta,
                     init_gammas=initial_gamma,
                     minimizer=minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options=vqe_option)

    return qaoa_inst, graph


def run(graph, noisy=False, show=False):
    connection = None
    samples = None
    if noisy:
        graph_len = len(graph.nodes) if isinstance(graph, nx.Graph) else len(graph)
        connection = get_qc(f"{graph_len}q-qvm", noisy=True)
        samples = 10

    inst, graph = maxcut_qaoa(graph, steps=2, rand_seed=42, connection=connection, samples=samples)
    betas, gammas = inst.get_angles()
    probs = inst.probabilities(np.hstack((betas, gammas)))
    for state, prob in zip(inst.states, probs):
        print('|' + state + '>', prob)

    print("Most frequent bitstring from sampling")
    most_freq_string, sampling_results = inst.get_string(betas, gammas)
    print(most_freq_string)

    if show:
        pos = nx.spring_layout(graph)
        nx.draw_networkx(graph, pos=pos, node_color=most_freq_string)
        nx.draw_networkx_edge_labels(graph, pos=pos,
                                     edge_labels={(i, j): d['weight'] for (i, j, d) in graph.edges(data=True)})
        plt.show()


if __name__ == "__main__":
    # Sample Run:
    # Cutting 0 -  c  - 2 (2-regular) graph!
    #          \       /
    #           a     b
    #            \   /
    #              1
    # where a,b,c are the edge weights
    a = 2.0
    b = 1.0
    c = 2.0
    run([(0, 1, a), (1, 2, b), (0, 2, c)], show=True)

    # The final results w/o noise were:
    # Parameters: [3.03828487 0.55160361 4.18748713 4.18842267]
    # E = > 4.500011480688697
    # |000> [3.16140167e-07 + 0.j]
    # |001> [0.4999967 + 0.j]
    # |010> [1.49192789e-06 + 0.j]
    # |011> [1.49192789e-06 + 0.j]
    # |100> [1.49192789e-06 + 0.j]
    # |101> [1.49192789e-06 + 0.j]
    # |110> [0.4999967 + 0.j]
    # |111> [3.16140167e-07 + 0.j]
    # Most frequent bitstring from sampling
    # (1, 0, 0)

    # The following has been commented out because when noise emulation is turned on it takes extremely long to simulate
    # For example, on my i7-8750H 2.2 GHz w/ 16GB of RAM it took over 30 minutes

    # run([(0, 1, a), (1, 2, b), (0, 2, c)], noisy=True, show=True)

    # The final results w/ noise turned on were:
    # Parameters: [3.04911042 1.1638854  4.54558829 3.89335339]
    # E = > -2.0999999999999996
    # |000> [0.17065766 + 0.j]
    # |001> [0.15589796 + 0.j]
    # |010> [0.08672219 + 0.j]
    # |011> [0.08672219 + 0.j]
    # |100> [0.08672219 + 0.j]
    # |101> [0.08672219 + 0.j]
    # |110> [0.15589796 + 0.j]
    # |111> [0.17065766 + 0.j]
    # Most frequent bitstring from sampling
    # (1, 0, 0)

    # Try out some other connected graphs with random weights
    max_weight = 4
    G = nx.complete_graph(5)
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = random.randint(1, max_weight)
    run(G, show=True)
