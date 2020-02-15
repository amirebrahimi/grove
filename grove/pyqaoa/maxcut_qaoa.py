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
    max_weight = 0
    for i, j, d in graph.edges(data=True):
        # According to https://grove-docs.readthedocs.io/en/latest/qaoa.html#quantum-approximate-optimization
        # the hamiltonian for the cost function for the edge returns a 1 (or the weight of the edge), so this
        # is where we introduce the edge weight
        weight = d['weight']
        max_weight = max(max_weight, abs(weight))
        coeff = 0.5 * weight
        cost_operators.append(PauliTerm("Z", i, coeff) * PauliTerm("Z", j) + PauliTerm("I", 0, -coeff))
    for i in graph.nodes():
        driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))

    if connection is None:
        connection = get_qc(f"{len(graph.nodes)}q-qvm")

    if minimizer_kwargs is None:
        # it's necessary to adjust tolerance by the order of magnitude to converge on noisy systems
        tolerance = 10 ** -(2 - np.math.ceil(np.math.log10(max_weight)))
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': tolerance, 'xtol': tolerance,
                                        'disp': True}}
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


def run(graph, noisy=False, show=False, steps=2, initial_beta=None, initial_gamma=None):
    connection = None
    samples = None
    if noisy:
        graph_len = len(graph.nodes) if isinstance(graph, nx.Graph) else len(graph)
        connection = get_qc(f"{graph_len}q-qvm", noisy=True)
        samples = 1000

    inst, graph = maxcut_qaoa(graph, steps=steps, rand_seed=42, connection=connection, samples=samples,
                              initial_beta=initial_beta, initial_gamma=initial_gamma)
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
    # Parameters: [3.92535501 1.17150349 4.05700461 3.92057841]
    # E = > -3.70988356218026
    # |000> [0.0142092 + 0.j]
    # |001> [0.40340158 + 0.j]
    # |010> [0.04119461 + 0.j]
    # |011> [0.04119461 + 0.j]
    # |100> [0.04119461 + 0.j]
    # |101> [0.04119461 + 0.j]
    # |110> [0.40340158 + 0.j]
    # |111> [0.0142092 + 0.j]
    # Most frequent bitstring from sampling (0, 1, 1)

    # The following has been commented out because when noise emulation is turned on it takes longer to simulate
    # For example, on my i7-8750H 2.2 GHz w/ 16GB of RAM it takes ~3 minutes

    from datetime import datetime
    started = datetime.now()
    #run([(0, 1, a), (1, 2, b), (0, 2, c)], noisy=True, show=True)
    print("Started at: " + started.strftime("%H:%M:%S"))
    print("Finished at: " + datetime.now().strftime("%H:%M:%S"))

    # The final results (incorrect results) w/ noise turned on were:
    # Parameters: [3.29577815 1.22176444 3.80235654 3.98958404]
    # E = > -2.8659999999999997
    # |000> [0.0063171 + 0.j]
    # |001> [0.04075252 + 0.j]
    # |010> [0.22646519 + 0.j]
    # |011> [0.22646519 + 0.j]
    # |100> [0.22646519 + 0.j]
    # |101> [0.22646519 + 0.j]
    # |110> [0.04075252 + 0.j]
    # |111> [0.0063171 + 0.j]
    # Most frequent bitstring from sampling
    # (1, 0, 1)

    # However, with 3 Trotterization steps the correct answer is found
    # run([(0, 1, a), (1, 2, b), (0, 2, c)], noisy=True, show=True, steps=3)

    # An interesting experiment was to try a noisy qc w/ an initial beta/gamma matching the result of the non-noisy run
    # run([(0, 1, a), (1, 2, b), (0, 2, c)], noisy=True, show=True,
    #     initial_beta=[3.92535501 1.17150349], initial_gamma=[4.05700461 3.92057841])

    # Try out some other connected graphs with random weights
    max_weight = 4
    G = nx.complete_graph(5)
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = random.randint(1, max_weight)
    run(G, show=True)
