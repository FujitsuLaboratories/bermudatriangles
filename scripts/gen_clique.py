#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# @author Yano Shotaro <yano.shotaro@fujitsu.com>
# @author Arseny Tolmachev <t.arseny@fujitsu.com>
#
# COPYRIGHT Fujitsu Limited 2020 and FUJITSU LABORATORIES LTD. 2020
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
# THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#-------------------------------------------------------------------------------

import os
import sys
import copy
import random
import string
import argparse
import textwrap
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def convert_to_list(argstr):
    try:
        arg_list = [int(i) for i in argstr.split(',')]
        if len(arg_list) > 2:
            return arg_list[:2]
        elif len(arg_list) == 1:
            arg_list.append(arg_list[0])
        return arg_list
    except ValueError:
        print('[Error] Please specify a number.')
        sys.exit(1)


def generateBA():
    n = random.randint(BA_nodeNum[0], BA_nodeNum[1]) # Number of nodes
    if BA_edge_rate == 0.0:
        m_max = Clique_nodeNum[0] - 2
    else:
        m_max = round((n-1) * BA_edge_rate)
    m = random.randint(1, m_max) # Number of edges to attach from a new node to existing nodes (1 <= m < n)
    #print('node: {}, edge: {}'.format(n, m))
    return nx.barabasi_albert_graph(n, m), n, m


def generateClique():
    n0 = random.randint(Clique_nodeNum[0], Clique_nodeNum[1])
    n1 = random.randint(Clique_nodeNum[0], Clique_nodeNum[1])
    #print('clique1: {}, clique2: {}'.format(n0, n1))
    G0, _ = relabelNodes(nx.complete_graph(n0), 'c0-')
    G1, _ = relabelNodes(nx.complete_graph(n1), 'c1-')
    return G0, G1


def relabelNodes(G, addstr=None):
    if not addstr is None:
        mapping = {i: addstr+str(i) for i in sorted(G)}
    else:
        mapping = {l: i for i, l in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping), mapping


def unionGraphs(G0, G1):
    G = nx.compose(G0, G1)
    n0 = random.choice(list(G0.nodes()))
    n1 = random.choice(list(G1.nodes()))
    #print('Union from {} to {}'.format(n0, n1))
    G.add_edge(n0, n1)
    return G, n1


def getDistance(G, n0, n1):
    shortestPath = nx.dijkstra_path(G, n0, n1)
    c0_length = len([n for n in shortestPath if str(n).startswith('c0-')])
    d = len(shortestPath) - c0_length
    if c0_length > 1:
        shortestPath = shortestPath[c0_length-1:]
    return d, shortestPath


def getRandomName(n, m):
    baseName = ''.join(random.choices(string.ascii_letters + string.digits, k=nameLength))
    return baseName + '_' + 'n' + n + '-' + 'm' + m


def getKeyFromValue(mapping, val):
    return [k for k, v in mapping.items() if v == val][0]


def addWrappingEdges(edges):
    edges_copy = copy.deepcopy(edges)
    edges_copy = [(edge[1], edge[0]) for edge in edges_copy]
    return edges + edges_copy


def writeFiles(cls0, cls1):
    # cl-file
    cl_list0 = [dataID+"\t"+str(0) for dataID in cls0.keys()]
    cl_list1 = [dataID+"\t"+str(1) for dataID in cls1.keys()]
    cl_list = cl_list0 + cl_list1
    cl_txt = '\n'.join(cl_list)

    # sp-file
    sp_list0 = [dataID+"\t"+str(edge[0])+"\t"+str(edge[1])+"\t"+str(1) \
        for dataID, edges in cls0.items() for edge in edges]
    sp_list1 = [dataID+"\t"+str(edge[0])+"\t"+str(edge[1])+"\t"+str(1) \
        for dataID, edges in cls1.items() for edge in edges]
    sp_list = sp_list0 + sp_list1
    sp_txt = '\n'.join(sp_list)

    # write
    with open(clfileName, mode='w', encoding='utf-8') as f1, open(spfileName, mode='w', encoding='utf-8') as f2:
        f1.write(cl_txt)
        f2.write(sp_txt)

    generateREADME()


def generateREADME():
    readmePath = os.path.join(dirName, 'README.md')

    string = textwrap.dedent(
        '''
        + Class definition (The number of data)
            - 0: Distance is less than {cls0}. ({num})
            - 1: Distance is {cls1} or more. ({num})
        + Number of nodes in BA graph. (rate: {rate})
            - min = {ba_min}
            - max = {ba_max}
        + Number of nodes in Clique.
            - min = {cl_min}
            - max = {cl_max}
        '''
    ).format(
        cls0=str(threshold[0]), cls1=str(threshold[1]),
        num=str(dataNum),
        rate=str(BA_edge_rate), ba_min=str(BA_nodeNum[0]), ba_max=str(BA_nodeNum[1]),
        cl_min=str(Clique_nodeNum[0]), cl_max=str(Clique_nodeNum[1])
        )

    with open(readmePath, mode='w', encoding='utf-8') as f:
        f.write(string)


def main(do_draw):
    cls0 = {}
    cls1 = {}
    pdfFile = os.path.join(dirName, 'all.pdf')
    pp = PdfPages(pdfFile)
    cnt = -1
    while (len(cls0) < dataNum) or (len(cls1) < dataNum):
        should_draw = False
        cnt += 1

        ''' 1. Generate BA(Barabashi Albert) graph (x1) '''
        BA_graph, BA_n, BA_m = generateBA()

        ''' 2. Generate clique graph (x2) '''
        clique0, clique1 = generateClique()

        ''' 3. Joins BA_graph, clique0 and clique1 '''
        G, n0 = unionGraphs(BA_graph, clique0)
        G, n1 = unionGraphs(G, clique1)

        ''' 4. Calculate the shortest distance from clique0 to clique1 '''
        d, shortestPath = getDistance(G, n0, n1)

        ''' 5. relabel (0~) '''
        G, mapping = relabelNodes(G)

        ''' 6. Get edge list '''
        edges = addWrappingEdges(list(G.edges()))

        dataID = f"G{cnt:08}-n{BA_n}-m{BA_m}-{mapping[n0]}t{mapping[n1]}-d{d}"

        if (d < threshold[0]) and (len(cls0) < dataNum):
            cls0[dataID] = edges
            should_draw = do_draw
        elif (d >= threshold[1]) and (len(cls1) < dataNum):
            cls1[dataID] = edges
            should_draw = do_draw

        # visualize
        if should_draw:
            fig = plt.figure()
            pos = nx.spring_layout(G, iterations=200)
            plt.title(dataID)
            ncolors = ['red' if getKeyFromValue(mapping, n) in shortestPath else 'blue' for n in G.nodes()]
            nx.draw_networkx(G, pos, with_labels=True, node_size=200, node_color=ncolors, font_color='white')
            plt.axis('off')
            pp.savefig(fig)
            plt.close()

        progress0 = "[class0] " + str(len(cls0)) + "/" + str(dataNum)
        progress1 = "[class1] " + str(len(cls1)) + "/" + str(dataNum)
        print("\r" + progress0+", "+progress1, end="")

    # write sp-file and cl-file
    writeFiles(cls0, cls1)
    pp.close()
    print("\n"+"End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate inter-clique distance problem dataset')
    parser.add_argument('--dataNum', '-n', default=10, type=int, help='Number of data in each class.')
    parser.add_argument('--BA_nodeNum', '-bn', default="10,15", type=str, help='Number of nodes in BA graph. (min,max)')
    parser.add_argument('--BA_edge_rate', '-br', default=0.0, type=float, help='Rate of changing the maximum value of m. (1 < m <= (n-1)*rate) When 0.0, m_max = min(Clique_nodeNum) - 2')
    parser.add_argument('--Clique_nodeNum', '-cn', default="4,5", type=str, help='Number of nodes in Clique. (min,max)')
    parser.add_argument('--threshold', '-th', default="2,2", type=str, help='Number of data in each class. -> Less than threshold[0] or more than threshold[1]')
    parser.add_argument('--spfileName', '-sp', default='./sp.txt', type=str, help='Path of sp-file to output.')
    parser.add_argument('--clfileName', '-cl', default='./cl.txt', type=str, help='Path of cl-file to output.')
    parser.add_argument('--nameLength', '-l', default=5, type=int, help='Number of characters in dataID.')
    parser.add_argument('--debug', '-d', action='store_true', help='Visualize the graph.')
    args = parser.parse_args()

    dataNum = args.dataNum
    BA_nodeNum = convert_to_list(args.BA_nodeNum)
    BA_edge_rate = args.BA_edge_rate
    Clique_nodeNum = convert_to_list(args.Clique_nodeNum)
    threshold = convert_to_list(args.threshold)
    spfileName = args.spfileName
    clfileName = args.clfileName
    nameLength = args.nameLength
    debug = args.debug

    dirName = os.path.dirname(spfileName)
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    main(debug)
