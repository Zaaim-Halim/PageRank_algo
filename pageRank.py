# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 11:24:57 2021

@author: Zaaim Halim
"""

import xml.etree.ElementTree as ET
import numpy as np

class GrapheConstruction:
    def __init__(self,file=None):
        self.graphe = {}
        self.graphe = self.construct_graphe(file)
    def construct_graphe(self,file):
        tree = ET.parse(file)
        root = tree.getroot()
        nodes = {}
        edges = []
        for child in root:
            if str(child.tag) == "nodes":
                
               for node in child:
                   nodes[node.attrib["id"]] = []
            if str(child.tag) == "edges":
                for edge in child:
                    edgee = edge.attrib
                    
                    edges.append(edgee)
        for node in nodes.keys():
            out_nodes = []
            for edge in edges:
                if node == edge["source"]:
                    out_nodes.append(edge["target"])
                nodes[node]=out_nodes 
        return nodes
    
    def get_graph(self):
        return self.graphe
    def get_graphe_size(self):
        return len(self.graphe)
        
        
class PageRank:
    def __init__(self , graphe=None,size=None,alpha=0.85,threshold=0.000001):
        self.probability_matrix = self.compute_matrix_probability(graphe,alpha,size)
        self.pageRank, self.iterr = self.compute_page_rank(graphe,self.probability_matrix,threshold)
    
    def transportation_matrix(self ,graphe,alpha,size):
        transportation = np.zeros(shape=(size,size),dtype=np.float32)
        keyList = [*graphe]
        for i in range(transportation.shape[0]):
           
            row_node = keyList[i]
            list_out = graphe[row_node]
            l= graphe[row_node]
            if len(list_out) != 0:
                for n in list_out:
                    pos = keyList.index(n)
                    transportation[i][pos] = alpha/(len(l))
        #print("transportation : ", transportation)            
        return transportation
            
    def teleportation_matrix(self,graphe,alpha,size):
        alpha_negation = 1 - alpha
        teleportation = np.ones(shape=(size,size),dtype=np.float32)
        keyList = [*graphe]
        for i in range(teleportation.shape[0]):
            row_node = keyList[i]
            list_out = graphe[row_node]
            s = len(list_out)
            if s == 0:
                for j in range(size):
                    teleportation[i][j] = 1 / size
            else:
                for j in range(size):
                    teleportation[i][j] = alpha_negation / size
        
        #print(" teleportation : " ,teleportation)
        return teleportation
    
    def compute_matrix_probability(self,graphe,alpha,size):
         p = self.transportation_matrix(graphe,alpha,size) + self.teleportation_matrix(graphe,alpha,size)
         #print("probability : ",p)
         return p
     
    def compute_page_rank(self,graphe,probability_matrix,threshold):
        keyList = [*graphe]
        # init page rank 
        pageR = np.full((1, probability_matrix.shape[0]), 1/probability_matrix.shape[0]).reshape(-1)
        iterr = 0
        while(True):
            old_pager = pageR.copy()
            pageR = pageR.dot(probability_matrix)
            res =np.absolute(pageR - old_pager)
            iterr = iterr + 1
            if np.linalg.norm(res) < threshold: 
                break
        
        pageRank = []
        for i in range(len(keyList)):
            pageRank.append((keyList[i],str(float("{:.2f}".format(pageR[i]*100)))+"%"))
        
       
        #print("Final Page Rank : ",pageRank)
        
        return (pageRank , iterr)
    
    def get_PageRank(self):
        return (self.pageRank, self.iterr)
    

def main():
    GC = GrapheConstruction('pageRank.xml')
    graphe = GC.get_graph()
    matrix_size = GC.get_graphe_size()
    PR = PageRank(graphe=graphe,size=matrix_size)
    pageRank , iterr = PR.get_PageRank()
    
    for tup in pageRank:
        print ("Page ",tup[0], " : ",tup[1])
        
    print ("Number of iteration : ",iterr)
    

main()