"""
Usage: 
conda activate nscl
export PATH=/Users/zhou/VILab/NS-CL/Jacinle/bin:$PATH
jac-run scripts/scene_graph_test.py
"""

import torch

import nscl.nn.scene_graph.scene_graph as sng


scene_graph = sng.SceneGraph(256, [None, 256, 256], 16)
print("SCENE_GRAPH")
print(scene_graph)


input = torch.rand(size=(1, 256, 16, 24))
objects = torch.rand(size=(1, 4))
objects_length = torch.tensor([1])

print("SCENE_GRAPH forward()")
print(scene_graph.forward(input, objects, objects_length))