import torch

import nscl.nn.scene_graph.scene_graph as sng


scene_graph = sng.SceneGraph(256, [None, 256, 256], 16)
print(scene_graph)


input = torch.rand(size=(1, 256, 16, 24))
objects = torch.rand(size=(1, 4)).cuda()
objects_length = torch.tensor([1]).cuda()


print(scene_graph.forward(input, objects, objects_length))