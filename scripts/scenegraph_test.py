"""
Usage: 
conda activate nscl
export PATH=/Users/zhou/VILab/NS-CL/Jacinle/bin:$PATH
jac-run scripts/scene_graph_test.py
"""

import torch

import nscl.nn.scene_graph.scene_graph as sng

# nscl/nn/reasoning_v1.py
#   number of channels = 256; downsample rate = 16.
#
# experiments/clevr/desc_nscl_derender.py
#   f_sng = self.scene_graph(f_scene, feed_dict.objects, feed_dict.objects_length)

scene_graph = sng.SceneGraph(256, [None, 256, 256], 16)
print("SCENE_GRAPH")
print(scene_graph)


input = torch.rand(size=(1, 256, 16, 24)) * 300
# objects = torch.rand(size=(1, 4))
# objects_length = torch.tensor([1])


objects = torch.tensor([[118.4000,  80.0000, 176.8000, 150.4000],
                        [143.2000, 168.8000, 177.6000, 203.2000],
                        [104.0000,  87.2000, 120.8000, 120.0000],
                        [216.0000,  86.4000, 299.2000, 168.0000],
                        [121.6000, 120.0000, 192.0000, 191.2000],
                        [ 84.8000, 116.8000, 114.4000, 145.6000],
                        [299.2000, 114.4000, 342.4000, 156.8000],
                        [328.8000, 108.0000, 357.6000, 136.0000],
                        [148.0000, 138.4000, 180.8000, 172.0000]])
objects_length = torch.tensor([3, 3, 3])
# image_filename = ['CLEVR_train_059460.png', 'CLEVR_train_061104.png', 'CLEVR_train_020008.png']
# image = tensor([3 x 3_channels x 256 x 256])

print("SCENE_GRAPH forward()")
print(scene_graph.forward(input, objects, objects_length))