from imgnetdatastuff import *
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Loads 250 items into dataloader
dataloader = dataset_imagenetvalpart("imgnet500", "ILSVRC2012_bbox_val_v3/val", 'synset_words.txt', 250)

modelvgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
modelvggbn = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Backward hook for saving grad norms
def addgradnorm(model, idx, item_name):
  def printgradnorm(module, grad_input, grad_output):
    item_filename = item_name.split("/")[-1].split(".")[0].split("_")[-1]
    filename = "grad_norm/%s/%s_%s_layer%d.pt" %(model, item_filename, module.__class__.__name__, idx)
    torch.save(grad_output[0].norm(), filename)
  return printgradnorm

# Save grad norms of conv layers closest to input into files
for i in range(0,250):
  item = dataloader.__getitem__(i)
  modelvgg.features[0].register_backward_hook(addgradnorm(model="vgg", idx=0, item_name=item['filename']))
  modelvgg.features[2].register_backward_hook(addgradnorm(model="vgg", idx=2, item_name=item['filename']))
  modelvggbn.features[0].register_backward_hook(addgradnorm(model="vggbn", idx=0, item_name=item['filename']))
  modelvggbn.features[3].register_backward_hook(addgradnorm(model="vggbn", idx=2, item_name=item['filename']))

  input_tensor = preprocess(item['image']).unsqueeze(0)
  output = modelvgg(input_tensor)
  output[-1][-1].backward()
  output = modelvggbn(input_tensor)
  output[-1][-1].backward()

# Stores vgg grad norms into array
grad_norm_array_vgg = []
for filename in os.listdir("grad_norm/vgg/"):
  x = torch.load("grad_norm/vgg/"+filename)
  grad_norm_array_vgg.append(x.item())

# Stores vggbn grad norms into array
grad_norm_array_vggbn = []
for filename in os.listdir("grad_norm/vggbn/"):
  x = torch.load("grad_norm/vggbn/"+filename)
  grad_norm_array_vggbn.append(x.item())

def calculate_percentiles(grad_norm_array):
  i = 5
  gradNorm_percentiles = []
  percentile_array = []
  while i < 100:
    gradNorm_percentiles.append(np.percentile(grad_norm_array, i))
    percentile_array.append(i)
    i += 5
  return [gradNorm_percentiles,percentile_array]

# Plots vgg graph
result_vgg = calculate_percentiles(grad_norm_array_vgg)
plt.plot(result_vgg[1], result_vgg[0])
plt.ylabel('Gradient Norms')
plt.xlabel('Percentile')
plt.suptitle('VGG')
plt.show()

# Plots vggbn graph
result_vggbn = calculate_percentiles(grad_norm_array_vggbn)
plt.plot(result_vggbn[1], result_vggbn[0])
plt.ylabel('Gradient Norms')
plt.xlabel('Percentile')
plt.suptitle('VGGBN')
plt.show()

#VGGbn has higher gradient norms than that of VGG due to batch normalisation
