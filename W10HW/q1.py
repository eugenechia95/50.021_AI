from imgnetdatastuff import *
import torch
import torch.nn as nn
from torchvision import transforms

dataloader = dataset_imagenetvalpart("imgnet500", "ILSVRC2012_bbox_val_v3/val", 'synset_words.txt', 1)

item1 = dataloader.__getitem__(0)

model1 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
model2 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(item1['image']).unsqueeze(0)
# input_tensor = preprocess(dataloader)

# print(model1)
# print(item1)
# print(input_tensor.shape)
# print(item1)

def addgradnorm(idx):

  def printgradnorm(module, grad_input, grad_output):
      print(module.__class__)
      print('Inside ' + module.__class__.__name__ + ' backward')
      print('Inside class:' + module.__class__.__name__)
      print('')
      print('grad_output: ', type(grad_output))
      print('grad_output[0]: ', type(grad_output[0]))
      print('')
      print('grad_output size:', grad_output[0].size())
      print('grad_output norm:', grad_output[0].norm())
      item_filename = item1['filename'].split("/")[-1].split(".")[0]
      filename = "%s_%s_layer%d.pt" %(item_filename, module.__class__.__name__, idx)
      print(filename)
      torch.save(grad_output[0].norm(), filename)
  return printgradnorm

model1.features[0].register_backward_hook(addgradnorm(idx=0))
model1.features[2].register_backward_hook(addgradnorm(idx=2))

output = model1(input_tensor)
loss_fn = nn.CrossEntropyLoss()
target = torch.tensor([3], dtype=torch.long)
# print(output)
err = loss_fn(output, target)
err.backward()
# print(err.backward())
# print(output[0])




# # print(model1)
# model1.register_backward_hook(printgradnorm)
