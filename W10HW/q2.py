import torch
import torch.nn as nn
from imgnetdatastuff import *
from guidedbpcodehelpers import *
from torchvision import transforms

class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        return input.clamp(min=0)

    # Modified backward pass for guided backprop
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[grad_input < 0] = 0
        return grad_input

class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return MyReLU.apply(input)

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load single image
dataloader = dataset_imagenetvalpart("imgnet500", "ILSVRC2012_bbox_val_v3/val", 'synset_words.txt', 250)
item1 = dataloader.__getitem__(1)
model1 = Linear(5, 2)

# Replace all relu layers in vgg model with custom layers
modelvgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)

reluLayers = ['1','3','6','8','11','13','15','18','20','22','25','27','29']
for i in reluLayers:
  setbyname(modelvgg.features, i, model1)

print(modelvgg)

input_tensor = preprocess(item1['image']).unsqueeze(0)
output = modelvgg(input_tensor)
output[-1][-1].backward()
