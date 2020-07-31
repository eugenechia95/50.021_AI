import torch
import torch.nn as nn
from imgnetdatastuff import *
from guidedbpcodehelpers import *
from torchvision import transforms

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # input, = ctx.saved_tensors
        # grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # return grad_input

        grad_input = grad_output.clone()
        if grad_input < 0:
          return 0
        return grad_input

class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        # return MyReLU.apply(input, self.weight, self.bias)
        return MyReLU.apply(input)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataloader = dataset_imagenetvalpart("imgnet500", "ILSVRC2012_bbox_val_v3/val", 'synset_words.txt', 250)
item1 = dataloader.__getitem__(1)
model1 = Linear(5, 2)
# print(model1.input_features)
modelvgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
# print(modelvgg)
# print(modelvgg)
# print(hasattr(modelvgg.features[27],"ReLU"))
reluLayers = ['1','3','6','8','11','13','15','18','20','22','25','27','29']
for i in reluLayers:
  setbyname(modelvgg.features, i, model1)

# print(modelvgg)

# input_tensor = preprocess(item1['image']).unsqueeze(0)
# output = modelvgg(input_tensor)
# loss_fn = nn.CrossEntropyLoss()
# target = torch.tensor([3], dtype=torch.long)
# err = loss_fn(output[-1], target)
# err.backward()
