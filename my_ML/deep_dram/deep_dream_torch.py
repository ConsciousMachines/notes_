
# aleksander morgintsev has created a version of deep dream in C89 because 
# he found it impossible to recreate the old 
# deep dream algo due to the amount of software reliance issues. 
# so one way for me to do something similar is to save the inception--5h network 
# architecture and weights into De-Zero !!!


# https://github.com/ProGamerGov/pytorch-old-tensorflow-models
# https://pytorch.org/hub/pytorch_vision_inception_v3/
# https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py
# ^ seems the pytorch inception v3 has conv_black be a normal nn.Conv2D + batch_norm. so 5h has no batch_norm?

# https://www.tensorflow.org/tutorials/generative/deepdream

# https://arxiv.org/pdf/1610.02391.pdf
# https://www.kaggle.com/code/sironghuang/understanding-pytorch-hooks/notebook 



''' numerical (in)compatibility with pytorch. after just one conv they are already apart by > 0.0001
obj1 = graph.get_tensor_by_name('import/conv2d0_pre_relu/conv:0')
obj1 = graph.get_tensor_by_name('import/conv2d0_pre_relu:0')
np.random.seed(1)
img = np.array(np.random.rand(224,224,3) + 100.0).astype(np.float32)
soy = sess.run([obj1], {t_input:img})[0]
soy[0,:2,:2,0]

torch needs to subtract 117 because that's a missing op.
obj2 = graph.get_operation_by_name('sub/y')
obj4 = graph.get_operation_by_name('ExpandDims/dim')

torch code
np.random.seed(1)
img = np.array(np.random.rand(224,224,3) + 100.0 - 117.0).astype(np.float32)
_in = Variable(torch.Tensor(img.transpose([2,0,1])).unsqueeze(0), requires_grad = True)
_out = incep(_in)
soy = incep._conv2d0_pre_relu_conv
soy[0,0,:2,:2].detach().numpy()
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.is_available()

class Inception5h(nn.Module):
    def __init__(self):
        super(Inception5h, self).__init__()
        self.conv2d0_pre_relu_conv                = nn.Conv2d(3, 64, (7, 7), (2, 2))
        self.conv2d1_pre_relu_conv                = nn.Conv2d(64, 64, (1, 1), (1, 1))
        self.conv2d2_pre_relu_conv                = nn.Conv2d(64, 192, (3, 3), (1, 1))
        self.mixed3a_1x1_pre_relu_conv            = nn.Conv2d(192, 64, (1, 1), (1, 1))
        self.mixed3a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(192, 96, (1, 1), (1, 1))
        self.mixed3a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(192, 16, (1, 1), (1, 1))
        self.mixed3a_pool_reduce_pre_relu_conv    = nn.Conv2d(192, 32, (1, 1), (1, 1))
        self.mixed3a_3x3_pre_relu_conv            = nn.Conv2d(96, 128, (3, 3), (1, 1))
        self.mixed3a_5x5_pre_relu_conv            = nn.Conv2d(16, 32, (5, 5), (1, 1))
        self.mixed3b_1x1_pre_relu_conv            = nn.Conv2d(256, 128, (1, 1), (1, 1))
        self.mixed3b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(256, 128, (1, 1), (1, 1))
        self.mixed3b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(256, 32, (1, 1), (1, 1))
        self.mixed3b_pool_reduce_pre_relu_conv    = nn.Conv2d(256, 64, (1, 1), (1, 1))
        self.mixed3b_3x3_pre_relu_conv            = nn.Conv2d(128, 192, (3, 3), (1, 1))
        self.mixed3b_5x5_pre_relu_conv            = nn.Conv2d(32, 96, (5, 5), (1, 1))
        self.mixed4a_1x1_pre_relu_conv            = nn.Conv2d(480, 192, (1, 1), (1, 1))
        self.mixed4a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(480, 96, (1, 1), (1, 1))
        self.mixed4a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(480, 16, (1, 1), (1, 1))
        self.mixed4a_pool_reduce_pre_relu_conv    = nn.Conv2d(480, 64, (1, 1), (1, 1))
        self.mixed4a_3x3_pre_relu_conv            = nn.Conv2d(96, 204, (3, 3), (1, 1))
        self.mixed4a_5x5_pre_relu_conv            = nn.Conv2d(16, 48, (5, 5), (1, 1))
        self.mixed4b_1x1_pre_relu_conv            = nn.Conv2d(508, 160, (1, 1), (1, 1))
        self.mixed4b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(508, 112, (1, 1), (1, 1))
        self.mixed4b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(508, 24, (1, 1), (1, 1))
        self.mixed4b_pool_reduce_pre_relu_conv    = nn.Conv2d(508, 64, (1, 1), (1, 1))
        self.mixed4b_3x3_pre_relu_conv            = nn.Conv2d(112, 224, (3, 3), (1, 1))
        self.mixed4b_5x5_pre_relu_conv            = nn.Conv2d(24, 64, (5, 5), (1, 1))
        self.mixed4c_1x1_pre_relu_conv            = nn.Conv2d(512, 128, (1, 1), (1, 1))
        self.mixed4c_3x3_bottleneck_pre_relu_conv = nn.Conv2d(512, 128, (1, 1), (1, 1))
        self.mixed4c_5x5_bottleneck_pre_relu_conv = nn.Conv2d(512, 24, (1, 1), (1, 1))
        self.mixed4c_pool_reduce_pre_relu_conv    = nn.Conv2d(512, 64, (1, 1), (1, 1))
        self.mixed4c_3x3_pre_relu_conv            = nn.Conv2d(128, 256, (3, 3), (1, 1))
        self.mixed4c_5x5_pre_relu_conv            = nn.Conv2d(24, 64, (5, 5), (1, 1))
        self.mixed4d_1x1_pre_relu_conv            = nn.Conv2d(512, 112, (1, 1), (1, 1))
        self.mixed4d_3x3_bottleneck_pre_relu_conv = nn.Conv2d(512, 144, (1, 1), (1, 1))
        self.mixed4d_5x5_bottleneck_pre_relu_conv = nn.Conv2d(512, 32, (1, 1), (1, 1))
        self.mixed4d_pool_reduce_pre_relu_conv    = nn.Conv2d(512, 64, (1, 1), (1, 1))
        self.mixed4d_3x3_pre_relu_conv            = nn.Conv2d(144, 288, (3, 3), (1, 1))
        self.mixed4d_5x5_pre_relu_conv            = nn.Conv2d(32, 64, (5, 5), (1, 1))
        self.mixed4e_1x1_pre_relu_conv            = nn.Conv2d(528, 256, (1, 1), (1, 1))
        self.mixed4e_3x3_bottleneck_pre_relu_conv = nn.Conv2d(528, 160, (1, 1), (1, 1))
        self.mixed4e_5x5_bottleneck_pre_relu_conv = nn.Conv2d(528, 32, (1, 1), (1, 1))
        self.mixed4e_pool_reduce_pre_relu_conv    = nn.Conv2d(528, 128, (1, 1), (1, 1))
        self.mixed4e_3x3_pre_relu_conv            = nn.Conv2d(160, 320, (3, 3), (1, 1))
        self.mixed4e_5x5_pre_relu_conv            = nn.Conv2d(32, 128, (5, 5), (1, 1))
        self.mixed5a_1x1_pre_relu_conv            = nn.Conv2d(832, 256, (1, 1), (1, 1))
        self.mixed5a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(832, 160, (1, 1), (1, 1))
        self.mixed5a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(832, 48, (1, 1), (1, 1))
        self.mixed5a_pool_reduce_pre_relu_conv    = nn.Conv2d(832, 128, (1, 1), (1, 1))
        self.mixed5a_3x3_pre_relu_conv            = nn.Conv2d(160, 320, (3, 3), (1, 1))
        self.mixed5a_5x5_pre_relu_conv            = nn.Conv2d(48, 128, (5, 5), (1, 1))
        self.mixed5b_1x1_pre_relu_conv            = nn.Conv2d(832, 384, (1, 1), (1, 1))
        self.mixed5b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(832, 192, (1, 1), (1, 1))
        self.mixed5b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(832, 48, (1, 1), (1, 1))
        self.mixed5b_pool_reduce_pre_relu_conv    = nn.Conv2d(832, 128, (1, 1), (1, 1))
        self.mixed5b_3x3_pre_relu_conv            = nn.Conv2d(192, 384, (3, 3), (1, 1))
        self.mixed5b_5x5_pre_relu_conv            = nn.Conv2d(48, 128, (5, 5), (1, 1))
        self.softmax2_pre_activation_matmul       = nn.Linear(in_features = 1024, out_features = 1008, bias = True)


    def forward(self, x):
        self._conv2d0_pre_relu_conv_pad                 = F.pad(x, (2, 3, 2, 3))
        self._conv2d0_pre_relu_conv                     = self.conv2d0_pre_relu_conv(self._conv2d0_pre_relu_conv_pad)
        self._conv2d0                                   = F.relu(self._conv2d0_pre_relu_conv)
        self._maxpool0_pad                              = F.pad(self._conv2d0, (0, 1, 0, 1), value=float('-inf'))
        self._maxpool0                                  = F.max_pool2d(self._maxpool0_pad, (3, 3), (2, 2), padding=0, ceil_mode=False)
        self._localresponsenorm0                        = F.local_response_norm(self._maxpool0, size=10, alpha=10 * 0.00009999999747378752, beta=0.5, k=2)
        self._conv2d1_pre_relu_conv                     = self.conv2d1_pre_relu_conv(self._localresponsenorm0)
        self._conv2d1                                   = F.relu(self._conv2d1_pre_relu_conv)
        self._conv2d2_pre_relu_conv_pad                 = F.pad(self._conv2d1, (1, 1, 1, 1))
        self._conv2d2_pre_relu_conv                     = self.conv2d2_pre_relu_conv(self._conv2d2_pre_relu_conv_pad)
        self._conv2d2                                   = F.relu(self._conv2d2_pre_relu_conv)
        self._localresponsenorm1                        = F.local_response_norm(self._conv2d2, size=10, alpha=10 * 0.00009999999747378752, beta=0.5, k=2)
        self._maxpool1_pad                              = F.pad(self._localresponsenorm1, (0, 1, 0, 1), value=float('-inf'))
        self._maxpool1                                  = F.max_pool2d(self._maxpool1_pad, (3, 3), (2, 2), padding=0, ceil_mode=False)
        self._mixed3a_1x1_pre_relu_conv                 = self.mixed3a_1x1_pre_relu_conv(self._maxpool1)
        self._mixed3a_3x3_bottleneck_pre_relu_conv      = self.mixed3a_3x3_bottleneck_pre_relu_conv(self._maxpool1)
        self._mixed3a_5x5_bottleneck_pre_relu_conv      = self.mixed3a_5x5_bottleneck_pre_relu_conv(self._maxpool1)
        self._mixed3a_pool_pad                          = F.pad(self._maxpool1, (1, 1, 1, 1), value=float('-inf'))
        self._mixed3a_pool                              = F.max_pool2d(self._mixed3a_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed3a_1x1                               = F.relu(self._mixed3a_1x1_pre_relu_conv)
        self._mixed3a_3x3_bottleneck                    = F.relu(self._mixed3a_3x3_bottleneck_pre_relu_conv)
        self._mixed3a_5x5_bottleneck                    = F.relu(self._mixed3a_5x5_bottleneck_pre_relu_conv)
        self._mixed3a_pool_reduce_pre_relu_conv         = self.mixed3a_pool_reduce_pre_relu_conv(self._mixed3a_pool)
        self._mixed3a_3x3_pre_relu_conv_pad             = F.pad(self._mixed3a_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed3a_3x3_pre_relu_conv                 = self.mixed3a_3x3_pre_relu_conv(self._mixed3a_3x3_pre_relu_conv_pad)
        self._mixed3a_5x5_pre_relu_conv_pad             = F.pad(self._mixed3a_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed3a_5x5_pre_relu_conv                 = self.mixed3a_5x5_pre_relu_conv(self._mixed3a_5x5_pre_relu_conv_pad)
        self._mixed3a_pool_reduce                       = F.relu(self._mixed3a_pool_reduce_pre_relu_conv)
        self._mixed3a_3x3                               = F.relu(self._mixed3a_3x3_pre_relu_conv)
        self._mixed3a_5x5                               = F.relu(self._mixed3a_5x5_pre_relu_conv)
        self._mixed3a                                   = torch.cat((self._mixed3a_1x1, self._mixed3a_3x3, self._mixed3a_5x5, self._mixed3a_pool_reduce), 1)
        self._mixed3b_1x1_pre_relu_conv                 = self.mixed3b_1x1_pre_relu_conv(self._mixed3a)
        self._mixed3b_3x3_bottleneck_pre_relu_conv      = self.mixed3b_3x3_bottleneck_pre_relu_conv(self._mixed3a)
        self._mixed3b_5x5_bottleneck_pre_relu_conv      = self.mixed3b_5x5_bottleneck_pre_relu_conv(self._mixed3a)
        self._mixed3b_pool_pad                          = F.pad(self._mixed3a, (1, 1, 1, 1), value=float('-inf'))
        self._mixed3b_pool                              = F.max_pool2d(self._mixed3b_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed3b_1x1                               = F.relu(self._mixed3b_1x1_pre_relu_conv)
        self._mixed3b_3x3_bottleneck                    = F.relu(self._mixed3b_3x3_bottleneck_pre_relu_conv)
        self._mixed3b_5x5_bottleneck                    = F.relu(self._mixed3b_5x5_bottleneck_pre_relu_conv)
        self._mixed3b_pool_reduce_pre_relu_conv         = self.mixed3b_pool_reduce_pre_relu_conv(self._mixed3b_pool)
        self._mixed3b_3x3_pre_relu_conv_pad             = F.pad(self._mixed3b_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed3b_3x3_pre_relu_conv                 = self.mixed3b_3x3_pre_relu_conv(self._mixed3b_3x3_pre_relu_conv_pad)
        self._mixed3b_5x5_pre_relu_conv_pad             = F.pad(self._mixed3b_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed3b_5x5_pre_relu_conv                 = self.mixed3b_5x5_pre_relu_conv(self._mixed3b_5x5_pre_relu_conv_pad)
        self._mixed3b_pool_reduce                       = F.relu(self._mixed3b_pool_reduce_pre_relu_conv)
        self._mixed3b_3x3                               = F.relu(self._mixed3b_3x3_pre_relu_conv)
        self._mixed3b_5x5                               = F.relu(self._mixed3b_5x5_pre_relu_conv)
        self._mixed3b                                   = torch.cat((self._mixed3b_1x1, self._mixed3b_3x3, self._mixed3b_5x5, self._mixed3b_pool_reduce), 1)
        self._maxpool4_pad                              = F.pad(self._mixed3b, (0, 1, 0, 1), value=float('-inf'))
        self._maxpool4                                  = F.max_pool2d(self._maxpool4_pad, (3, 3), (2, 2), padding=0, ceil_mode=False)
        self._mixed4a_1x1_pre_relu_conv                 = self.mixed4a_1x1_pre_relu_conv(self._maxpool4)
        self._mixed4a_3x3_bottleneck_pre_relu_conv      = self.mixed4a_3x3_bottleneck_pre_relu_conv(self._maxpool4)
        self._mixed4a_5x5_bottleneck_pre_relu_conv      = self.mixed4a_5x5_bottleneck_pre_relu_conv(self._maxpool4)
        self._mixed4a_pool_pad                          = F.pad(self._maxpool4, (1, 1, 1, 1), value=float('-inf'))
        self._mixed4a_pool                              = F.max_pool2d(self._mixed4a_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed4a_1x1                               = F.relu(self._mixed4a_1x1_pre_relu_conv)
        self._mixed4a_3x3_bottleneck                    = F.relu(self._mixed4a_3x3_bottleneck_pre_relu_conv)
        self._mixed4a_5x5_bottleneck                    = F.relu(self._mixed4a_5x5_bottleneck_pre_relu_conv)
        self._mixed4a_pool_reduce_pre_relu_conv         = self.mixed4a_pool_reduce_pre_relu_conv(self._mixed4a_pool)
        self._mixed4a_3x3_pre_relu_conv_pad             = F.pad(self._mixed4a_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed4a_3x3_pre_relu_conv                 = self.mixed4a_3x3_pre_relu_conv(self._mixed4a_3x3_pre_relu_conv_pad)
        self._mixed4a_5x5_pre_relu_conv_pad             = F.pad(self._mixed4a_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed4a_5x5_pre_relu_conv                 = self.mixed4a_5x5_pre_relu_conv(self._mixed4a_5x5_pre_relu_conv_pad)
        self._mixed4a_pool_reduce                       = F.relu(self._mixed4a_pool_reduce_pre_relu_conv)
        self._mixed4a_3x3                               = F.relu(self._mixed4a_3x3_pre_relu_conv)
        self._mixed4a_5x5                               = F.relu(self._mixed4a_5x5_pre_relu_conv)
        self._mixed4a                                   = torch.cat((self._mixed4a_1x1, self._mixed4a_3x3, self._mixed4a_5x5, self._mixed4a_pool_reduce), 1)
        self._mixed4b_1x1_pre_relu_conv                 = self.mixed4b_1x1_pre_relu_conv(self._mixed4a)
        self._mixed4b_3x3_bottleneck_pre_relu_conv      = self.mixed4b_3x3_bottleneck_pre_relu_conv(self._mixed4a)
        self._mixed4b_5x5_bottleneck_pre_relu_conv      = self.mixed4b_5x5_bottleneck_pre_relu_conv(self._mixed4a)
        self._mixed4b_pool_pad                          = F.pad(self._mixed4a, (1, 1, 1, 1), value=float('-inf'))
        self._mixed4b_pool                              = F.max_pool2d(self._mixed4b_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed4b_1x1                               = F.relu(self._mixed4b_1x1_pre_relu_conv)
        self._mixed4b_3x3_bottleneck                    = F.relu(self._mixed4b_3x3_bottleneck_pre_relu_conv)
        self._mixed4b_5x5_bottleneck                    = F.relu(self._mixed4b_5x5_bottleneck_pre_relu_conv)
        self._mixed4b_pool_reduce_pre_relu_conv         = self.mixed4b_pool_reduce_pre_relu_conv(self._mixed4b_pool)
        self._mixed4b_3x3_pre_relu_conv_pad             = F.pad(self._mixed4b_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed4b_3x3_pre_relu_conv                 = self.mixed4b_3x3_pre_relu_conv(self._mixed4b_3x3_pre_relu_conv_pad)
        self._mixed4b_5x5_pre_relu_conv_pad             = F.pad(self._mixed4b_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed4b_5x5_pre_relu_conv                 = self.mixed4b_5x5_pre_relu_conv(self._mixed4b_5x5_pre_relu_conv_pad)
        self._mixed4b_pool_reduce                       = F.relu(self._mixed4b_pool_reduce_pre_relu_conv)
        self._mixed4b_3x3                               = F.relu(self._mixed4b_3x3_pre_relu_conv)
        self._mixed4b_5x5                               = F.relu(self._mixed4b_5x5_pre_relu_conv)
        self._mixed4b                                   = torch.cat((self._mixed4b_1x1, self._mixed4b_3x3, self._mixed4b_5x5, self._mixed4b_pool_reduce), 1)
        self._mixed4c_1x1_pre_relu_conv                 = self.mixed4c_1x1_pre_relu_conv(self._mixed4b)
        self._mixed4c_3x3_bottleneck_pre_relu_conv      = self.mixed4c_3x3_bottleneck_pre_relu_conv(self._mixed4b)
        self._mixed4c_5x5_bottleneck_pre_relu_conv      = self.mixed4c_5x5_bottleneck_pre_relu_conv(self._mixed4b)
        self._mixed4c_pool_pad                          = F.pad(self._mixed4b, (1, 1, 1, 1), value=float('-inf'))
        self._mixed4c_pool                              = F.max_pool2d(self._mixed4c_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed4c_1x1                               = F.relu(self._mixed4c_1x1_pre_relu_conv)
        self._mixed4c_3x3_bottleneck                    = F.relu(self._mixed4c_3x3_bottleneck_pre_relu_conv)
        self._mixed4c_5x5_bottleneck                    = F.relu(self._mixed4c_5x5_bottleneck_pre_relu_conv)
        self._mixed4c_pool_reduce_pre_relu_conv         = self.mixed4c_pool_reduce_pre_relu_conv(self._mixed4c_pool)
        self._mixed4c_3x3_pre_relu_conv_pad             = F.pad(self._mixed4c_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed4c_3x3_pre_relu_conv                 = self.mixed4c_3x3_pre_relu_conv(self._mixed4c_3x3_pre_relu_conv_pad)
        self._mixed4c_5x5_pre_relu_conv_pad             = F.pad(self._mixed4c_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed4c_5x5_pre_relu_conv                 = self.mixed4c_5x5_pre_relu_conv(self._mixed4c_5x5_pre_relu_conv_pad)
        self._mixed4c_pool_reduce                       = F.relu(self._mixed4c_pool_reduce_pre_relu_conv)
        self._mixed4c_3x3                               = F.relu(self._mixed4c_3x3_pre_relu_conv)
        self._mixed4c_5x5                               = F.relu(self._mixed4c_5x5_pre_relu_conv)
        self._mixed4c                                   = torch.cat((self._mixed4c_1x1, self._mixed4c_3x3, self._mixed4c_5x5, self._mixed4c_pool_reduce), 1)
        self._mixed4d_1x1_pre_relu_conv                 = self.mixed4d_1x1_pre_relu_conv(self._mixed4c)
        self._mixed4d_3x3_bottleneck_pre_relu_conv      = self.mixed4d_3x3_bottleneck_pre_relu_conv(self._mixed4c)
        self._mixed4d_5x5_bottleneck_pre_relu_conv      = self.mixed4d_5x5_bottleneck_pre_relu_conv(self._mixed4c)
        self._mixed4d_pool_pad                          = F.pad(self._mixed4c, (1, 1, 1, 1), value=float('-inf'))
        self._mixed4d_pool                              = F.max_pool2d(self._mixed4d_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed4d_1x1                               = F.relu(self._mixed4d_1x1_pre_relu_conv)
        self._mixed4d_3x3_bottleneck                    = F.relu(self._mixed4d_3x3_bottleneck_pre_relu_conv)
        self._mixed4d_5x5_bottleneck                    = F.relu(self._mixed4d_5x5_bottleneck_pre_relu_conv)
        self._mixed4d_pool_reduce_pre_relu_conv         = self.mixed4d_pool_reduce_pre_relu_conv(self._mixed4d_pool)
        self._mixed4d_3x3_pre_relu_conv_pad             = F.pad(self._mixed4d_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed4d_3x3_pre_relu_conv                 = self.mixed4d_3x3_pre_relu_conv(self._mixed4d_3x3_pre_relu_conv_pad)
        self._mixed4d_5x5_pre_relu_conv_pad             = F.pad(self._mixed4d_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed4d_5x5_pre_relu_conv                 = self.mixed4d_5x5_pre_relu_conv(self._mixed4d_5x5_pre_relu_conv_pad)
        self._mixed4d_pool_reduce                       = F.relu(self._mixed4d_pool_reduce_pre_relu_conv)
        self._mixed4d_3x3                               = F.relu(self._mixed4d_3x3_pre_relu_conv)
        self._mixed4d_5x5                               = F.relu(self._mixed4d_5x5_pre_relu_conv)
        self._mixed4d                                   = torch.cat((self._mixed4d_1x1, self._mixed4d_3x3, self._mixed4d_5x5, self._mixed4d_pool_reduce), 1)
        self._mixed4e_1x1_pre_relu_conv                 = self.mixed4e_1x1_pre_relu_conv(self._mixed4d)
        self._mixed4e_3x3_bottleneck_pre_relu_conv      = self.mixed4e_3x3_bottleneck_pre_relu_conv(self._mixed4d)
        self._mixed4e_5x5_bottleneck_pre_relu_conv      = self.mixed4e_5x5_bottleneck_pre_relu_conv(self._mixed4d)
        self._mixed4e_pool_pad                          = F.pad(self._mixed4d, (1, 1, 1, 1), value=float('-inf'))
        self._mixed4e_pool                              = F.max_pool2d(self._mixed4e_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed4e_1x1                               = F.relu(self._mixed4e_1x1_pre_relu_conv)
        self._mixed4e_3x3_bottleneck                    = F.relu(self._mixed4e_3x3_bottleneck_pre_relu_conv)
        self._mixed4e_5x5_bottleneck                    = F.relu(self._mixed4e_5x5_bottleneck_pre_relu_conv)
        self._mixed4e_pool_reduce_pre_relu_conv         = self.mixed4e_pool_reduce_pre_relu_conv(self._mixed4e_pool)
        self._mixed4e_3x3_pre_relu_conv_pad             = F.pad(self._mixed4e_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed4e_3x3_pre_relu_conv                 = self.mixed4e_3x3_pre_relu_conv(self._mixed4e_3x3_pre_relu_conv_pad)
        self._mixed4e_5x5_pre_relu_conv_pad             = F.pad(self._mixed4e_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed4e_5x5_pre_relu_conv                 = self.mixed4e_5x5_pre_relu_conv(self._mixed4e_5x5_pre_relu_conv_pad)
        self._mixed4e_pool_reduce                       = F.relu(self._mixed4e_pool_reduce_pre_relu_conv)
        self._mixed4e_3x3                               = F.relu(self._mixed4e_3x3_pre_relu_conv)
        self._mixed4e_5x5                               = F.relu(self._mixed4e_5x5_pre_relu_conv)
        self._mixed4e                                   = torch.cat((self._mixed4e_1x1, self._mixed4e_3x3, self._mixed4e_5x5, self._mixed4e_pool_reduce), 1)
        self._maxpool10_pad                             = F.pad(self._mixed4e, (0, 1, 0, 1), value=float('-inf'))
        self._maxpool10                                 = F.max_pool2d(self._maxpool10_pad, (3, 3), (2, 2), padding=0, ceil_mode=False)
        self._mixed5a_1x1_pre_relu_conv                 = self.mixed5a_1x1_pre_relu_conv(self._maxpool10)
        self._mixed5a_3x3_bottleneck_pre_relu_conv      = self.mixed5a_3x3_bottleneck_pre_relu_conv(self._maxpool10)
        self._mixed5a_5x5_bottleneck_pre_relu_conv      = self.mixed5a_5x5_bottleneck_pre_relu_conv(self._maxpool10)
        self._mixed5a_pool_pad                          = F.pad(self._maxpool10, (1, 1, 1, 1), value=float('-inf'))
        self._mixed5a_pool                              = F.max_pool2d(self._mixed5a_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed5a_1x1                               = F.relu(self._mixed5a_1x1_pre_relu_conv)
        self._mixed5a_3x3_bottleneck                    = F.relu(self._mixed5a_3x3_bottleneck_pre_relu_conv)
        self._mixed5a_5x5_bottleneck                    = F.relu(self._mixed5a_5x5_bottleneck_pre_relu_conv)
        self._mixed5a_pool_reduce_pre_relu_conv         = self.mixed5a_pool_reduce_pre_relu_conv(self._mixed5a_pool)
        self._mixed5a_3x3_pre_relu_conv_pad             = F.pad(self._mixed5a_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed5a_3x3_pre_relu_conv                 = self.mixed5a_3x3_pre_relu_conv(self._mixed5a_3x3_pre_relu_conv_pad)
        self._mixed5a_5x5_pre_relu_conv_pad             = F.pad(self._mixed5a_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed5a_5x5_pre_relu_conv                 = self.mixed5a_5x5_pre_relu_conv(self._mixed5a_5x5_pre_relu_conv_pad)
        self._mixed5a_pool_reduce                       = F.relu(self._mixed5a_pool_reduce_pre_relu_conv)
        self._mixed5a_3x3                               = F.relu(self._mixed5a_3x3_pre_relu_conv)
        self._mixed5a_5x5                               = F.relu(self._mixed5a_5x5_pre_relu_conv)
        self._mixed5a                                   = torch.cat((self._mixed5a_1x1, self._mixed5a_3x3, self._mixed5a_5x5, self._mixed5a_pool_reduce), 1)
        self._mixed5b_1x1_pre_relu_conv                 = self.mixed5b_1x1_pre_relu_conv(self._mixed5a)
        self._mixed5b_3x3_bottleneck_pre_relu_conv      = self.mixed5b_3x3_bottleneck_pre_relu_conv(self._mixed5a)
        self._mixed5b_5x5_bottleneck_pre_relu_conv      = self.mixed5b_5x5_bottleneck_pre_relu_conv(self._mixed5a)
        self._mixed5b_pool_pad                          = F.pad(self._mixed5a, (1, 1, 1, 1), value=float('-inf'))
        self._mixed5b_pool                              = F.max_pool2d(self._mixed5b_pool_pad, (3, 3), (1, 1), padding=0, ceil_mode=False)
        self._mixed5b_1x1                               = F.relu(self._mixed5b_1x1_pre_relu_conv)
        self._mixed5b_3x3_bottleneck                    = F.relu(self._mixed5b_3x3_bottleneck_pre_relu_conv)
        self._mixed5b_5x5_bottleneck                    = F.relu(self._mixed5b_5x5_bottleneck_pre_relu_conv)
        self._mixed5b_pool_reduce_pre_relu_conv         = self.mixed5b_pool_reduce_pre_relu_conv(self._mixed5b_pool)
        self._mixed5b_3x3_pre_relu_conv_pad             = F.pad(self._mixed5b_3x3_bottleneck, (1, 1, 1, 1))
        self._mixed5b_3x3_pre_relu_conv                 = self.mixed5b_3x3_pre_relu_conv(self._mixed5b_3x3_pre_relu_conv_pad)
        self._mixed5b_5x5_pre_relu_conv_pad             = F.pad(self._mixed5b_5x5_bottleneck, (2, 2, 2, 2))
        self._mixed5b_5x5_pre_relu_conv                 = self.mixed5b_5x5_pre_relu_conv(self._mixed5b_5x5_pre_relu_conv_pad)
        self._mixed5b_pool_reduce                       = F.relu(self._mixed5b_pool_reduce_pre_relu_conv)
        self._mixed5b_3x3                               = F.relu(self._mixed5b_3x3_pre_relu_conv)
        self._mixed5b_5x5                               = F.relu(self._mixed5b_5x5_pre_relu_conv)
        self._mixed5b                                   = torch.cat((self._mixed5b_1x1, self._mixed5b_3x3, self._mixed5b_5x5, self._mixed5b_pool_reduce), 1)
        self._avgpool0                                  = F.avg_pool2d(self._mixed5b, (7, 7), (1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        self._avgpool0_reshape                          = torch.reshape(input = self._avgpool0, shape = (-1,1024))
        self._softmax2_pre_activation_matmul            = self.softmax2_pre_activation_matmul(self._avgpool0_reshape)
        self._softmax2                                  = F.softmax(self._softmax2_pre_activation_matmul)
        return self._softmax2



import os
from PIL import Image
from torchvision import transforms
import numpy as np
import PIL.Image as im
from torch.autograd import Variable
import torchvision




def showarray(a): # create a jpeg file from an array a and visualize it
    a = np.uint8(np.clip(a, 0, 1) * 255) # clip the values to be between 0 and 255
    im.fromarray(a).show()
    
def visstd(a, s = 0.1): # Normalize the image range for visualization
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5 # i think this is an arbitrary way to put array in the range 0,1



out_dir = r'/home/chad/Desktop/_backups/notes/my_ML/deep_dram'

_weights_path = r'/home/chad/Desktop/_backups/notes/ignore/pytorch-old-tensorflow-models-master/inception5h.pth'

incep = Inception5h()
incep.load_state_dict(torch.load(os.path.join(out_dir, _weights_path)))
incep.eval()


os.getcwd()



# sample execution (requires torchvision)
filename = os.path.join(out_dir, 'dog.jpg')
input_image = Image.open(filename)
input_image = input_image.resize((224,224))
preprocess = transforms.Compose([
    #transforms.CenterCrop(224),
    #transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
with torch.no_grad():
    output = incep(input_batch) 
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0) 

# Read the categories
categories_file = os.path.join(out_dir, r'inception5h\imagenet_comp_graph_label_strings.txt')
with open(categories_file, "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())





#   R E N D E R   N A I V E 
# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
np.random.seed(0)
img_noise = np.array(np.random.rand(224,224,3) + 100.0 - 117.0).astype(np.float32)
img = img_noise.copy()
_in = Variable(torch.Tensor(img.transpose([2,0,1])).unsqueeze(0), requires_grad = True)

for i in range(20):
    _out = incep(_in)
    t_obj = incep._mixed4d_3x3_bottleneck_pre_relu_conv[:,139,:,:]
    _loss = torch.mean(t_obj)
    _loss.backward()

    g = _in.grad
    g /= g.std() + 1e-8
    _in.data += g

    _in.grad = None
    for i in incep.parameters():
        i.grad = None

showarray(visstd(_in.squeeze().detach().numpy().transpose([1,2,0])))




#   M U L T I S C A L E

# https://stackoverflow.com/questions/58676688/how-to-resize-a-pytorch-tensor
# https://github.com/assafshocher/ResizeRight

# TODO: convert this mess to pytorch image dims

def nearest_odd(hw): # make the dims odd so conv dims are easier 
    h1, w1 = int(hw[0]), int(hw[1])
    if h1 % 2 == 1:
        h1 += 1 
    if w1 % 2 == 1:
        w1 += 1 
    return (h1, w1)

img                                 = img_noise.copy()
octave_scale                        = 1.4

for octave in range(3): # number of octaves
    if octave > 0:      # calc new height & width when scaling up by octave_sclae
        hw                          = np.float32(img.shape[:2]) * octave_scale
        __in                        = torch.Tensor(img.transpose([2,0,1])[None,:,:,:])
        img                         = F.interpolate(__in, nearest_odd(hw), mode = 'bilinear', align_corners = False).squeeze().numpy().transpose([1,2,0])
    for i in range(10):
        # compute grad of image by taking grad of small tiles. apply random shifts to image to blur boundaries between iterations
        sz                          = 512 # tile size
        h, w                        = img.shape[:2] # size of the image
        sx, sy                      = np.random.randint(sz, size=2) # random shift numbers generated
        img_shift                   = np.roll(np.roll(img, sx, 1), sy, 0) #shift the whole image. np.roll = Roll array elements along a given axis
        grad                        = np.zeros_like(img)

        for y in range(0, max(h - sz//2, sz), sz):     # NOTE: this leaves patches of the image un-gradiented.
            for x in range(0, max(w - sz//2, sz), sz): # alternative is to pad (then waste grad comp?) also the net takes in diff dims.
                sub                 = img_shift[y:y + sz,x:x + sz]           # tile of varying size: <= 512

                _in = Variable(torch.Tensor(sub.transpose([2,0,1])).unsqueeze(0), requires_grad = True)

                _out = incep(_in)
                t_obj = incep._mixed4d_3x3_bottleneck_pre_relu_conv[:,139,:,:]
                _loss = torch.mean(t_obj)
                _loss.backward()

                g                   = _in.grad.squeeze().permute([1,2,0])
                grad[y:y+sz,x:x+sz] = g                                      # assemble real grad from tile
        g                           = np.roll(np.roll(grad, -sx, 1), -sy, 0) # unroll
        g                          /= g.std() + 1e-8 # normalizing the gradient, so the same step size should work for different layers and networks
        img                        += g # update 
showarray(visstd(img))







#   L A P L A C I A N

# TODO: using scale 2.0 because torch's transpose2d doesn't let us define a specific size. 
# to circumvent this, I would need to place a bunch of if-statements to crop the output to be of agreeable size in the conv-deconv part. 

k                         = np.float32([1,4,6,4,1])                                           # used to make Gaus kernel
k                         = np.outer(k, k)                                                    # Gaus kernel
k5x5                      = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)              # some sort of Conv-Transpose Gaussian kernel. 
k5x5 = torch.Tensor(k5x5.transpose([3,2,0,1]))

def normalize_std(img):                                                            # Normalize image by making its standard deviation = 1.0
    std               = torch.sqrt(torch.mean(img**2))
    return img / torch.maximum(std, torch.Tensor([1e-10]))


def lap_grad_norm(g):
    img                       = torch.Tensor(g.transpose([2,0,1])[None,:,:,:])

    # Build Laplacian pyramid with 4 splits
    levels                    = []
    for i in range(4):
        # Split the image into lo and hi frequency components
        
        lo                = F.conv2d(img, k5x5, None, (2,2), (2,2))
        lo2 = F.conv_transpose2d(lo, k5x5 * 4.0, None, (2,2))[:,:,2:-1,2:-1] # if img always even, we have 2 extra on left and 1 extra on right
        hi                = img - lo2                                                         # the hi-freq noise, Laplacian part
        levels.append(hi)                                                                         # add Laplacian part to list
        img                   = lo                                                                # on next iteration, do the same thing to the downsampled image.
    levels.append(img)                                                                            # at the end we append the last downsampled image. 
    tlevels                   = [normalize_std(i) for i in levels[::-1]]                          # reverse order of downsampled imgs, and normalize
    # Merge Laplacian pyramid
    out                       = tlevels[0]                                                        # start with the smallest downsampled image
    for hi in tlevels[1:]:                                                                        # loop over the Laplacian hi-freq components
        out = F.conv_transpose2d(out, k5x5 * 4.0, None, (2,2))[:,:,2:-1,2:-1] + hi
    out                       = out[0,:,:,:]                                                      # unbatch
    return out.numpy().transpose([1,2,0])




octave_scale                        = 2.0

img                                 = img_noise.copy()
for octave in range(1):
    if octave > 0:
        hw                          = np.float32(img.shape[:2]) * octave_scale
        __in                        = torch.Tensor(img.transpose([2,0,1])[None,:,:,:])
        img                         = F.interpolate(__in, nearest_odd(hw), mode = 'bilinear', align_corners = False).squeeze().numpy().transpose([1,2,0])
    for i in range(100):
        # G R A D   T I L E D   F N 
        sz                          = 512
        h, w                        = img.shape[:2]
        sx, sy                      = np.random.randint(sz, size=2) 
        img_shift                   = np.roll(np.roll(img, sx, 1), sy, 0)
        grad                        = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub                 = img_shift[y:y+sz,x:x+sz] 

                _in = Variable(torch.Tensor(sub.transpose([2,0,1])).unsqueeze(0), requires_grad = True)

                _out = incep(_in)
                t_obj = incep._mixed4d_3x3_bottleneck_pre_relu_conv[:,139,:,:]
                _loss = torch.mean(t_obj)
                _loss.backward()

                g                   = _in.grad.squeeze().permute([1,2,0])
                grad[y:y+sz,x:x+sz] = g 
        g                           = np.roll(np.roll(grad, -sx, 1), -sy, 0) 
        g                           = lap_grad_norm(g)
        img                        += g
showarray(visstd(img))






