import torch


class SmoothLayer(torch.nn.Module):
    """Will smooth a layer in a module

    input -- torch.nn.Module
    return -- the input data convoluted with a 0.5 Å Gaussian
    """
    def __init__(self):
        super().__init__()
        self.kernel = torch.FloatTensor(
            [[[0.001223804,0.002464438,0.004768176,0.008863697,0.015830903,
               0.027165938,0.044789061,0.070949186,0.107981933,0.157900317,
               0.221841669,0.299454931,0.38837211,0.483941449,0.579383106,
               0.666449206,0.736540281,0.782085388,0.797884561,0.782085388,
               0.736540281,0.666449206,0.579383106,0.483941449,0.38837211,
               0.299454931,0.221841669,0.157900317,0.107981933,0.070949186,
               0.044789061,0.027165938,0.015830903,0.008863697,0.004768176,
               0.002464438,0.001223804]]])/10
            # width 0.5 for the Gaussian (number of points for step of 0.1 A)

    def forward(self, input):
        input.unsqueeze_(1)
        output = torch.nn.functional.conv1d(input,self.kernel,padding=18)
        return output.squeeze_(1)


class ZeroLayer(torch.nn.Module):
    """Will zero a layer in a module. This replaces the ReLu as there was
    different behaviour with and without inplace=True. Could not reproduce the 
    differences so replaced the ReLu with this and just moved forward.

    input -- torch.nn.Module
    return -- input[input<0] = 0
    """    
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input[input<0] = 0
        return input
    

class Fitting_Net2B(torch.nn.Module):
    """Creates the model denoted 2B

    input -- torch.nn.Module
    return -- predicted distance distribution
    """     
    def __init__(self, num_classes=901):
        super(Fitting_Net2B, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1, 120, kernel_size=1, dilation=1),
            ZeroLayer(),
            torch.nn.Conv1d(120, 120, kernel_size=3, dilation=2),
            ZeroLayer(),
            torch.nn.Conv1d(120, 120, kernel_size=5, dilation=4),
            ZeroLayer(),
            torch.nn.Conv1d(120, 120, kernel_size=7, dilation=8),
            ZeroLayer(),
            torch.nn.Conv1d(120, 120, kernel_size=9, dilation=16),
            ZeroLayer(),
            torch.nn.Conv1d(120, 120, kernel_size=11, dilation=1),
            ZeroLayer(),
            torch.nn.Conv1d(120, 120, kernel_size=13, dilation=1),
            ZeroLayer(),
            torch.nn.Conv1d(120, 120, kernel_size=15, dilation=1),
            ZeroLayer(),  
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(120 * 68, 8192),
            ZeroLayer(),
            torch.nn.Linear(8192, 4096),
            ZeroLayer(),
            torch.nn.Linear(4096, 4096),
            ZeroLayer(),
            torch.nn.Linear(4096, 2048),
            ZeroLayer(),
            torch.nn.Linear(2048, 2048),
            ZeroLayer(),
            torch.nn.Linear(2048, 901),
            SmoothLayer(),
            ZeroLayer(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 120 * 68)
        x = self.classifier(x)
        return x

    
def load(model_dir,model_name):
    """Load a CNN model for predicting a distance distribtuion from a Deer decay.
    
    Input arguments:
        model_dir -- the directory containing the checkpoint for the model
        model -- the number of model to use (CURRENTLY only 2B can be used)
        gpu -- set to gpu to use gpu
    
    Returns:
        model -- The
    The distance and deer data used in training and
    the four simulations termed 1A, 1B, 2A, and 2B.    
    """
    # Lock the pretrained network
    model = Fitting_Net2B()    
    checkpoint = torch.load('{}/DN{}_checkpoint.pth'.format(model_dir,model_name)) 
    model.load_state_dict(checkpoint)
    for param in model.parameters():
        param.requires_grad = False
# Just leave running on cpu for now as it runs a single trace just fine        
#    if gpu == 'gpu':
#        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#    else:
#        device = torch.device('cpu')
#    model = model.to(device)        
    return model

