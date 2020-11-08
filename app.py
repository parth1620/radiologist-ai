import torch
import streamlit as st
from torch import nn
from torchvision import models
from torchvision import transforms as T
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import cv2
from plotly.subplots import make_subplots


st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
logo = Image.open('Logo.png')
st.image(logo,width = 130)
st.title("Radiologist Assistant")

uploaded_file = st.file_uploader("Choose an image...",type="png")

pathology_list = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

class DenseNet121(nn.Module):

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

model = DenseNet121(14)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('CheXNet.pt',map_location = torch.device('cpu')))

if uploaded_file:

    image = Image.open(uploaded_file).convert('RGB')
    image_copy = image
    st.image(image,use_column_width=True)

    data_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    image = data_transforms(image)
    image = image.unsqueeze(0)

    model.eval()

    ps = model(image)
    ps = ps.data.numpy()
    ps = 100*ps
    ps = ps.tolist()[0]

    layout = {'yaxis':  {'range': [1, 100]}}

    fig = go.Figure([go.Bar(x=pathology_list, y=ps)],layout)
    
    st.plotly_chart(fig)

    class FeatureExtractor():

        def __init__(self, model, target_layers):
            self.model = model
            self.target_layers = target_layers
            self.gradients = []

        def save_gradient(self, grad):
            self.gradients.append(grad)

        def __call__(self, x):
            outputs = []
            self.gradients = []
            for name, module in self.model.module.densenet121._modules.items():
                x = module(x)
                if name in self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
                    x = x.mean([2,3])
            
            return outputs, x

    class ModelOutputs():

        def __init__(self, model, target_layers):
            self.model = model
            self.feature_extractor = FeatureExtractor(self.model, target_layers)

        def get_gradients(self):
            return self.feature_extractor.gradients

        def __call__(self, x):
            target_activations = []
            for name, module in self.model._modules.items():
            
                target_activations, x = self.feature_extractor(x)
            
            return target_activations, x

    
    class GradCam:

        def __init__(self,model,target_layer_names,use_cuda = False):

            self.model = model
            self.model.eval()
            self.cuda = use_cuda

            self.extractor = ModelOutputs(self.model,target_layer_names)

        def forward(self,input_img):
            return self.model(input_img)

        def __call__(self,input_img,index = None):

            if self.cuda:
                features, output = self.extractor(input_img.cuda())
            else:
                features, output = self.extractor(input_img)

            ps = output[0][index].requires_grad_(True)
            
            self.model.zero_grad()
            ps.backward(retain_graph = True)

            
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
            print(grads_val.shape)

            target = features[-1]
            target = target.cpu().data.numpy()[0,:]

            weights = np.mean(grads_val, axis = (2,3))[0,:]
            cam = np.zeros(target.shape[1:], dtype = np.float32)
            print(cam.shape)
            print(weights.shape)

            for i,w in enumerate(weights):
                cam += w * target[i, : ,:]

            print(cam.shape)
            
            cam = np.maximum(cam,0)
            
            return cam

    grad_cam = GradCam(model = model,target_layer_names='features',use_cuda = False)

    img_class = []
    
    print(len(image))

    label = st.selectbox('Select Class for GradCam',pathology_list)

    num = pathology_list.index(label)

    cam = grad_cam(image,num)
    cam = cv2.resize(cam, image_copy.size,cv2.INTER_NEAREST)
    cam = cam/np.max(cam)

    plt.imshow(image_copy)
    plt.imshow(cam, cmap='magma', alpha=0.5)
    st.pyplot()



    







    

