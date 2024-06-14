from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torch.nn.parallel import DataParallel

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import timm
import vit
import one_load
import two_load
import three_load
import four_load

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark = True
ResultSave_path='record_meta_csiq.txt'

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass


class ImageRatingsDataset2(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('L')
            if im.mode == 'P':
                im = im.convert('L')

            img_dct = cv2.dct(np.array(im, np.float32))  #get dct image

            image = np.asarray(img_dct)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}

class Normalize2(object):
    def __init__(self):
        self.amplitude_scaling_factor = 1.0  # Adjust this based on your requirements
        self.mean_amplitude = 0.0  # Adjust this based on your data
        self.std_amplitude = 1.0    # Adjust this based on your data

    def __call__(self, sample):
        frequency_domain_image, rating = sample['image'], sample['rating']
        amplitude_spectrum = np.abs(frequency_domain_image)

        # Normalize amplitude spectrum
        amplitude_spectrum = (amplitude_spectrum - self.mean_amplitude) / self.std_amplitude
        amplitude_spectrum *= self.amplitude_scaling_factor

        # Combine the normalized amplitude spectrum with the original phase spectrum
        image = amplitude_spectrum * np.exp(1j * np.angle(frequency_domain_image))
     
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}

class ToTensor2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image[np.newaxis, :, :]   #make new line for grey image
        #print(image.shape)
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}


class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)              #add norm
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        x = self.fc1(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        #print(out.shape)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out


class Net(nn.Module):
    def __init__(self, net1, net3, net4, vit, linear):
        super(Net, self).__init__()
        self.Net1 = net1
        #self.Net2 = net2
        self.Net3 = net3
        self.Net5 = net4
        self.Linear = linear
        self.Vit = vit
        
        self.con = nn.Conv2d(384, 768, kernel_size=1, stride=1, bias=False)      

    def feature_genetic(self, x):
        x = rearrange(x, 'b c h w  -> b c (h w) ', h=16, w=16)
        num_splits = 512
        num_channels = x.size(1)
        # randomly split the index of channels
        channel_indices = torch.randperm(num_channels)[:num_splits]
        #new_random_indices = torch.randperm(512)
        #channel_indices = channel_num[new_random_indices]
        
        split_tensors = []
        split_son = []
        split_son_tensors = []
        count_tensors = []
        
        for i in range(0, num_splits, 4):    #split 128 groups and each has 4 channels
            channel_index1 = channel_indices[i]
            channel_index2 = channel_indices[i+1]
            channel_index3 = channel_indices[i+2]
            channel_index4 = channel_indices[i+3]
            split1 = x[:, channel_index1, :]
            split_son.append(split1)
            split2 = x[:, channel_index2, :]
            split_son.append(split2)
            split3 = x[:, channel_index3, :]
            split_son.append(split3)
            split4 = x[:, channel_index4, :]
            split_son.append(split4)
            split = torch.stack(split_son, dim=1)
            split_son.clear()
            #print(split.shape)
            split_tensors.append(split)
        
        for i, son_tensor in enumerate(split_tensors):
            num_c = son_tensor.size(1)
            for i in range(256):
                random_integer = np.random.randint(0, num_c)
                i_son_tensor = son_tensor[0:,random_integer,i]
                #print(i_son_tensor.shape)
                split_son_tensors.append(i_son_tensor)
            merged_tensor = torch.stack(split_son_tensors, dim=1)
            #print(merged_tensor.shape)
            count_tensors.append(merged_tensor)
            split_son_tensors.clear()
    
        out_tensor = torch.stack(count_tensors, dim=1)
        #print(out_tensor.shape)

        out = rearrange(out_tensor, 'b c (h w)  -> b c h w ', h=16, w=16)
        #print(x.shape)
        crop_size = (14, 14)
        cropped_tensors = []
        for i in range(128):
            random_cropindex_h = np.random.randint(0, 2)
            random_cropindex_w = np.random.randint(0, 2)
            cropped_channel = out[:, i, random_cropindex_h : random_cropindex_h+14, random_cropindex_w : random_cropindex_w+14]
            #print(cropped_channel.shape)
            cropped_tensors.append(cropped_channel)
        
        out = torch.stack(cropped_tensors, dim=1)
        return out
        
        
    
    def forward(self, x1, x3, x5):
       
        x1 = self.Net1(x1)
        #x2 = self.CONV1(x2)
        #x2 = self.Net2(x2)
        x3 = self.Net3(x3)
        #x4 = self.CONV2(x4)
        #x4 = self.Net1(x4)
        x5 = self.Net5(x5)    #512,16,16
        
        x1 = self.feature_genetic(x1)
        #x2 = self.feature_genetic(x2)
        x3 = self.feature_genetic(x3)
        x5 = self.feature_genetic(x5)  #128,14,14
        
        features2 = torch.cat((x1, x3, x5), dim=1) #384,14,14
        #features2 = self.linc(features2)
        #print(features2.shape)
        #features2 = rearrange(features2, 'b (h w) c -> b c h w', h=14, w=14)  #512*14*14
        features2 = self.con(features2)  #768*14*14
        #print(features2.shape)
 
        
        #print(x1.shape)
        #features1 = torch.cat((x1, x2, x3, x4), dim=1)
        #print('features1', features1.shape)
        #features1 = rearrange(features1, ' b (h w) c -> b c h w', h=14, w=14)      #(,512,14,14)
        #print('features1', features1.shape)

        #add_features = self.add_conv(features1)
        #features1 = self.sample1(features1)
        #print('features1', features1.shape)
        #add_features = rearrange(add_features, 'b c h w  -> b (h w) c', h=7, w=7)
        #print('a_features1', add_features.shape)
        
        features =  self.Vit(features2)  
        #print(features.shape)
        out = self.Linear(features)
        return out
    
           
def computeSpearman(dataloader_valid1, dataloader_valid3, dataloader_valid5, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        #total_iterations = len(dataloader_valid1)
        for data1, data3, data5 in zip(dataloader_valid1, dataloader_valid3, dataloader_valid5):
            inputs1 = data1['image']
            batch_size1 = inputs1.size()[0]
            labels1 = data1['rating'].view(batch_size1, -1)
            # labels = labels / 10.0
            #inputs2 = data2['image']
            #batch_size2 = inputs2.size()[0]
            #labels2 = data2['rating'].view(batch_size2, -1)
            inputs3 = data3['image']
            #inputs4 = data4['image']
            inputs5 = data5['image']

            if use_gpu:
                try:
                    inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                    #inputs2 = Variable(inputs2.float().cuda())
                    inputs3 = Variable(inputs3.float().cuda())
                    #inputs4 = Variable(inputs4.float().cuda())
                    inputs5 = Variable(inputs5.float().cuda())
                except:
                    print(inputs1, labels1, inputs3, inputs5)
            else:
                inputs1, labels1 = Variable(inputs1), Variable(labels1)
                #inputs2 = Variable(inputs2)
                inputs3 = Variable(inputs3)
                #inputs4 = Variable(inputs4)
                inputs5 = Variable(inputs5)

            outputs_a = model(inputs1, inputs3, inputs5)
            ratings.append(labels1.float())
            predictions.append(outputs_a.float())

    ratings_i = np.vstack([r.cpu().numpy() for r in ratings])
    predictions_i = np.vstack([p.cpu().numpy() for p in predictions])
    #ratings_i = np.vstack(ratings)
    #predictions_i = np.vstack(predictions)
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a,b)[0]
    return sp, pl

def finetune_model():
    epochs = 35
    srocc_l = []
    plcc_l = []
    epoch_record = []
    best_srocc = 0
    print('=============Saving Finetuned Prior Model===========')
    data_dir = os.path.join('/home/user/DFST-master/csiq/')
    images = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_score.csv'), sep=',')
    images_fold = "/home/user/DFST-master/csiq/"
    if not os.path.exists(images_fold):
        os.makedirs(images_fold)
    for i in range(10):
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print(i,file=f1)

        print('\n')
        print('--------- The %2d rank trian-test (24epochs) ----------' % i )
        
        images_train, images_test = train_test_split(images, train_size = 0.8)

        train_path = images_fold + "train_image" + ".csv"
        test_path = images_fold + "test_image" + ".csv"
        images_train.to_csv(train_path, sep=',', index=False)
        images_test.to_csv(test_path, sep=',', index=False)
        

        #model = torch.load('model_IQA/TID2013_IQA_Meta_resnet18-1.pt')

        net_1 = one_load.densenetnew(pretrained=False)
        #net_2 = two_load.densenetnew(pretrained=False)
        net_3 = three_load.densenetnew(pretrained=False)
        net_4 = four_load.densenetnew(pretrained=False)
        #net_5 = five_load.densenetnew(pretrained=False)

        '''
        densenet_model = models.densenet121(pretrained = True)
        state_dict = densenet_model.features.state_dict()

        for name in list(state_dict.keys()):
            if name.startswith('denseblock4.'):
                del state_dict[name]
            if name.startswith('norm5.'):
                del state_dict[name]
        #print(list(state_dict.keys()))
        net_1.features.load_state_dict(state_dict)
        net_2.features.load_state_dict(state_dict)
        net_3.features.load_state_dict(state_dict)
        #net_4.features.load_state_dict(state_dict)
        net_4.features.load_state_dict(state_dict)
        '''
        VIT = vit.VisionTransformer()  
        #state_dict = vit_model.state_dict()
        #VIT.load_state_dict(state_dict)                
        l_net = BaselineModel1(1, 0.5, 1000)  
        #net_1 = models.densenet121(pretrained = True)
        model = Net(net1 = net_1, net3 = net_3, net4 = net_4, vit = VIT, linear = l_net)
        
        new_state_dict = {}
        state_dict = torch.load('model_IQA/TID2013_KADID10K_4_1.pt')
        
        pretrained_cfg_overlay = {'file': r"/home/user/use_trans/pytorch_model.bin"}
        vit_model = timm.create_model('vit_base_patch16_224', pretrained_cfg_overlay = pretrained_cfg_overlay ,pretrained=True)
        state_dict_vit = vit_model.state_dict()
        
        for key, value in state_dict.items():
            if not key.startswith('module.Vit'):
                new_state_dict[key] = value   
                
        for key, value in state_dict_vit.items():
            new_key = "module.Vit." + key
            new_state_dict[new_key] = value
                
        torch.save(new_state_dict, 'model_IQA/new_model_csiq.pt')
        
        
        '''
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")
        #model.load_state_dict(torch.load('model_IQA/TID2013_KADID10K_IQA_Meta_densenet_newload.pt'))
        #model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        '''

        for m in model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
             
        '''
        for param in model.Net1.parameters():
            param.requires_grad = False
        '''
        
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4,  weight_decay=0)
        model.cuda()      
        model = DataParallel(model)
        model.load_state_dict(torch.load('model_IQA/new_model.pt'))
        #model.load_state_dict(torch.load('model_IQA/TID2013_KADID10K_4_1.pt'))
        
        spearman = 0
        for epoch in range(epochs):
            
            #start_time = time.time()  # 记录当前时间
            
            optimizer = exp_lr_scheduler(optimizer, epoch)
            count = 0

            if epoch == 0:
                dataloader_valid1 = load_data('train1')
                #dataloader_valid2 = load_data('train2')
                dataloader_valid3 = load_data('train3')
                #dataloader_valid4 = load_data('train4')
                dataloader_valid5 = load_data('train5')

                model.eval()

                sp = computeSpearman(dataloader_valid1, dataloader_valid3, dataloader_valid5, model)[0]
                if sp > spearman:
                    spearman = sp
                print('no train srocc {:4f}'.format(sp))

            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train1 = load_data('train1')
            #dataloader_train2 = load_data('train2')
            dataloader_train3 = load_data('train3')
            #dataloader_train4 = load_data('train4')
            dataloader_train5 = load_data('train5')
            model.train()  # Set model to training mode
            running_loss = 0.0
            #total_iterations = len(dataloader_train1)
            for data1, data3, data5 in zip(dataloader_train1, dataloader_train3, dataloader_train5):
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating'].view(batch_size1, -1)
                #print('input1', inputs1)
                # labels = labels / 10.0  
                #labels2 = data2['rating'].view(batch_size2, -1)
                #print('input2', inputs2)
                inputs3 = data3['image']
                inputs5 = data5['image']

                if use_gpu:
                    try:
                        inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                        inputs3 = Variable(inputs3.float().cuda())
                        inputs5 = Variable(inputs5.float().cuda())
                    except:
                        print(inputs1, labels1, inputs3, inputs5)
                else:
                    inputs1, labels1 = Variable(inputs1), Variable(labels1)
                    inputs3 = Variable(inputs3)
                    inputs5 = Variable(inputs5)
                    
                start_time = time.time()  # 记录当前时间

                optimizer.zero_grad()
                outputs = model(inputs1, inputs3, inputs5)
                loss = criterion(outputs, labels1)
                loss.backward()
                optimizer.step()
                
                #print('t  e  s  t %.8f' %loss.item())
                try:
                    running_loss += loss.item()

                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                count += 1

            epoch_loss = running_loss / count
            epoch_record.append(epoch_loss)
            print(' The %2d epoch : current loss = %.8f ' % (epoch,epoch_loss))

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid1 = load_data('test1')
            dataloader_valid3 = load_data('test3')
            dataloader_valid5 = load_data('test5')

            model.eval()

            sp, pl = computeSpearman(dataloader_valid1, dataloader_valid3, dataloader_valid5, model)
            if sp > spearman:
                spearman = sp
                plcc=pl
            if sp > best_srocc:
                best_srocc = sp
                print('=====Prior model saved===Srocc:%f========'%best_srocc)
                best_model = copy.deepcopy(model)
                torch.save(best_model.cuda(),'model_IQA/dct.pt')
                
            end_time = time.time()  # 记录当前时间
            epoch_time = end_time - start_time  # 计算当前epoch所需的时间
            num_epochs = 35

            print('Validation Results - Epoch: {:2d}, PLCC: {:4f}, SROCC: {:4f}, '
                  'best SROCC: {:4f}'.format(epoch, pl, sp, spearman))
            print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f} seconds")

        srocc_l.append(spearman)
        plcc_l.append(pl)
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print('PLCC: {:4f}, SROCC: {:4f}'.format(plcc, spearman),file=f1)

    mean_srocc = sum(srocc_l)/len(srocc_l)
    mean_plcc = sum(plcc_l)/len(plcc_l)
    print('PLCC & SROCC', mean_srocc, mean_plcc)

    '''
    epoch_count = 0
    f = open('loss_record.txt','w')
    for line in epoch_record:
        epoch_record += 1
        f.write('epoch' + epoch_count + line + '\n')
        if epoch_record == 100:
            epoch_record = 0
    f.save()
    f.close()
    '''
    # ind = 'Results/LIVEWILD'
    # file = pd.DataFrame(columns=[ind], data=srocc_l)
    # file.to_csv(ind+'.csv')
    # print('average srocc {:4f}'.format(np.mean(srocc_l)))

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.8**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train'):

    meta_num = 24
    data_dir = os.path.join('/home/user/DFST-master/csiq/')
    train_path = os.path.join(data_dir,  'train_image.csv')
    test_path = os.path.join(data_dir,  'test_image.csv')

    output_size = (256, 256)
    transformed_dataset_train1 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/csiq/salient_image/',
                                                    transform=transforms.Compose([Rescale(output_size=(300, 300)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_train2 = ImageRatingsDataset2(csv_file=train_path,
                                                    root_dir='/home/user/data/csiq/salient_image/',
                                                    transform=transforms.Compose([Rescale(output_size=(300, 300)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(output_size=output_size),
                                                                                  Normalize2(),
                                                                                  ToTensor2(),
                                                                                  ]))
    transformed_dataset_train3 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/csiq/non_salient_image/',
                                                    transform=transforms.Compose([Rescale(output_size=(300, 300)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_train4 = ImageRatingsDataset2(csv_file=train_path,
                                                    root_dir='/home/user/data/csiq/non_salient_image/',
                                                    transform=transforms.Compose([Rescale(output_size=(300, 300)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(output_size=output_size),
                                                                                  Normalize2(),
                                                                                  ToTensor2(),
                                                                                  ]))
    transformed_dataset_train5 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/csiq/image/',
                                                    transform=transforms.Compose([Rescale(output_size=(300, 300)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid1 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/csiq/salient_image/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid2 = ImageRatingsDataset2(csv_file=test_path,
                                                    root_dir='/home/user/data/csiq/salient_image/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  Normalize2(),
                                                                                  ToTensor2(),
                                                                                  ]))
    transformed_dataset_valid3 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/csiq/non_salient_image/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid4 = ImageRatingsDataset2(csv_file=test_path,
                                                    root_dir='/home/user/data/csiq/non_salient_image/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  Normalize2(),
                                                                                  ToTensor2(),
                                                                                  ]))
    transformed_dataset_valid5 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/csiq/image/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    
    bsize = meta_num

    if mod == 'train1':
        dataloader = DataLoader(transformed_dataset_train1, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'train2':
        dataloader = DataLoader(transformed_dataset_train2, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'train3':
        dataloader = DataLoader(transformed_dataset_train3, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'train4':
        dataloader = DataLoader(transformed_dataset_train4, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'train5':
        dataloader = DataLoader(transformed_dataset_train5, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'test1':
        dataloader = DataLoader(transformed_dataset_valid1, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'test2':
        dataloader = DataLoader(transformed_dataset_valid2, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'test3':
        dataloader = DataLoader(transformed_dataset_valid3, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'test4':
        dataloader = DataLoader(transformed_dataset_valid4, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    if mod == 'test5':
        dataloader = DataLoader(transformed_dataset_valid5, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)

    return dataloader

finetune_model()
