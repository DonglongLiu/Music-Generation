from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from torch.utils.data import Dataset
import json
import os
import math 
import random
import warnings
from module import LMD


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
    
class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        # _ts = torch.tensor([self.n_T]).view(x.shape[0],).to(self.device)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None] * x
            + self.sqrtmab[_ts, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # print('after add noise:', x_t)
        noise_pred = self.nn_model(x_t, _ts, c)
        
        return self.loss_mse(noise, noise_pred)

    def sample(self, n_sample, size, test_lyrics, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = test_lyrics 

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i]).to(device)
            t_is = t_is.repeat(n_sample,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps1 = self.nn_model(x_i, t_is, c_i, False)
            eps2 = self.nn_model(x_i, t_is, c_i)
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i.transpose(-1, -2), x_i_store

def get_train_data():
    with open('./data_test_49/lyrics_equal_1315.lst', 'r') as f:
        lyrics = json.loads(f.read())
    with open('./data_test_49/pitches_PAD_equal_1315.lst', 'r') as f:
        pitches = json.loads(f.read())
    with open('./data_test_49/durations_PAD_equal_1315.lst', 'r') as f:
        durations = json.loads(f.read())
    with open('./duration_vocab.dict', 'r') as f:
        duration_vocab = json.loads(f.read())
        duration_vocab = {float(i): v for i, v in duration_vocab.items()}
    
    duration_tokens = [[duration_vocab[j] for j in i] for i in durations]
    token_durations = {id: number for id, number in enumerate(duration_vocab)}
        
    pds = torch.tensor([pitch_duration for pitch_duration in zip(pitches, duration_tokens)], dtype=torch.float32).transpose(-1,-2)

    class PitchDuration_Lyrics_Dataset(Dataset):

        def __init__(self, x,y):
            self.x = x
            self.y = y
            self.idx = list()
            for item in x:
                self.idx.append(item)
            pass

        def __getitem__(self, index):
            input_data = self.idx[index]
            target = self.y[index]
            return input_data, target

        def __len__(self):
            return len(self.idx)

    datasets = PitchDuration_Lyrics_Dataset(pds, lyrics)  # 初始化
    
    return datasets

def train():

    # hardcoding these here
    music_size = (20, 2) # L, D
    n_epoch = 20
    batch_size = 16
    n_T = 500 # 500
    device = "cuda:0"
    lrate = 1e-4

    save_model = True
    save_tag = 'new'
    save_dir = './model_test_1/model1/'

    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    n_sample = 1

    input_size=20
    hidden_size=512
    depth=12
    num_heads=8
    mlp_ratio=4.0

    denoise_model = LMD(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        depth=depth,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        )
    ddpm = DDPM(nn_model=denoise_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.)
    ddpm.load_state_dict(torch.load('./model_test_1/model1/model_new.pth'))
    ddpm.eval()
    ddpm.to(device)

    # dataloader = DataLoader(get_train_data(), batch_size=batch_size, shuffle=True)
    # optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    # for ep in range(n_epoch):
    #     print(f'epoch {ep}')
    #     ddpm.train()
    #     # linear lrate decay
    #     optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

    #     pbar = tqdm(dataloader)

    #     for x, c in pbar:
    #         optim.zero_grad()
    #         x = x.to(torch.float32).to(device)
    #         loss = ddpm(x, c)
    #         loss.backward()
    #         loss_ema = loss.item()
    #         pbar.set_description(f"loss: {loss_ema:.4f}")
    #         optim.step()
        
        
    #     # optionally save model
    # if save_model:
    #     torch.save(ddpm.state_dict(), save_dir + f"model_{save_tag}.pth")
    #     print('saved model at ' + save_dir + f"model_{save_tag}.pth")

    evaluate(ddpm, n_sample, music_size, device, ws_test)


def evaluate(ddpm_model, n_sample, music_size, device, ws_test):
    ddpm_model.eval()
    with torch.no_grad():
        #读进来n_sample个歌词
        test_lyrics = []
        real_pitches = []
        real_durations = []
        with open('./data_test_1/test_model/durations_PAD_equal_55.lst', 'r') as f:
            real_durations = json.loads(f.read())[:n_sample]
        with open('./data_test_1/test_model/pitches_PAD_equal_55.lst', 'r') as f:
            real_pitches = json.loads(f.read())[:n_sample]
        with open('./data_test_1/test_model/lyrics_equal_55.lst', 'r') as f:
            test_lyrics = json.loads(f.read())[:n_sample]
        with open('./duration_vocab.dict', 'r') as f:
            duration_vocab = json.loads(f.read())
            duration_vocab = {float(i): v for i, v in duration_vocab.items()}
            token_durations = {id: number for id, number in enumerate(duration_vocab)}
        
        # 开始采样，看看跟原来差距如何
        for w_i, w in enumerate(ws_test):
            print('sample on the w =',w)
            x_gen, x_gen_store = ddpm_model.sample(n_sample, music_size, test_lyrics, device, guide_w=w)

            
            with open('./data_test_1/test_result/test_result_0/w_' + str(w) + '_sampleNum_' + str(n_sample) + '.res', 'w') as f:
                for i in range(n_sample):
                    pitches_cur_sample = [round(int(pitch), 1) for pitch in x_gen[i][0].cpu().numpy().tolist()]
                    durations_cur_sample = [token_durations.get(int(duration), 0) for duration in x_gen[i][1].cpu().numpy().tolist()]

                    f.write('sample ' + str(i + 1) + '\n')
                    f.write('lyrics: ' + ' '.join(test_lyrics[i]) + '\n')
                    f.write('real pitches and durations:\n')
                    f.write('\t' + str(real_pitches[i]) + '\n')
                    f.write('\t' + str(real_durations[i]) + '\n')
                    f.write('generative pitches and durations:\n')
                    f.write('\t' + str(pitches_cur_sample) + '\n')
                    f.write('\t' + str(durations_cur_sample) + '\n')
                    f.write('-------------------------------------------------------------------------------------\n\n\n')
                    
                    
                

train()
