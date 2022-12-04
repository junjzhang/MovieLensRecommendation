import torch
import numpy as np
import pandas as pd
from model import GLocalNet, KernelNet
from pathlib2 import Path


def load_data(path, delimiter='\t'):
    train = np.loadtxt(path / 'movielens_100k_u1.base',
                       skiprows=0,
                       delimiter=delimiter).astype('int32')
    test = np.loadtxt(path / 'movielens_100k_u1.test',
                      skiprows=0,
                      delimiter=delimiter).astype('int32')
    total = np.concatenate((train, test), axis=0)
    n_m = np.unique(total[:, 1]).shape[0]  # num of movies
    n_u = np.unique(total[:, 0]).shape[0]  # num of users
    n_r = total.shape[0]  # num of ratings
    R = np.zeros((n_m, n_u), dtype='float32')
    for i in range(n_r):
        R[total[i, 1] - 1, total[i, 0] - 1] = total[i, 2]

    item_info = pd.read_csv(path / 'u.item',
                            sep='|',
                            header=None,
                            encoding='latin-1',
                            usecols=[1]).rename(columns={1: 'title'})
    return R, item_info


class InferenceModel:

    def __init__(self,
                 n_hid=500,
                 n_emb=4,
                 n_layers=2,
                 gk_size=3,
                 dot_scale=1,
                 data_dir=Path('movie_lens_100k')):
        n_u = 943
        n_m = 1682
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.local_net = KernelNet(n_u, n_hid, n_emb, n_layers)
        self.local_net.load_state_dict(torch.load('weights/best_pretrain.pth'))
        self.local_net.to(self.device)
        self.local_net.eval()
        kernel_net = KernelNet(n_u, n_hid, n_emb, n_layers)
        self.complete_model = GLocalNet(kernel_net, n_m, gk_size, dot_scale)
        self.complete_model.load_state_dict(torch.load('weights/best.pth'))
        self.complete_model.to(self.device)
        self.complete_model.eval()
        self.R, self.item_info = load_data(data_dir)

    def show_movie(self, movie_id):
        return self.item_info.iloc[movie_id].title

    def recommend(self, user_id, n=10):
        R = torch.tensor(self.R, device=self.device)
        with torch.no_grad():
            local_emb = self.local_net(R)
            pred = self.complete_model(R, local_emb)
        pred = pred[:, user_id - 1].cpu().numpy()
        pred[self.R[:, user_id - 1] > 0] = -np.inf
        top_n = np.argsort(pred)[-n:][::-1]
        print(f'Recommendations for user {user_id}:')
        for i, idx in enumerate(top_n):
            print(f'{i+1}. movie id:{idx+1}, {self.show_movie(idx)}')

    def rate_movie(self, user_id, movie_id, rating):
        self.R[movie_id - 1, user_id - 1] = rating
        print(f'User {user_id} rated movie {movie_id} with {rating}')


def check_digit_input(min, max, description=''):
    while True:
        n = input(description)
        if n.isdigit() and min <= int(n) <= max:
            return int(n)
        else:
            print(f'Please enter a number between {min} and {max}')


if __name__ == '__main__':
    print('Loading...')
    model = InferenceModel()
    print(
        'Done. You can now login as a user to rate movies and see recommendations for you'
    )
    print(
        'Please use the following commands by typing the index of the command')
    print('1. login')
    print('2. exit')
    command = check_digit_input(1, 2)
    if command == 1:
        user_id = check_digit_input(
            1, 943, 'Please enter your user id (from 1 to 943): ')
        while True:
            print('Please use the following commands by typing the index')
            print('1. rate movie')
            print('2. show recommendations')
            print('3. exit')
            command2 = check_digit_input(1, 3)
            if command2 == 1:
                movie_id = check_digit_input(
                    1, 1682, 'Please enter movie id (from 1 to 1682): ')
                rating = check_digit_input(
                    1, 5, 'Please enter rating (from 1 to 5): ')
                model.rate_movie(user_id, movie_id, rating)
            elif command2 == 2:
                num = check_digit_input(
                    1, 30,
                    'Please enter number of movies to be recommended (from 1 - 30): '
                )
                model.recommend(user_id, num)
                print('----------------------------------------------')
            else:
                exit()
    else:
        exit()
