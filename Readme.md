## Introduction
This is a simple movie recommendation system based on the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/). The core movie recommendation algorithm is based on [Glocal-K](https://arxiv.org/pdf/2108.12184.pdf)
## Usage
### Train
just follow the code in `train.ipynb`
### Play
you can run `python CLI.py`, which is a simple command line interface for the recommendation system. It allows you to login, rate movies, and get recommendations.
## Performance
As you can see `baseline.ipynb`, we've compared Glocal-K with random walk and collaborative filtering. Glocal-K outperforms the other two algorithms in terms of both accuracy and speed.
|      | GLocal-K | CF(user-based) | CF(item-based) | RW    |
|------|----------|----------------|----------------|-------|
| RMSE | **0.922**    | 2.65           | 2.78           | 1.32  |
| MAE  | **0.726**    | 2.41           | 2.53           | 1.05  |
| NDCG | **0.895**    | 0.822          | 0.817          | 0.842 |
## Reference
1. The core algorithm is based on [this implementation](https://github.com/fleanend/TorchGlocalK)
2. Harper, F. Maxwell, and Joseph A. Konstan. "The movielens datasets: History and context." Acm transactions on interactive intelligent systems (tiis) 5.4 (2015): 1-19.
3. Han, Soyeon Caren, et al. "GLocal-K: Global and Local Kernels for Recommender Systems." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.