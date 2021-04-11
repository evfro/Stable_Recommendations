This repository contains the source code to reproduce our work, accepted as a short paper at UMAP'2021.

## Requirements
- Python 3.8.1
- see additional dependencies in the file `requirements.txt`

## Running experiments

### Amazon Games

#### SVD:
```bash
python Script_All_Algo.py --algorithm='SVD' --data_path=AMZGamesDF.csv --df_='AMZG_' --n_tmonths=18 --n_train=10 --user_column='userId' --item_column='productId' --max_rank=80
```

#### PSI:

```bash
python Script_All_Algo.py --algorithm='PSI' --data_path=AMZGamesDF.csv --df_='AMZG_' --n_tmonths=18 --n_train=10 --user_column='userId' --item_column='productId' --max_rank=80
```


### Amazon Beauty:

#### SVD:

```bash
python Script_All_Algo.py --algorithm='SVD' --data_path=AMZBeautyDF.csv --df_='AMZB_' --n_tmonths=18 --n_train=10 --user_column='userId' --item_column='productId' --max_rank=80
```

#### PSI:

```bash
python Script_All_Algo.py --algorithm='PSI' --data_path=AMZBeautyDF.csv --df_='AMZB_' --n_tmonths=18 --n_train=10 --user_column='userId' --item_column='productId' --max_rank=80
```


### Movielens-1M:

#### SVD:

```bash
python Script_All_Algo.py --algorithm='SVD' --data_path=MovieLenDF_new.csv --df_='ML_' --n_tmonths=14 --n_train=6 --user_column='userId' --item_column='movieId' --max_rank=80
```

#### PSI:

```bash
python Script_All_Algo.py --algorithm='PSI' --data_path=MovieLenDF_new.csv --df_='ML_' --n_tmonths=14 --n_train=6 --user_column='userId' --item_column='movieId' --max_rank=80
```
