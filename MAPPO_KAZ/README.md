# MAPPO in Knights Archers Zombies ('KAZ') environment
This is a concise Pytorch implementation of MAPPO in KAZ environment.<br />

## How to use my code?  
You can directly run 'MAPPO_KAZ_main.py' in your own IDE.<br />

## Trainning environments
- Check out the [PIPELINE](https://colab.research.google.com/drive/1iwUXTvtUi1mj5z-9MvOpylY8yJfUNmHH)
- We train our MAPPO in KAZ environment.<br />

## Requirements
```
pip install pettingzoo[butterfly]
```

## Some details
we do not use RNN in 'actor' and 'critic' which can result in the better performence according to our experimental results.<br />
However, we also provide the implementation of using RNN. You can set 'use_rnn'=True in the hyperparameters setting, if you want to use RNN.<br />

## Trainning result

## Reference
[1] Yu C, Velu A, Vinitsky E, et al. The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games[J]. arXiv preprint arXiv:2103.01955, 2021.<br />
[2] [Official implementation of MAPPO](https://github.com/marlbenchmark/on-policy)<br />
[3] [EPyMARL](https://github.com/uoe-agents/epymarl)
