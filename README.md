# MARL-code-pytorch
Concise pytorch implements of MARL algorithms, including MAPPO, MADDPG, MATD3, QMIX and VDN.
- Check out the [PIPELINE](https://colab.research.google.com/drive/1Ffmd-AXx6NehiddVK5NIgQOGM5MJE4Qt) to run mappo on MPE.
- Check out the [PIPELINE](https://colab.research.google.com/drive/1f8Aa8KTXwHayQgXitOxrx0ODThNpMpxS) to run mappo on SMAC.
- Check out the [PIPELINE](https://colab.research.google.com/drive/1kBkdh8z7fFi_YG96nyfVYGwwGJMd_65Y) to run mappo on MULTIWALKER (PettingZoo[sisl]).
- Check out the [PIPELINE](https://colab.research.google.com/drive/1iwUXTvtUi1mj5z-9MvOpylY8yJfUNmHH) to run mappo on 'Knights Archers Zombies' (PettingZoo[butterfly]).
# Requirements
[Multi-Agent Particle-World Environment(MPE)](https://github.com/openai/multiagent-particle-envs)<br />
[SMAC-StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac)<br />

# Trainning results
## 1. MAPPO in MPE (discrete action space)
![image](assets/spread.png)

## 2. MAPPO in MULTIWALKER (continuous action space)
<p align="center"> <img src="assets/mappo-multiwalker/multiwalker_steps_2000041.gif" width="300" alt="multiwalker" /> <img src="assets/mappo-shapley-multiwalker/multiwalker_steps_2000234.gif" width="300" alt="multiwalker" /> </p> <p align="center"> <em>Left: MAPPO, Right: MAPPO-SBAL</em> </p>


If you want to see the learning process during training, please check out [YOUTUBE](https://www.youtube.com/watch?v=erYeBC8ItZQ).

![image](assets/multiwalker.png)

## 3. MAPPO in Knights Archers Zombies (discrete action space)
In this experiment, I used an environment where agents have different tasks to test performance while Shapley rewards are distributed during training. I computed the average every 20k training steps.

Note that: KAZ consists of 4 agents in the following order: archer 0, archer 1, knight 0, knight 1, and they are redistributed using Shapley in the order: agent 1, agent 2, agent 3, agent 4.
![image](assets/KAZ_seed_0_shapley.png)

Comparison with the original MAPPO.
![image](assets/kaz.png)

## 4. MAPPO in  StarCraft II(SMAC)
![image](assets/smac_3m.png)

## 5. QMIX and VDN in StarCraft II(SMAC)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/3.QMIX_VDN_SMAC/QMIX_SMAC_training_result.png)

## 6. MADDPG and MATD3 in MPE (continuous action space)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/4.MADDPG_MATD3_MPE/MADDPG_MATD3_training_result.png)

