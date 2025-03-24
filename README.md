# MARL-code-pytorch
Concise pytorch implements of MARL algorithms, including MAPPO, MADDPG, MATD3, QMIX and VDN.
- Check out the [PIPELINE](https://colab.research.google.com/drive/1Ffmd-AXx6NehiddVK5NIgQOGM5MJE4Qt) to run mappo on MPE.
- Check out the [PIPELINE](https://colab.research.google.com/drive/1f8Aa8KTXwHayQgXitOxrx0ODThNpMpxS) to run mappo on SMAC.
- Check out the [PIPELINE](https://colab.research.google.com/drive/1kBkdh8z7fFi_YG96nyfVYGwwGJMd_65Y) to run mappo on MULTIWALKER (PettingZoo[sisl]).
# Requirements
[Multi-Agent Particle-World Environment(MPE)](https://github.com/openai/multiagent-particle-envs)<br />
[SMAC-StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac)<br />

# Trainning results
## 1. MAPPO in MPE (discrete action space)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/1.MAPPO_MPE/MAPPO_MPE_training_result.png)

## 2. MAPPO in MULTIWALKER (continuous action space)
<p align="center"> <img src="assets/mappo-multiwalker/multiwalker_steps_2000041.gif" width="300" alt="multiwalker" /> <img src="assets/mappo-shapley-multiwalker/multiwalker_steps_2000234.gif" width="300" alt="multiwalker" /> </p> <p align="center"> <em>Left: MAPPO, Right: MAPPO-SBAL</em> </p>

![image](assets/multiwalker.png)

## 3. MAPPO in  StarCraft II(SMAC)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/2.MAPPO_SMAC/MAPPO_SMAC_training_result.png)

## 4. QMIX and VDN in StarCraft II(SMAC)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/3.QMIX_VDN_SMAC/QMIX_SMAC_training_result.png)

## 5. MADDPG and MATD3 in MPE (continuous action space)
![image](https://github.com/Lizhi-sjtu/MARL-code-pytorch/blob/main/4.MADDPG_MATD3_MPE/MADDPG_MATD3_training_result.png)

