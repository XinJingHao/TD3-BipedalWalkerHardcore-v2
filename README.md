# Utilizing TD3 to solve BipedalWalkerHardcore-v2. 


Some discussion can be found [here](https://zhuanlan.zhihu.com/p/409553262)  
Vedio of all training process can be found [here](https://www.bilibili.com/video/BV1oa411F7e7?t=4)  
**Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**

Pytorch version. 

Author: Jinghao Xin, SJTU,China

Simulation Result:  
![avatar](https://github.com/XinJingHao/TD3/blob/main/final%20result.gif)

-----------------------------------------

## Dependencies
```bash
python=3.7.9 
pytorch=1.7.0 
numpy=1.18.5 
gym=0.17.3 
matplotlib=3.3.2 
box2d-py=2.3.8
```
-----------------------------------------
## How to use my code

Load trained model and render : just run 'python main.py' 

Train your own model : Change 'render' and 'Loadmodel' in 'main.py' to False,and run 'python main.py'
