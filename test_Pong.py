"""
###############################################################################################################
##################################### Reinforcement Learning Assignment#3 #####################################
###############################################################################################################
###############################################################################################################
########################################## TEST MODULE FOR PONG ###########################################
###############################################################################################################
############################################## NAMES AND IDS ##################################################
1- Abdelrahman Mahmoud Alsayed -> 20190732
2- Yousef Yasser Ezzat -> 20190652
3- Omar Mohammed Abdelbary -> 20190358
4- Belal Kamal Ashraf -> 20190137
"""
#################################################### Imports ##################################################
from Pong_v4_Deep_Q_Learning import make_env,DQN
import torch
import time
import numpy as np
###############################################################################################################
################################################# INSTANTIATION ###############################################
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25
model='/Users/abdelrahmanibrahim/Desktop/Senior/Reinforcement_Learning/Assignment#3/pong_dqn_model.dat'
visualize=True
ENV = make_env(DEFAULT_ENV_NAME,render_mode=True)
NET = DQN(ENV.observation_space.shape, ENV.action_space.n)
NET.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
################################################# TEST FUNCTION ###############################################
def test(env,net):
    state = env.reset()
    total_reward = 0.0
    i = 0
    while i < 2000 :
            start_ts = time.time()
            state_v = torch.tensor(np.array([state], copy=False))
            q_vals = net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            env.render()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            if visualize:
                delta = 1/FPS - (time.time() - start_ts)
                if delta > 0:
                    time.sleep(delta)
            i +=1                
    print("Total reward: %.2f" % total_reward)
    env.close()
###############################################################################################################
################################################### RENDERING #################################################
test(ENV,NET)
#----------------------------------------------------DONE-----------------------------------------------------#

