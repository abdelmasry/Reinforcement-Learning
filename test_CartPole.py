"""
###############################################################################################################
##################################### Reinforcement Learning Assignment#3 #####################################
###############################################################################################################
###############################################################################################################
######################################## TEST MODULE FOR Deep-CartPole ########################################
###############################################################################################################
############################################## NAMES AND IDS ##################################################
1- Abdelrahman Mahmoud Alsayed -> 20190732
2- Yousef Yasser Ezzat -> 20190652
3- Omar Mohammed Abdelbary -> 20190358
4- Belal Kamal Ashraf -> 20190137
"""
#################################################### Imports ##################################################
import gym
import torch
from CartPole_v1_Deep_Q_Learning import AGENT


################################################# TEST FUNCTION ###############################################
def test(agent, env):
    state = env.reset()
    for j in range(1000):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()


###############################################################################################################
################################################### RENDERING #################################################
AGENT.qnetwork_local.load_state_dict(
    torch.load(
        "/Users/abdelrahmanibrahim/Desktop/Senior/Reinforcement_Learning/Assignment#3/cartpole_dqn_model.pth"
    )
)
ENV = gym.make("CartPole-v1", render_mode="human")
test(AGENT, ENV)
#----------------------------------------------------DONE-----------------------------------------------------#
