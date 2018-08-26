# DRLND-Project1 Report
DRLND Project 1 - Navigation (Banana environment)

## Learning Algorithm

The learning algorithm implemented for this project is a **Double DQN** patterned on the Deep Q Network example in the [lesson from Udacity](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/dqn_agent.py). Double DQN, as described by van Hasselt et al in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), "can decouple the selection from the evaluation" by using [Double Q-learning](http://papers.nips.cc/paper/3964-double-q-learning.pdf) in a deep learning network context.

Specifically, the **diff** below shows the addition of a flag for Double DQN, **.eval()** (training of) the second TARGET network and selection of the action values based on the flag:

```python
> diff dqn_agent.py deep-reinforcement-learning/dqn/solution/dqn_agent.py
...
58c58
<     def act(self, state, eps=0., double_dqn=True):
---
>     def act(self, state, eps=0.):
65d64
<             double_dqn (boolean): flag for Double DQN
69,70d67
<         # Add second TARGET network
<         self.qnetwork_target.eval()
72,76c69
<             # Assign action values based on double_dqn flag
<             if double_dqn:
<                 action_values = self.qnetwork_target(state)
<             else:
<                 action_values = self.qnetwork_local(state)
---
>             action_values = self.qnetwork_local(state)
```

#### Hyperparameters

```python
BATCH_SIZE = 64         # minibatch size
BUFFER_SIZE = 10000     # replay buffer size
GAMMA = 0.99            # discount factor
LEARNING_RATE = 0.0005  # learning rate 
TAU = 0.001             # for soft update of target parameters
UPDATE_EVERY = 4        # how often to update the network
```

These are unchanged from the Udacity example. The BATCH_SIZE and BUFFER_SIZE are parameters for the **ReplayBuffer** class, an "memory" randomly sampled at each step to obtain _experiences_ passed into the **learn** method with a discount of GAMMA. LEARNING_RATE is a parameter to the **Adam** optimizer. TAU is a parameter for a _soft update_ of the target and local models. Finally, UPDATE_EVERY determines the number of steps before learning from a new sample.

#### Model Architecture

 The model is a mapping of state to action values via fully connected **Linear** layers with **relu** activation. 

## Plot of Rewards

![Plot of Rewards](https://github.com/jg1141/DRLND-Project1/blob/master/Plot%20of%20Rewards.png)



## Ideas for Future Work

The paper [Rainbow: Combining Improvements in Deep Reinforcement Learning ](https://arxiv.org/pdf/1710.02298.pdf) suggests combining all the optimizations of RL into a single model.

Lots more to study at [http://bit.ly/drlndlinks](http://bit.ly/drlndlinks)!