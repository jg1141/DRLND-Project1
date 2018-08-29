# DRLND-Project1 Report
DRLND Project 1 - Navigation (Banana environment)

## Learning Algorithm

The learning algorithm implemented for this project is a **Double DQN** patterned on the Deep Q Network example in the [lesson from Udacity](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/dqn_agent.py). Double DQN, as described by van Hasselt et al in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), "can decouple the selection from the evaluation" by using [Double Q-learning](http://papers.nips.cc/paper/3964-double-q-learning.pdf) in a deep learning network context.

Specifically, the **diff** below shows the addition of a flag for Double DQN, **.eval()** (training of) the second TARGET network and selection of the action values based on the flag:

```
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

 The model is a mapping of state to action values via fully connected **Linear** layers with **relu** activation and **Dropout**. The final output layer yields weighted outputs for each of the four possible actions, with action selection made via **argmax** wrapped in **Epsilon-greedy action selection** logic to allow random exploration (governed by the value of epsilon, which declines through the course of the training).

```markdown
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 1, 64]           2,432
           Dropout-2                [-1, 1, 64]               0
            Linear-3                [-1, 1, 64]           4,160
           Dropout-4                [-1, 1, 64]               0
            Linear-5                 [-1, 1, 4]             260
================================================================
Total params: 6,852
```

 

## Plot of Rewards

![Plot of Rewards](https://github.com/jg1141/DRLND-Project1/blob/master/Plot%20of%20Rewards.png)



## Ideas for Future Work

These are exciting times for Reinforcement Learning (RL). The paper [Rainbow: Combining Improvements in Deep Reinforcement Learning ](https://arxiv.org/pdf/1710.02298.pdf) suggested combining many known optimizations of RL into a single model. While this project was underway, Google released [Dopamine](https://github.com/google/dopamine) "a research framework for fast prototyping of reinforcement learning algorithms" which includes a **Rainbow** [agent](https://github.com/google/dopamine/tree/master/dopamine/agents/rainbow):

```
Specifically, we implement the following components from Rainbow:

  * n-step updates;
  * prioritized replay; and
  * distributional RL.
```

The next step for improving agent performance in this project would be to re-implement it using Dopamine.

Then, there is lots more to study at [http://bit.ly/drlndlinks](http://bit.ly/drlndlinks)!