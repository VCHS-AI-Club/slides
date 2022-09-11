---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
math: katex
---

# <!-- fit --> AI <br> Snake Game

---

# How Does It Work?

Reinforcement learning (RL) is an area of machine
learning concerned with how software agents ought to
take actions in an environment in order to maximize the notion of
cumulative reward.

Or:
RL is teaching a software agent how to behave in an environment by
telling it how good it's doing.

**agent**
**environment**
**reward**

# RL Basics

S: state space; all possible states
P_0: starting state distribution;
A: action space: all possible actions
P_1: transition function; P_1(S, A) = P(S' | S, A); What is the probability of transitioning from state S to state S' given action A (in snake its 100%)
R(s_t, a_t, a_t+1): reward function; R(s_t, a_t, a_t+1) = Reward for taking action a in state s at time t and transitioning to state S' at time t+1

# Deep Q Learning

Feed forward linear q net

---

# Training Loop

---

# Bellman Equation

$$
Q_{\text{new}}(s,a)
= Q_{\text{old}}(s,a)
+\alpha[
    R(a \rightarrow s)
    + \gamma \max_a Q'(s',a \in \Gamma(s))
    - Q(s,a)
]
$$

$$
Q(s) = \sum^{t-1}_{t=0}\gamma^t R(s)
$$
