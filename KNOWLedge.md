All question in 
    config_object ?

Exploration: the car tries a slightly different steering angle, braking point, or racing line.
Exploitation: the car repeats the line that already gave high reward.

Why have both trainer and worker ?

# SAC alogrithm
The Big Picture: 3 Processes Running in Parallel
┌─────────────┐     samples      ┌─────────────┐     weights     ┌─────────────┐
│   Worker    │ ──────────────►  │   Server    │ ◄────────────── │   Trainer   │
│ (plays TM)  │                  │ (relay hub) │ ──────────────► │ (runs SAC)  │
└─────────────┘                  └─────────────┘                 └─────────────┘
The server consolidates model weights and passes them to the trainer, while the trainer is responsible for actually training. The worker is responsible for interacting with the game. The trainer runs SAC in a loop on batches sampled from the replay memory — a large buffer of past (observation, action, reward, next_observation, done) transitions collected by workers.

NetworkRoleactor
The policy — takes observation, outputs a probability distribution over actions (steering, gas, brake)
q1, q2Twin critics — each takes (observation, action) and outputs a Q-value. Two used to prevent overestimation
q1_target, q2_target - Frozen slow-moving copies of critics, used to compute stable training targets

train() loop
    each training step does this with a batch (o, a, r, o2, d) — observation, action, reward, next observation, done
    Compute the critic target (what Q should be)
    Update the critics: loss_q = MSE(q1(o, a), backup) + MSE(q2(o, a), backup)
    Update the actor

## Q-value
Q-value comes from Q-learning. It answers the question:
    "If I'm in state S and take action A, how much total future reward will I get?"
More formally:
    Q(s, a) = immediate reward + discounted sum of all future rewards
        s: Current observation — speed, LIDAR distances, history of past frames
        a: Action chosen — gas, brake, steering angle
        Q(s, a): Expected total future reward (e.g., lap completion speed) if you take action a now and then follow the best policy afterward
Why SAC has two critics (twin Q-networks):
    SAC takes the minimum of two independent Q-value estimates to avoid overestimation bias. This is why self.output_layers in PopArt accepts a list — both critic heads get normalized together with the same mean/std.

