#---------------------------- 
# The DQN agent, simplified and modified for 1 vs 1 case.  
#
# This code is only for research purpose.   
#
# This code is modified from keras-rl (https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py) and dqn-pysc2. 
# Most, if not all, modifications were explicitly pointed out in the code.    
#     
#  Xun Huang, Jul 29, 2021
#---------------------------- 


from __future__ import division
import warnings

# framework imports
from keras.layers import Lambda, Input, Dense, Conv2D, Flatten
from rl.memory import RingBuffer
#from rl.agents.dqn import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from baselines.common.schedules import LinearSchedule
from agent import Agent3 
import pdb




class Sc2Action:
    # default: noop
    def __init__(self, act=0, x=0, y=0):
        self.coords = (x, y)
        self.action = act
              


class AbstractSc2DQNAgent3(Agent3):
        def __init__(self, nb_actions, screen_size, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                     train_interval=1, memory_interval=1, target_model_update=10000, screen=32,
                     delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):
            super(AbstractSc2DQNAgent3, self).__init__(**kwargs)

            # Soft vs hard target model updates.
            if target_model_update < 0:
                raise ValueError('`target_model_update` must be >= 0.')
            elif target_model_update >= 1:
                # Hard update every `target_model_update` steps.
                target_model_update = int(target_model_update)
            else:
                # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
                target_model_update = float(target_model_update)

            if delta_range is not None:
                warnings.warn(
                    '`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(
                        delta_range[1]))
                delta_clip = delta_range[1]

            # Parameters.
            self._SCREEN = screen   #included by xun
            
            self.nb_actions = nb_actions
            self.screen_size = screen_size
            self.gamma = gamma
            self.batch_size = batch_size
            self.nb_steps_warmup = nb_steps_warmup
            self.train_interval = train_interval
            self.memory_interval = memory_interval
            self.target_model_update = target_model_update
            self.delta_clip = delta_clip
            self.custom_model_objects = custom_model_objects

            # Related objects.
            self.memory = memory

            # State.
            self.compiled = False

        # This code looks ridiculous to me. xun    
        def process_state_batch(self, batch):
            batch = np.array(batch)
            if self.processor is None:
                return batch
            return self.processor.process_state_batch(batch)

        def compute_batch_q_values(self, state_batch):
            batch = self.process_state_batch(state_batch)
#            print('debug step sc2agent 1')            
            #pdb.set_trace()
            q_values = self.model.predict_on_batch(batch)
            # assert q_values.shape == (len(state_batch), self.nb_actions) (len(state_batch), 2)
            return q_values

        def compute_q_values(self, state):
#            pdb.set_trace()        
#            q_values = self.compute_batch_q_values([state])
            #Modify by Xun to avoid unnecessary function calls        
            batch0=[state]
            batch0 = np.array(batch0)
            size_first_dim = len(batch0)
            size_second_dim = len(batch0[0,0])
            batch=np.reshape(batch0, (size_first_dim, size_second_dim, self._SCREEN, self._SCREEN))  
            #pdb.set_trace()        
            q_values = self.model.predict_on_batch(batch)
            return q_values



        def get_config(self):
            return {
                'nb_actions': self.nb_actions,
                'screen_size': self.screen_size,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'nb_steps_warmup': self.nb_steps_warmup,
                'train_interval': self.train_interval,
                'memory_interval': self.memory_interval,
                'target_model_update': self.target_model_update,
                'delta_clip': self.delta_clip,
                'memory': get_object_config(self.memory),
            }



class Sc2DqnAgent_v5(AbstractSc2DQNAgent3):
    def __init__(self, model, policy=None, test_policy=None,
                 prio_replay=True, prio_replay_beta=(0.5, 1.0, 200000),
                 bad_prio_replay=False, multi_step_size=3, *args, **kwargs):
        super(Sc2DqnAgent_v5, self).__init__(*args, **kwargs)

        # Validate (important) input. Falls man sein Model falsch definiert hat (  ^:
        if hasattr(model.output, '__len__') and len(model.output) != 2:
            raise ValueError(
                'Model "{}" has more or less than two outputs. DQN expects a model that has exactly 2 outputs.'.format(
                    model))

        # Parameters.
        self.prio_replay = True #prio_replay  Set to true by Xun but don't know why
        self.prio_replay_beta = prio_replay_beta
        self.bad_prio_replay = bad_prio_replay
        self.multi_step_size = multi_step_size

        # Related objects.
        self.model = model
        assert policy is not None
        if test_policy is None:
            test_policy = policy
        self.policy = policy
        self.test_policy = test_policy

#        if self.prio_replay:
        assert len(prio_replay_beta) == 3
        self.beta_schedule = LinearSchedule(prio_replay_beta[2],
                                            initial_p=prio_replay_beta[0],
                                            final_p=prio_replay_beta[1])

        self.recent = RingBuffer(maxlen=multi_step_size)
        # RingBuffer für Rewards
        self.recent_r = RingBuffer(maxlen=multi_step_size)

        # State.
        self.reset_states()

    def get_config(self):
        config = super(Sc2DqnAgent_v5, self).get_config()
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        
        self.target_model = clone_model(self.model, self.custom_model_objects)
        print("custom_model_objects: ", self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')


        # Lambda-Layer, welche den Loss des Netzwerks berechnet!
        def clipped_masked_error(args):
            y_true_a, y_true_b, y_pred_a, y_pred_b, mask_a, mask_b = args
            loss = [huber_loss(y_true_a, y_pred_a, self.delta_clip),
                    huber_loss(y_true_b, y_pred_b, self.delta_clip)]
            loss[0] *= mask_a  # apply element-wise mask
            loss[1] *= mask_b  # apply element-wise mask
            sum_loss_a = K.sum(loss[0])
            sum_loss_b = K.sum(loss[1])
            return K.sum([sum_loss_a, sum_loss_b], axis=-1)


        y_pred = self.model.output

        y_true_a = Input(name='y_true_a', shape=(self.nb_actions,))
        y_true_b = Input(name='y_true_b', shape=(self.screen_size, self.screen_size, 1))
        mask_a = Input(name='mask_a', shape=(self.nb_actions,))
        mask_b = Input(name='mask_b', shape=(self.screen_size, self.screen_size, 1))

        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')(
            [y_true_a, y_true_b, y_pred[0], y_pred[1], mask_a, mask_b])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input

 
        trainable_model = Model(inputs=ins + [y_true_a, y_true_b, mask_a, mask_b],
                                outputs=[loss_out, y_pred[0], y_pred[1]])
        print(trainable_model.summary())

        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses)  # metrics=combined_metrics
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def forward(self, observation):
        # Select an action.
        state = [observation]
#        print('debug step sc2agent 0')            
        #pdb.set_trace()   
        q_values = self.compute_q_values(state)
     
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent.append((observation, action))

        return action
        
        

    # Compared to keras-rl.core, here we have one new input, observation_1. xun    
    def backward(self, reward, terminal, observation_1):
        # RingBuffer.
        self.recent_r.append(reward)

        # Store most recent experience in memory. (s_t, a_t, r_t1 + gamma*r_t2, s_t2, ter2)
        # ??? I don't get the meaning of the following code, differnt from keras-rl code. Xun 
        if self.step % self.memory_interval == 0:
            # some resetting after terminal/done stuff to not save cross episodes.
            if self.recent.__len__() == self.recent.maxlen:
                if self.recent.__getitem__(0) is not None:
                    acc_r = 0
                    for i in range(self.recent_r.maxlen):
                        acc_r += self.recent_r.__getitem__(i) * (self.gamma ** i)

                    rec_0 = self.recent.__getitem__(0)
                    obs_0 = rec_0[0]
                    act_0 = rec_0[1]

                    self.memory.add(obs_0, act_0, acc_r, observation_1, terminal)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:

            experiences = self.memory.sample(self.batch_size, self.beta_schedule.value(self.step))
            assert len(experiences[0]) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            action_batch = []
            reward_batch = []
            state2_batch = []
            terminal2_batch = []
            if self.prio_replay:
                prio_weights_batch = []
                id_batch = []

            if self.prio_replay:
                experiences = zip(experiences[0], experiences[1], experiences[2], experiences[3], experiences[4],
                                  experiences[5], experiences[6])
            else:
                experiences = zip(experiences[0], experiences[1], experiences[2], experiences[3], experiences[4])

            for e in experiences:
                state0_batch.append(e[0])
                action_batch.append(e[1])
                reward_batch.append(e[2])
                state2_batch.append(e[3])
                terminal2_batch.append(0. if e[4] else 1.)
                if self.prio_replay:
                    prio_weights_batch.append(e[5])
                    id_batch.append(e[6])

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state2_batch = self.process_state_batch(state2_batch)
            terminal2_batch = np.array(terminal2_batch)
            reward_batch = np.array(reward_batch)
            if self.prio_replay:
                prio_weights_batch = np.array(prio_weights_batch)
            else:
                prio_weights_batch = np.ones(reward_batch.shape)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal2_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            target_q2_values = self.target_model.predict_on_batch(state2_batch)
            q_batch_a = np.max(target_q2_values[0], axis=-1)
            q_batch_b = np.max(target_q2_values[1], axis=(1, 2))[:, 0]
            q_batch_a = np.array(q_batch_a)
            q_batch_b = np.array(q_batch_b)

            targets_a = np.zeros((self.batch_size, self.nb_actions,))
            targets_b = np.zeros((self.batch_size, self.screen_size, self.screen_size, 1))

            masks_a = np.zeros((self.batch_size, self.nb_actions,))
            masks_b = np.zeros((self.batch_size, self.screen_size, self.screen_size, 1))

            # Compute r_t+n (included discounting) + gamma^n * max_a Q(s_t+n, a) and update the targets accordingly,
            # but only for the affected output units (as given by action_batch). (Called Rs_a and Rs_b)
            discounted_reward_batch_a = (self.gamma ** self.multi_step_size) * q_batch_a
            discounted_reward_batch_b = (self.gamma ** self.multi_step_size) * q_batch_b
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch_a = discounted_reward_batch_a * terminal2_batch[:]
            discounted_reward_batch_b = discounted_reward_batch_b * terminal2_batch[:]
            Rs_a = reward_batch[:] + discounted_reward_batch_a
            Rs_b = reward_batch[:] + discounted_reward_batch_b

            for idx, (target_a, target_b, mask_a, mask_b, R_a, R_b, action, prio_weight) in \
                    enumerate(zip(targets_a, targets_b, masks_a, masks_b, Rs_a, Rs_b, action_batch, prio_weights_batch)):
                target_a[action.action] = R_a  # update action with estimated accumulated reward
                target_b[action.coords] = R_b  # update action with estimated accumulated reward

                mask_a[action.action] = prio_weight  # enable loss for this specific action
                mask_b[action.coords] = prio_weight  # enable loss for this specific action

            targets_a = np.array(targets_a).astype('float32')
            targets_b = np.array(targets_b).astype('float32')
            masks_a = np.array(masks_a).astype('float32')
            masks_b = np.array(masks_b).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch

            metrics = self.trainable_model.train_on_batch(ins + [targets_a, targets_b, masks_a, masks_b],
                                                          [np.zeros(self.batch_size), targets_a, targets_b])

            metrics = [metric for idx, metric in enumerate(metrics) if
                       idx not in (1, 2)]  # throw away individual losses

            if self.prio_replay:
                pred = self.trainable_model.predict_on_batch(ins + [targets_a, targets_b, masks_a, masks_b])

            # update priority batch
            if self.prio_replay:
                prios = []
  
                # Richtige Implementierung.
                for (pre_a, pre_b, target_a, target_b, mask_a, mask_b, prio_weight) \
                        in zip(pred[1], pred[2], targets_a, targets_b, masks_a, masks_b, prio_weights_batch):
                        # need to remove prio weight from masks
                    mask_a = mask_a / prio_weight
                    mask_b = mask_b / prio_weight
                    loss = [pre_a - target_a,
                            pre_b - target_b]
                    loss[0] *= mask_a  # apply element-wise mask
                    loss[1] *= mask_b  # apply element-wise mask
                    sum_loss_a = np.sum(loss[0])
                    sum_loss_b = np.sum(loss[1])
                    prios.append(np.abs(np.sum([sum_loss_a, sum_loss_b])))

                self.memory.update_priorities(id_batch, prios)

            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 3
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
        
  





def mean_q(y_true, y_pred):
    mean_a = K.mean(K.max(y_pred[0], axis=-1))
    mean_b = K.mean(K.max(y_pred[1], axis=(1, 2)))
    return K.mean(mean_a, mean_b)
    



    


