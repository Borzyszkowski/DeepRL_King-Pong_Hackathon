import numpy as np 
import _pickle as pickle
import gym

input_size = 80*80
neurons = 320
gamma = 0.96
learning_rate = 1.5e-4
decay_rate = 0.96
render = False
resume = False

if resume:
  model = pickle.load(open('save.p', 'rb'))
  W1 = model['W1']
  W2 = model['W2']
else:
  model = {}
  W1 = np.random.normal(size=(neurons, input_size), scale=0.0001)
  W2 = np.random.normal(size=(neurons), scale=0.0001)

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def neuralNet(observations):
	h = W1.dot(observations)
	h[h < 0] = 0 # ReLU
	o = W2.dot(h)
	o = sigmoid(o)
	return o, h

def take_action(output):
	if(np.random.uniform() < output):
		return 2
	else:
		return 3

def compute_gradient(layers, losses):
	global W2, e_obs
	dW2 = np.dot(layers.T, losses).ravel()
	dh = np.outer(losses, W2)
	dh[layers <= 0] = 0
	dW1 = np.dot(dh.T, e_obs)
	return dW1, dW2

# me_pong - 
def discounted_reward(rewards):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    global gamma
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return discounted_rewards

# K - preprocesing frames
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

running_reward = None
running_ep = 0
layers, obs, losses, rewards = [],[],[],[]
reward_sum = 0
rmsprop_W1 = np.zeros_like(W1)
rmsprop_W2 = np.zeros_like(W2)
W1g = np.zeros_like(W1)
W2g = np.zeros_like(W2)
env = gym.make("Pong-v0")
observations = env.reset()
action = 3
test = True
episodes = 0
prev_x = None
while True:
	if(render):
		env.render()

	# K - frame difference (detecting move)
	cur_x = prepro(observations)
	x = cur_x - prev_x if prev_x is not None else np.zeros(input_size)
	prev_x = cur_x

	output, layer_1 = neuralNet(x)
	action = take_action(output)

	obs.append(x)
	layers.append(layer_1)
	fake_label = 1 if action == 2 else 0
	loss = fake_label - output
	losses.append(loss)

	observations, reward, done, _ = env.step(action)
	rewards.append(reward)
	reward_sum += reward

	if(done):

		episodes += 1
		running_ep += 1

		e_layers = np.vstack(layers)
		e_obs = np.vstack(obs)
		e_losses = np.vstack(losses)
		e_rewards = np.vstack(rewards)
		layers, obs, losses, rewards = [],[],[],[]

		e_losses *= discounted_reward(e_rewards)

		a, b = compute_gradient(e_layers, e_losses)
		W1g += a
		W2g += b

		if episodes % 11 == 0:
			# K - RMSprop
			rmsprop_W1 = decay_rate * rmsprop_W1 + (1 - decay_rate) * W1g**2
			W1 += learning_rate * W1g / (np.sqrt(rmsprop_W1) + 1e-5)
			rmsprop_W2 = decay_rate * rmsprop_W2 + (1 - decay_rate) * W2g**2
			W2 += learning_rate * W2g / (np.sqrt(rmsprop_W2) + 1e-5)

			W1g = np.zeros_like(W1)
			W2g = np.zeros_like(W2)
			print("weigths updated!")
			model['W1'] = W1 # "Xavier" initializatio
			model['W2'] = W2

		if episodes % 100 == 0: pickle.dump(model, open('save.p', 'wb'))

		# K - running reward
		running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

		print("game: " + str(running_ep) + " running reward: " + str(running_reward))
		reward_sum = 0
		observations = env.reset()
		prev_x = None
