"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training
ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    # TODO Your code here

    #print(f'state_1 = {state_1}')
    #print(f'state_2 = {state_2}')
    #print(f'epsilon = {epsilon}')
    

    action_index, object_index = None, None
    random_number = np.random.uniform(0,1)
    #print(f'Random number selected is {random_number}')
    # Toma valor aleatorio
    if random_number <= epsilon: 
        #print(f'Entró en random')
        action_index = np.random.randint(0, NUM_ACTIONS) # accion aleatoria
        object_index = np.random.randint(0, NUM_OBJECTS) # objeto aleatorio
    # Toma la mejor decision
    else:
        #print(f'Entró en decision correcta')
        submatrix = q_func[state_1, state_2, :, :]

        #print(f'submatrix = {submatrix}')
        # Índice plano del máximo
        flat_index = np.argmax(submatrix)
        # Convertir a índices (c, d)
        action_index, object_index = np.unravel_index(flat_index, submatrix.shape)

    return (action_index, object_index)


# pragma: coderesponse end


# pragma: coderesponse template
def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    # TODO Your code here
    
    #q_func[current_state_1, current_state_2, action_index,
    #       object_index] = 0  # TODO Your update here

    # current_state_1 = room
    # current_stante_2 = mision o quest
    # action = accion
    # object = objecto 


    q_actual = q_func[current_state_1, current_state_2, action_index, object_index]
    if terminal == True:
        target = reward
    else:
        max_q = np.max(q_func[next_state_1, next_state_2, :, :])

        target = reward + GAMMA * max_q

    q_new = (1-ALPHA) * q_actual + ALPHA * target

    q_func[current_state_1, current_state_2, action_index, object_index] = q_new

    return None  # This function shouldn't return anything


# pragma: coderesponse end


# pragma: coderesponse template
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = None
    # initialize for each episode
    # TODO Your code here

    # Inicializar Q_func como puros ceros.
    # Ya está hecho en la funcion run()

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    epi_reward = 0
    step_index = 0  # para el descuento gamma^t


    while not terminal:
        # Choose next action and execute
        # TODO Your code here
        current_room = dict_room_desc[current_room_desc]
        current_quest = dict_quest_desc[current_quest_desc]

        next_action, next_object = epsilon_greedy(current_room, current_quest, q_func, epsilon)
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(current_room_desc, current_quest_desc, next_action, next_object)

        next_room = dict_room_desc[next_room_desc]
        next_quest = dict_quest_desc[next_quest_desc]

        if for_training:
            # update Q-function.

            tabular_q_learning(q_func, current_room, current_quest, next_action,
                       next_object, reward, next_room, next_quest,
                       terminal)

        if not for_training:
            # update reward
            epi_reward += (GAMMA ** step_index) * reward
            step_index += 1  # para el descuento gamma^t
        
        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc

    if not for_training:
        return epi_reward


# pragma: coderesponse end


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
