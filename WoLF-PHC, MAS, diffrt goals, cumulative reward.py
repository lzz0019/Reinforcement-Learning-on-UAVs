import numpy as np
import math
import matplotlib.pyplot as plt


def find_state_from_position(position):
    x = position[0]
    y = position[1]
    if x != 100 and y != 100:
        state_num = (y//1) * 100 + ((x//1) + 1)
    elif x == 100 and y != 100:
        state_num = (y//1) * 100 + x
    elif y == 100 and x != 100:
        state_num = (y-1)*100 + ((x//1) + 1)
    else:
        state_num = 10000
    return int(state_num)


def calculate_reward(previous_state, current_state, goal_state):
    if current_state == goal_state:
        r = 20
    elif abs(current_state % 100 - goal_state % 100) < abs(previous_state % 100 - goal_state % 100) \
            or abs(current_state/100 - goal_state/100) < abs(previous_state/100 - goal_state/100):
        r = 5
    else:
        r = -5
    return r


def find_max_q_at_state_s(s, matrix):
    q_list_at_state_s = matrix[s-1]
    max_q_value = np.max(q_list_at_state_s)
    return max_q_value


def calculate_delta(matrix_pi, matrix_pi_mean, matrix_q, state, win_learning_rate, lose_learning_rate):
    left_side = 0
    right_side = 0
    for action in range(4):
        left_side += matrix_pi[state-1, action] * matrix_q[state-1, action]
        right_side += matrix_pi_mean[state-1, action] * matrix_q[state-1, action]
    if left_side > right_side:
        learning_rate = win_learning_rate
    else:
        learning_rate = lose_learning_rate
    return learning_rate


def __calculate_delta_sa(matrix_pi, state, action, learning_rate):
    compare_list = np.array([ matrix_pi[state-1, action], learning_rate / (4-1)])
    return np.min(compare_list)


def calculate_triangle_delta_sa(state, action, matrix_pi, matrix_q, learning_rate):
    q_values_list = matrix_q[state-1]
    arg_max_a = np.argmax(q_values_list)
    if action != arg_max_a:
        result = 0 - __calculate_delta_sa(matrix_pi, state, action, learning_rate)
    else:
        result = 0
        for i in range(4):
            if i != arg_max_a:
                result += __calculate_delta_sa(matrix_pi, state, i, learning_rate)
    return result


def wolf_phc_learning(position, goal_state):
    cumulRwrdList = []
    currentState = find_state_from_position(position)
    Q_matrix = np.zeros((10000, 4))
    pi_matrix = np.zeros((10000, 4))
    pi_mean_matrix = np.zeros((10000, 4))
    C_matrix = np.zeros((10000, 1), dtype=np.int)
    # Initialize pi-matrix
    pi_matrix.fill(0)
    for j in range(0, 10000):
        if j == 0:
            pi_matrix[j, 0] = 0.5
            pi_matrix[j, 3] = 0.5
        elif j == 99:
            pi_matrix[j, 0] = 0.5
            pi_matrix[j, 2] = 0.5
        elif j == 9900:
            pi_matrix[j, 1] = 0.5
            pi_matrix[j, 3] = 0.5
        elif j == 9999:
            pi_matrix[j, 1] = 0.5
            pi_matrix[j, 2] = 0.5
        elif 1 <= j <= 98:
            pi_matrix[j, 0] = 1.0 / 3.0
            pi_matrix[j, 2] = 1.0 / 3.0
            pi_matrix[j, 3] = 1.0 / 3.0
        elif 9901 <= j <= 9998:
            pi_matrix[j, 1] = 1.0 / 3.0
            pi_matrix[j, 2] = 1.0 / 3.0
            pi_matrix[j, 3] = 1.0 / 3.0
        elif j % 100 == 0:
            pi_matrix[j, 0] = 1.0 / 3.0
            pi_matrix[j, 1] = 1.0 / 3.0
            pi_matrix[j, 3] = 1.0 / 3.0
        elif (j+1) % 100 == 0:
            pi_matrix[j, 0] = 1.0 / 3.0
            pi_matrix[j, 1] = 1.0 / 3.0
            pi_matrix[j, 2] = 1.0 / 3.0
        else:
            pi_matrix[j] = 0.25

    step = 0
    cumulativeReward = 0
    cumulRwrdList.append(cumulativeReward)
    while step != 1000:
        step += 1
        # print 'currentState= ', currentState
        #   a) From state s select action a according to mixed strategy Pi with suitable exploration
        actionProbability = pi_matrix[currentState - 1]
        # print 'actionProbability = ', actionProbability
        currentAction = np.random.choice(4, 1, p=actionProbability)
        # print 'currentAction = ', currentAction
        #   b) Observing reward and next state s'
        nextState = currentState + calculate_nextState_table[currentAction[0]]
        # print 'nextSate = ', nextState
        reward = calculate_reward(currentState, nextState, goal_state)
        cumulativeReward += reward
        cumulRwrdList.append(cumulativeReward)
        max_Q = find_max_q_at_state_s(nextState, Q_matrix)
        # print 'max_Q = ', max_Q
        Q_matrix[currentState-1, currentAction[0]] = \
            (1 - alpha) * Q_matrix[currentState-1, currentAction[0]] + alpha * (
                reward + gama * max_Q)
        #   c) update estimate of average policy, pi_mean_matrix
        pi_mean_matrix = (pi_mean_matrix * (step - 1) + pi_matrix) / step
        C_matrix[currentState-1] += 1
        for each_action in range(4):
            pi_mean_matrix[currentState-1, each_action] += (1 / C_matrix[currentState-1]) * (
                pi_matrix[currentState-1, each_action] - pi_mean_matrix[currentState-1, each_action]
            )
        #   d) Step pi closer to the optimal policy w.r.t Q
        delta = calculate_delta(pi_matrix, pi_mean_matrix, Q_matrix, currentState, delta_W, delta_L)
        triangle_delta_sa = calculate_triangle_delta_sa(currentState, currentAction, pi_matrix, Q_matrix, delta)
        pi_matrix[currentState-1, currentAction] += triangle_delta_sa
        sum_probability = np.sum(pi_matrix[currentState-1])
        pi_matrix[currentState-1] /= sum_probability
        currentState = nextState
        # caculate normalized Q value
        Q_value = math.sqrt(np.sum((Q_matrix ** 2)))
    print 'reached goal!'
    print 'total steps = ', step
    return Q_matrix, pi_matrix, pi_mean_matrix, cumulRwrdList


# --------------------- Each agent has different goal ------------------------ #

Agent1_position = [55.0, 46.0]
Agent2_position = [28.0, 70.0]
Agent3_position = [24.0, 50.0]
Agent4_position = [77.0, 45.0]
Agent5_position = [20.8, 28.8]

calculate_nextState_table = (100, -100, -1, 1)
alpha = 0.5
gama = 0.4
delta_L = 0.8
delta_W = 0.4

Agent1_goal = 3228
Agent2_goal = 6799
Agent3_goal = 4923
Agent4_goal = 5054
Agent5_goal = 8179

Agent1_trainingResult = wolf_phc_learning(Agent1_position, Agent1_goal)
Agent2_trainingResult = wolf_phc_learning(Agent2_position, Agent2_goal)
Agent3_trainingResult = wolf_phc_learning(Agent3_position, Agent3_goal)
Agent4_trainingResult = wolf_phc_learning(Agent4_position, Agent4_goal)
Agent5_trainingResult = wolf_phc_learning(Agent5_position, Agent5_goal)

Agent1_cumulRwrd = np.asarray(Agent1_trainingResult[3])
Agent2_cumulRwrd = np.asarray(Agent2_trainingResult[3])
Agent3_cumulRwrd = np.asarray(Agent3_trainingResult[3])
Agent4_cumulRwrd = np.asarray(Agent4_trainingResult[3])
Agent5_cumulRwrd = np.asarray(Agent5_trainingResult[3])

x = np.arange(1001)
plt.plot(x, Agent1_cumulRwrd, label="Agent1")
plt.plot(x, Agent2_cumulRwrd, label="Agent2")
plt.plot(x, Agent3_cumulRwrd, label="Agent3")
plt.plot(x, Agent4_cumulRwrd, label="Agent4")
plt.plot(x, Agent5_cumulRwrd, label="Agent5")
plt.legend()
plt.title("WoLF-PHC MAS, cumulative reward")
plt.show()

