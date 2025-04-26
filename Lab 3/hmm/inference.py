import numpy as np
import graphics
import rover

CORNER_PROB = 0.333333
BORDER_PROB = 0.25
NON_CORNER_BORDER_PROB = 0.2

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 

    forward_messages[0] = rover.Distribution()
    for state in all_possible_hidden_states:
        if observations[0] is None:
            state_prob = prior_distribution[state] * 1
        else:
            state_prob = prior_distribution[state] * observation_model(state)[observations[0]] 

        if state_prob > 0:
            forward_messages[0][state] = state_prob
    forward_messages[0].renormalize()

    for t in range(1, num_time_steps):
        forward_messages[t] = rover.Distribution()
        for state in all_possible_hidden_states:
            if observations[t] is None:
                observation_prob = 1
            else:
                observation_prob = observation_model(state)[observations[t]] 
            state_prob = observation_prob * \
                sum(
                    forward_messages[t-1][prev_state] * transition_model(prev_state)[state]
                    for prev_state in forward_messages[t-1]
                )
            if state_prob > 0:
                forward_messages[t][state] = state_prob
        forward_messages[t].renormalize()


    # TODO: Compute the backward messages
    backward_messages[num_time_steps-1] = rover.Distribution()
    for state in all_possible_hidden_states:
        backward_messages[num_time_steps-1][state] = 1
    
    for t in range(num_time_steps-2, -1, -1):
        backward_messages[t] = rover.Distribution()
        for state in all_possible_hidden_states:
            state_prob = 0
            for next_state in backward_messages[t+1]:
                if observations[t+1] is None:
                    observation_prob = 1
                else:
                    observation_prob = observation_model(next_state)[observations[t+1]]
                state_prob += transition_model(state)[next_state] * observation_prob * backward_messages[t+1][next_state]

            if state_prob > 0:
                backward_messages[t][state] = state_prob
        backward_messages[t].renormalize()

    # TODO: Compute the marginals 
    for t in range(num_time_steps):
        marginals[t] = rover.Distribution()
        for state in all_possible_hidden_states:
            state_prob = forward_messages[t][state] * backward_messages[t][state]
            if state_prob > 0:
                marginals[t][state] = state_prob
        marginals[t].renormalize()

    # Sanity Check: 
    print("Marginals at time 30:")
    print(marginals[30])
    
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    estimated_hidden_states = [None] * len(observations)

    w = [None] * len(observations)
    phi = [{} for _ in range(len(observations))] 
    w[0] = rover.Distribution()

    for state in all_possible_hidden_states:
        if observations[0] is None:
            w[0][state] = np.log(prior_distribution[state]) + np.log(1)
        else:
            w[0][state] = np.log(prior_distribution[state]) + np.log(observation_model(state)[observations[0]])
    w[0].renormalize()


    for t in range(1, len(observations)):
        w[t] = rover.Distribution()
        phi[t] = {}
        for curr_state in all_possible_hidden_states:
            if observations[t] is None:
                obs_prob = 1
            else:
                obs_prob = observation_model(curr_state)[observations[t]]
            
            max_prob = float('-inf')
            best_prev = None
            for prev_state in w[t-1]:
                prob = (w[t-1][prev_state] + 
                       np.log(transition_model(prev_state)[curr_state]) + 
                       np.log(obs_prob))
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_state
                
            if max_prob > float('-inf'):
                w[t][curr_state] = max_prob
                phi[t][curr_state] = best_prev
        w[t].renormalize()
    
    # Find most likely final state
    last_t = len(observations) - 1
    if w[last_t]:  
        curr_state = max(w[last_t].items(), key=lambda x: x[1])[0]
        estimated_hidden_states[last_t] = curr_state

        for t in range(last_t, 0, -1):
            curr_state = phi[t][curr_state]  # Get previous state from backpointer
            estimated_hidden_states[t-1] = curr_state

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
