def value_iteration_step(values, transitions, rewards, gamma):
    """
    Performs one step of value iteration for a Markov Decision Process.
    
    Args:
        values: List of current value estimates for each state.
        transitions: 3D list where T[s][a][s'] is the transition probability.
        rewards: 2D list where R[s][a] is the reward for action a in state s.
        gamma: Discount factor (float).
        
    Returns:
        list[float]: Updated value estimates for each state.
    """
    num_states = len(values)
    new_values = [0.0] * num_states
    
    # Iterate over every state s
    for s in range(num_states):
        q_values = []
        
        # Consider every possible action a from state s
        for a in range(len(transitions[s])):
            # Calculate expected future value: sum_{s'} T(s, a, s') * V(s')
            expected_future_value = 0
            for s_next in range(num_states):
                expected_future_value += transitions[s][a][s_next] * values[s_next]
            
            # Apply Bellman equation: Q(s, a) = R(s, a) + gamma * expected_future_value
            q_s_a = rewards[s][a] + gamma * expected_future_value
            q_values.append(q_s_a)
        
        # The new value of the state is the max Q-value available
        new_values[s] = float(max(q_values))
        
    return new_values