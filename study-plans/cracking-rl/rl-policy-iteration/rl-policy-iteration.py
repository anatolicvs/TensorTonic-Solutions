def policy_iteration(P, R, gamma, eval_tol=1e-8, max_iters=200):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    S = len(P)
    A = len(P[0])

    def q_value(s, a, V):
        q = 0.0
        for sp in range(S):
            q += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
        return q

    policy = [0] * S
    V = [0.0] * S

    for _ in range(max_iters):

        # evaluation: ONE action per state, chosen by the policy
        while True:
            V_new = []
            for s in range(S):
                a = policy[s]                  # ← the action, named explicitly
                V_new.append(q_value(s, a, V))
            moved = max(abs(V_new[s] - V[s]) for s in range(S))
            V = V_new
            if moved < eval_tol:
                break

        # improvement: ALL actions contested
        policy_new = []
        for s in range(S):
            best_q, best_a = None, 0
            for a in range(A):
                q = q_value(s, a, V)
                if best_q is None or q > best_q:
                    best_q, best_a = q, a
            policy_new.append(best_a)

        if policy_new == policy:
            break
        policy = policy_new

    return [round(v, 4) for v in V], policy


