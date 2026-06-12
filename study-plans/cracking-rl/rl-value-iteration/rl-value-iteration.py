def value_iteration(P, R, gamma, tol=1e-6, max_iters=1000):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    S = len(P)
    A = len(P[0])

    V = [0.0] * S
    policy = [0] * S

    for _ in range(max_iters):
        V_new = []
        for s in range(S):
            best_q, best_a = None, 0
            for a in range(A):
                q = 0.0
                for sp in range(S):
                    q += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])   # THE line, once
                if best_q is None or q > best_q:
                    best_q, best_a = q, a
            V_new.append(best_q)
            policy[s] = best_a            # remember the winner of this sweep

        moved = 0.0
        for s in range(S):
            diff = abs(V_new[s] - V[s])
            if diff > moved:
                moved = diff
        V = V_new
        if moved < tol:
            break

    rounded = []
    for v in V:
        rounded.append(round(v, 4))
    return rounded, policy
