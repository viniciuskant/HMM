import numpy as np


# ---------- utilidades numéricas ----------

def logsumexp(a, axis=None):
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(a_max), a_max, 0.0)
    out = safe + np.log(np.sum(np.exp(a - safe), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def estimate_self_prob(avg_frames, n_states, min_self=0.5, max_self=0.95):
    # baseado no numero de estados determina o tempo esperado para cada estado baseado no número médio de frames
    dur_per_state = max(avg_frames / n_states, 1.0)

    # Relação entre self-loop e duração esperada
    # Isso significa que o estado fica X frames antes de avançar.
    self_prob = 1.0 - 1.0 / dur_per_state

    # limites para manter a topologia funcional:
    #       < 0.5: avançaria rápido demais (menos de 2 frames por estado)
    #       > 0.95: ficaria preso no estado (duração > 20 frames)
    return float(np.clip(self_prob, min_self, max_self))


class DiagonalGMMHMM:
    def __init__(self, n_components, n_mix, n_iter=30, tol=1e-4,
                 seed=0, self_prob=None):
        self.n_components = n_components
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.self_prob = self_prob
        self.rng = np.random.default_rng(seed)

        self.means_ = None       # (n, m, D)
        self.covars_ = None      # (n, m, D)  — diagonal
        self.weights_ = None     # (n, m)
        self.transmat_ = None    # (n, n)
        self.startprob_ = None   # (n,)

    def _init_topology(self):
        n = self.n_components # número de estados
        sp = self.self_prob if self.self_prob is not None else 0.7
        A = np.zeros((n, n)) # matriz com a propabilidade entre ir de um estado i para o j (A[i][j])
        for i in range(n):
            if i < n - 1: #todo estados menos o último, so pode ir para si mesmo ou para o próximo
                A[i, i] = sp
                A[i, i + 1] = 1.0 - sp
            else:
                A[i, i] = 1.0 #o último so para si próprio
        self.transmat_ = A
        pi = np.zeros(n)
        pi[0] = 1.0
        self.startprob_ = pi

    def _init_gmm(self, X):
        T, D = X.shape #mfcc concatenada, tamanho de cada mfcc
        n, m = self.n_components, self.n_mix #número de estados, número de "representações"

        self.means_ = np.zeros((n, m, D)) # o "centro" daquela gaussiana
        self.covars_ = np.zeros((n, m, D)) # Variâncias das gaussianas

        self.weights_ = np.full((n, m), 1.0 / m) # pesos das misturas, no início, todas as misturas são igualmente importantes

        idx = np.linspace(0, T, n + 1).astype(int)
        for i in range(n):
            a, b = idx[i], max(idx[i + 1], idx[i] + 1)
            chunk = X[a:b]
            if len(chunk) == 0:
                chunk = X
            center = chunk.mean(axis=0)
            var = chunk.var(axis=0) + 1e-3
            for k in range(m):
                self.means_[i, k] = center + self.rng.normal(0, 1e-2, D)
                self.covars_[i, k] = var


    def _log_gaussian(self, X, mu, var):
        var = np.maximum(var, 1e-6)
        D = X.shape[1]
        diff = X - mu
        # print("diff:", diff)
        # print("X:", X)
        # print("mu:", mu)
        maha = np.sum((diff * diff) / var, axis=1)
        # print("maha:", maha)
        log_det = np.sum(np.log(var))
        return -0.5 * (D * np.log(2 * np.pi) + log_det + maha)

    def _log_emission(self, X):
        T = X.shape[0]
        n, m = self.n_components, self.n_mix
        log_B = np.empty((T, n))
        log_w = np.log(self.weights_ + 1e-300)   # (n, m)
        for i in range(n):
            lg = np.empty((T, m))
            for k in range(m):
                lg[:, k] = self._log_gaussian(X, self.means_[i, k], self.covars_[i, k])
                # print("lg", lg)
            log_B[:, i] = logsumexp(log_w[i][None, :] + lg, axis=1)
        return log_B

    def _forward(self, log_B):
        T, n = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)
        log_pi = np.log(self.startprob_ + 1e-300)

        log_alpha = np.full((T, n), -np.inf)
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            a = log_alpha[t - 1][:, None] + log_A   # (n, n)
            log_alpha[t] = logsumexp(a, axis=0) + log_B[t]
        log_P = logsumexp(log_alpha[-1], axis=0)
        return log_alpha, log_P

    def _backward(self, log_B):
        T, n = log_B.shape
        log_A = np.log(self.transmat_ + 1e-300)
        log_beta = np.full((T, n), -np.inf)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            b = log_A + (log_B[t + 1] + log_beta[t + 1])[None, :]
            log_beta[t] = logsumexp(b, axis=1)
        return log_beta

    # ---------- E-step ----------

    def _responsibilities(self, X):
        T = X.shape[0]
        n, m = self.n_components, self.n_mix

        log_B = self._log_emission(X)
        log_alpha, log_P = self._forward(log_B)
        log_beta = self._backward(log_B)

        # gamma
        gamma = np.exp(log_alpha + log_beta - log_P)   # (T, n)

        # xi
        log_A = np.log(self.transmat_ + 1e-300)
        xi = np.empty((T - 1, n, n))
        for t in range(T - 1):
            a = (log_alpha[t][:, None]
                 + log_A
                 + (log_B[t + 1] + log_beta[t + 1])[None, :])
            xi[t] = np.exp(a - log_P)

        # responsabilidade por mistura
        gamma_mix = np.empty((T, n, m))
        for i in range(n):
            lg = np.empty((T, m))
            for k in range(m):
                lg[:, k] = self._log_gaussian(X, self.means_[i, k],
                                              self.covars_[i, k])
            log_w = np.log(self.weights_[i] + 1e-300)
            log_wN = log_w[None, :] + lg             # (T, m)
            log_denom = logsumexp(log_wN, axis=1)    # (T,)
            log_resp = log_wN - log_denom[:, None]
            gamma_mix[:, i, :] = gamma[:, i][:, None] * np.exp(log_resp)

        return gamma, gamma_mix, xi, log_P

    def _m_step(self, X, lengths, gammas, gammas_mix, xis):
        n, m = self.n_components, self.n_mix
        D = X.shape[1]

        num = sum(xi.sum(axis=0) for xi in xis)              # (n, n)
        den = sum(g[:-1].sum(axis=0) for g in gammas)        # (n,)
        A = num / (den[:, None] + 1e-10)
        for i in range(n):
            for j in range(n):
                if not (j == i or j == i + 1):
                    A[i, j] = 0.0
            s = A[i].sum()
            if s > 0:
                A[i] /= s
        self.transmat_ = A

        # GMM por estado — agrega em todas as sequências
        N = np.zeros((n, m))
        Sx = np.zeros((n, m, D))
        Sx2 = np.zeros((n, m, D))
        start = 0
        for (L, gm) in zip(lengths, gammas_mix):
            Xi = X[start:start + L]
            start += L
            for i in range(n):
                g = gm[:, i, :]                              # (L, m)
                N[i] += g.sum(axis=0)
                Sx[i] += g.T @ Xi
                Sx2[i] += g.T @ (Xi * Xi)

        for i in range(n):
            total = N[i].sum() + 1e-10
            self.weights_[i] = N[i] / total
            for k in range(m):
                sw = N[i, k] + 1e-10
                mu = Sx[i, k] / sw
                var = Sx2[i, k] / sw - mu * mu
                self.means_[i, k] = mu
                self.covars_[i, k] = np.maximum(var, 1e-4)

    # ---------- API ----------
    def fit(self, X, lengths):
        # X: mfcc concatenadas
        # lengths: numero de frames de cada mfcc

        self._init_topology()
        self._init_gmm(X)

        slices, start = [], 0
        for L in lengths:
            slices.append((start, start + L))
            start += L

        prev_ll = -np.inf
        for it in range(self.n_iter):
            gammas, gammas_mix, xis = [], [], []
            total_ll = 0.0
            for (a, b) in slices:
                g, gm, xi, log_P = self._responsibilities(X[a:b])
                gammas.append(g)
                gammas_mix.append(gm)
                xis.append(xi)
                total_ll += log_P
            self._m_step(X, lengths, gammas, gammas_mix, xis)
            delta = total_ll - prev_ll
            if abs(delta) < self.tol:
                print(f"    convergiu em it={it+1} loglik={total_ll:.2f}")
                break
            prev_ll = total_ll
            if (it + 1) % 5 == 0:
                print(f"    it={it+1} loglik={total_ll:.2f} delta={delta:.4f}")
        return self


    def score(self, X):
        # Viterbi otimizado para topologia left-to-right.
        # Só 2 transições válidas por estado -> substitui logsumexp por max de 2.
        
        T = X.shape[0]
        n = self.n_components
        # print(f"X: {X[0]}")
        log_B = self._log_emission(X)
        # print("log_B: ")
        # for linha in log_B:
        #     for x in linha:
        #         print(f"{x:.2f}", end=" ")
        #     print()
        # exit()
        # Só precisamos de A[i,i] e A[i,i+1]
        log_self = np.log(np.diag(self.transmat_) + 1e-300)        # (n,)
        log_next = np.log(np.diag(self.transmat_, k=1) + 1e-300)   # (n-1,)
        # log_next[i] = log A[i, i+1]

        v = np.full(n, -np.inf)
        v[0] = log_B[0, 0]        # começa sempre no estado 0

        for t in range(1, T):
            v_new = np.empty(n)
            v_new[0] = v[0] + log_self[0] + log_B[t, 0]
            for j in range(1, n):
                # 2 candidatos: veio de j (self-loop) ou de j-1 (avanço)
                stay  = v[j] + log_self[j]
                advan = v[j - 1] + log_next[j - 1]
                v_new[j] = (stay if stay > advan else advan) + log_B[t, j]
            v = v_new
        print(f"{float(np.max(v))}")
        return float(np.max(v))

    def decode(self, X):
        """Viterbi: sequência de estados mais provável."""
        T = X.shape[0]
        n = self.n_components
        log_B = self._log_emission(X)
        log_A = np.log(self.transmat_ + 1e-300)
        log_pi = np.log(self.startprob_ + 1e-300)

        delta = np.full((T, n), -np.inf)
        psi = np.zeros((T, n), dtype=int)
        delta[0] = log_pi + log_B[0]
        for t in range(1, T):
            for j in range(n):
                v = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(v))
                delta[t, j] = v[psi[t, j]] + log_B[t, j]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path