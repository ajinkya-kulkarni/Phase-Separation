if dimension == '2D':
    
    # adapted from https://mail.scipy.org/pipermail/scipy-user/2013-June/034744.html

    def halton(dim: int, nbpts: int):
        h = np.full(nbpts * dim, np.nan)
        p = np.full(nbpts, np.nan)
        P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        lognbpts = math.log(nbpts + 1)
        for i in range(dim):
            b = P[i]
            n = int(math.ceil(lognbpts / math.log(b)))
            for t in range(n):
                p[t] = pow(b, -(t + 1))

            for j in range(nbpts):
                d = j + 1
                sum_ = math.fmod(d, b) * p[0]
                for t in range(1, n):
                    d = math.floor(d / b)
                    sum_ += math.fmod(d, b) * p[t]

                h[j*dim + i] = sum_
        return h.reshape(nbpts, dim)

    seq = halton(2, N_DROPLETS)

    temp_positions = np.vstack((SYSTEM_SIZE*seq[:,0],
                                SYSTEM_SIZE*seq[:,1]))

    position_of_droplets = temp_positions.T