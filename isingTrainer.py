import netket as nk
import numpy as np
import pandas as pd
import ray

# from netket.experimental import QSR
from netket.experimental.models.fast_rnn import FastLSTMNet
from netket.operator.spin import sigmax, sigmaz
from optax import adamw

from regularizedQSR import QSR

nSpins = 32

sx = lambda a, b: (1 / 2) * sigmax(a, b)
sz = lambda a, b: (1 / 2) * sigmaz(a, b)


@ray.remote(num_cpus=1)
def runIsing(h):
    def callback(step, log, driver):
        if step % 100 == 0:
            indices = np.random.randint(len(sigmas), size=(1000,))
            logPTheta = 2 * driver.state.log_value(sigmas[indices])
            log["KL"] = np.sum(logPs[indices][:, 0] - logPTheta)
            print(log["KL"])
            log["Energy"] = driver.state.expect(H).mean.real
        return True

    for HType in ["XX+Z"]:
        sigmas = pd.read_csv(
            f"traindata/isingTrainingSet_{HType}_h={h}.csv", delimiter="\t", header=None
        ).to_numpy()[:, :nSpins]
        ps = pd.read_csv(
            f"traindata/ps_{HType}_h={h}.csv", delimiter="\t", header=None
        ).to_numpy()
        logPs = np.log(ps)

        Us = len(sigmas) * [nSpins * "I"]

        g = nk.graph.Hypercube(length=nSpins, n_dim=1, pbc=True)
        hilbert = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
        sampler = nk.sampler.ARDirectSampler(hilbert=hilbert)
        # sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=16)

        ######################################################################################################
        # num_layers = 2  # 4
        # d_model = 10  # 128 #128
        # dff = 8  # 128 # 512
        # num_heads = 2  # 8
        #
        # model = Transformer(
        #     hilbert=hilbert,
        #     num_layers=num_layers,
        #     num_heads=num_heads,
        #     d_model=d_model,
        #     dff=dff,
        #     autoreg=True,
        # )
        model = FastLSTMNet(layers=3, features=10, hilbert=hilbert, graph=g)
        # model = nk.models.RBM(alpha=1.5, param_dtype=float)
        # model = SymmetrizedRBM(alpha=1.5)
        ######################################################################################################
        vstate = nk.vqs.MCState(sampler, model=model, n_samples=256)
        opt = adamw(5e-5)
        print(vstate.n_parameters)

        if HType == "XX+Z":
            H = sum(
                [
                    -h * sz(hilbert, i)
                    - 4.0 * sx(hilbert, i) * sx(hilbert, (i + 1) % hilbert.size)
                    for i in range(hilbert.size)
                ]
            )
        elif HType == "ZZ+X":
            H = sum(
                [
                    -h * sx(hilbert, i)
                    - 4.0 * sz(hilbert, i) * sz(hilbert, (i + 1) % hilbert.size)
                    for i in range(hilbert.size)
                ]
            )

        qsr = QSR(
            training_data=(sigmas, Us),
            training_batch_size=128,
            optimizer=opt,
            variational_state=vstate,
        )
        qsr.run(out=f"logs/{HType}_RNN_h={h}", n_iter=100000, callback=callback)


if __name__ == "__main__":
    hs = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    ids = [runIsing.remote(h) for h in hs]
    ray.get(ids)
