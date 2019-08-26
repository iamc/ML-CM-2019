"""
this is only for a illustrative purpose, see this paper for a serious study
https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.042145
"""

import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)
torch.manual_seed(42)

d = 2  # fixed
D = 16  # bond dimension of boundary MPS

B = 0.01 * torch.randn(d, D, D)
# symmetrize initial boundary PEPS
B = (B + B.permute(0, 2, 1)) / 2.0
A = torch.nn.Parameter(B.view(d, D ** 2))

# local tensor for dimer covering
T = torch.zeros(d, d, d, d)
T[0, 0, 0, 1] = 1.0
T[0, 0, 1, 0] = 1.0
T[0, 1, 0, 0] = 1.0
T[1, 0, 0, 0] = 1.0
T = T.view(d, d ** 2, d)

optimizer = torch.optim.LBFGS([A], max_iter=20)


def symmetrize(A):
    Asymm = A.view(d, D, D)
    Asymm = (Asymm + Asymm.permute(0, 2, 1)) / 2
    return Asymm.view(d, D ** 2)


def closure():
    optimizer.zero_grad()
    Asymm = symmetrize(A)

    T1 = (
        torch.einsum("xa,xby,yc", (Asymm, T, Asymm))
        .view(D, D, d, d, D, D)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(D ** 2 * d, D ** 2 * d)
    )
    T2 = (
        torch.einsum("xa,xb", (Asymm, Asymm))
        .view(D, D, D, D)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(D ** 2, D ** 2)
    )

    eigval_1, _ = torch.symeig(T1, eigenvectors=True)
    eigval_2, _ = torch.symeig(T2, eigenvectors=True)

    lnT = torch.log(eigval_1.max())
    lnZ = torch.log(eigval_2.max())
    loss = -lnT + lnZ
    loss.backward()
    return loss


exact = 0.2915609040  # G/pi
Nepoch = 100
for epoch in range(Nepoch):
    loss = optimizer.step(closure)
    print(
        "epoch {}, entropy density {:.16f}, relative error {:.16f}".format(
            epoch, -loss.item(), (-loss.item() - exact) / exact
        )
    )
