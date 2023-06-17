import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input):
        # print '0000000', input.device
        m = input.detach().cpu().numpy().astype(np.float_)
        batch_size = m.shape[0]
        sqrtm = []
        for i in range(batch_size):
            sqrtm.append(torch.from_numpy(scipy.linalg.sqrtm(m[i]).real).type_as(input))
        sqrtm = torch.stack(sqrtm)
        # print '111111111', sqrtm.device
        sqrtm = sqrtm.to(input.device)
        # print '222222222', sqrtm.device

        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        # print '999999999', grad_output.device
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = []
            sqrtm, = ctx.saved_variables
            sqrtm = sqrtm.detach().cpu().numpy().astype(np.float_)
            gm = grad_output.detach().cpu().numpy().astype(np.float_)
            batch_size = sqrtm.shape[0]
            for i in range(batch_size):
                # Given a positive semi-definite matrix X,
                # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
                # matrix square root dX^{1/2} by solving the Sylvester equation:
                # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
                grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm[i], sqrtm[i], gm[i])

                grad_input.append(torch.from_numpy(grad_sqrtm).type_as(grad_output))
            grad_input = torch.stack(grad_input)
            # print '3333333333', grad_input.device
            # grad_input = grad_input.to(grad_output.device)
            # print '4444444444', grad_input.device

        return grad_input


sqrtm = MatrixSquareRoot.apply


def main():
    from torch.autograd import gradcheck
    k = torch.randn(20, 10).double()
    # Create a positive definite matrix
    pd_mat = k.t().matmul(k).unsqueeze(0)
    pd_mat.requires_grad = True
    test = gradcheck(MatrixSquareRoot.apply, (pd_mat,))
    print(test)


if __name__ == '__main__':
    main()