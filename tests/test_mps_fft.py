import torch
import traceback

print('torch version:', torch.__version__)
print('mps built:', torch.backends.mps.is_built())
print('mps available:', torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device('mps')
    try:
        print('\n--- test: FFT forward/backward ---')
        x = torch.randn(2, 8, requires_grad=True, device=device)
        x_ft = torch.fft.rfftn(x, s=[8], dim=(1,))
        y = torch.view_as_real(x_ft).sum()
        y.backward()
        print('grad ok, x.grad norm =', x.grad.norm().item())
    except Exception as e:
        print('FFT backward failed:')
        traceback.print_exc()

    try:
        print('\n--- test: complex indexed assignment ---')
        new_shape = (2, 5)
        z = torch.zeros(new_shape, dtype=torch.cfloat, device=device, requires_grad=True)
        mask = torch.tensor([True, False, True, True, False], device=device)
        # create a compatible complex block to assign
        block = (torch.randn(2, 3, device=device) + 1j * torch.randn(2, 3, device=device)).to(torch.cfloat)
        z[..., mask] = block
        # try a simple scalar reduction to trigger backward
        loss = torch.view_as_real(z).sum()
        loss.backward()
        print('assignment backward ok, z.grad norm =', z.grad.norm().item())
    except Exception as e:
        print('complex assignment/backward failed:')
        traceback.print_exc()
else:
    print('MPS not available on this machine; nothing to test.')
