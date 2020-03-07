# Pytorch BPDA

A simple way to implement the **Backward Pass Differentiable Approximation (BPDA) [1]** in a Pytorch model.
  

```python
import torch
import torch.nn.functional as F


def round_func_normal(input):
    out = torch.round(input)
    return out


def round_func_BPDA(input):
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def forward(x, round_func):
    linear = torch.nn.Linear(2, 1)
    linear.weight.data = torch.tensor([-0.3917, 0.2402])
    linear.bias.data = torch.tensor([-0.3856])
    out = linear(x)            # differentiable
    out = out * 10             # differentiable
    out = round_func(out)      # defended by non-differentiable operation (shattered gradients)
    out = out * 0.01           # differentiable
    out = torch.sigmoid(out)   # differentiable
    return out


# compare the three scenarios

# scenario 1: No Defence
x = torch.tensor([4, -1.12]).view(1, 1, -1).requires_grad_(True)
out = forward(x, lambda x: x)
loss = F.binary_cross_entropy(out, torch.tensor([1.]).view(1, -1))
loss.backward()
print(loss)           # tensor(0.8104, grad_fn=<BinaryCrossEntropyBackward>)
print(x.grad)         # tensor([[[ 0.0218, -0.0133]]])

# scenario 2: Defended by round function (shattered gradients)
x = torch.tensor([4, -1.12]).view(1, 1, -1).requires_grad_(True)
out = forward(x, round_func_normal)
loss = F.binary_cross_entropy(out, torch.tensor([1.]).view(1, -1))
loss.backward()
print(loss)           # tensor(0.8092, grad_fn=<BinaryCrossEntropyBackward>)
print(x.grad)         # tensor([[[-0., 0.]]])

# scenario 3: Defended by round function, attached by BPDA
x = torch.tensor([4, -1.12]).view(1, 1, -1).requires_grad_(True)
out = forward(x, round_func_BPDA)
loss = F.binary_cross_entropy(out, torch.tensor([1.]).view(1, -1))
loss.backward()
print(loss)           # tensor(0.8092, grad_fn=<BinaryCrossEntropyBackward>)
print(x.grad)         # tensor([[[ 0.0217, -0.0133]]])

```
  
[1]: [Athalye, Anish, Nicholas Carlini, and David Wagner. "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples." arXiv preprint arXiv:1802.00420 (2018).](https://github.com/anishathalye/obfuscated-gradients)
