# Pytorch BPDA

A simple way to implement the **Backward Pass Differentiable Approximation (BPDA) [1]** in Pytorch.  


```python
import torch


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
    w = torch.tensor([2.3])
    b = torch.tensor([5.7])
    y1 = x * w            # differentiable
    y2 = y1 + b           # differentiable
    y3 = y2 ** 2          # differentiable
    y4 = round_func(y3)   # non-differentiable (obfuscated gradients)
    y5 = y4 * 3           # differentiable
    y6 = y5.sum()         # differentiable
    return y6


# compare the two
x = torch.tensor([2.4, 3.5], requires_grad=True)
out = forward(x, round_func_normal)
out.backward()
print("output:", out, "x.grad:", x.grad)

x = torch.tensor([2.4, 3.5], requires_grad=True)
out = forward(x, round_func_BPDA)
out.backward()
print("output:", out, "x.grad:", x.grad)

```
  
[1]: [Athalye, Anish, Nicholas Carlini, and David Wagner. "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples." arXiv preprint arXiv:1802.00420 (2018).](https://github.com/anishathalye/obfuscated-gradients)
