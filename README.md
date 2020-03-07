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
    out = x * w + b            # differentiable
    out = round_func(out)      # non-differentiable (obfuscated gradients)
    out = out * 0.1            # differentiable
    out = torch.sigmoid(out)   # differentiable
    return out


# compare the two
x = torch.tensor([2.4, 3.5], requires_grad=True)
out = forward(x, round_func_normal)
loss = torch.nn.functional.smooth_l1_loss(out, torch.tensor([0.4, 0.62]))
loss.backward()
print("output:", loss, ", x.grad:", x.grad)

x = torch.tensor([2.4, 3.5], requires_grad=True)
out = forward(x, round_func_BPDA)
loss = torch.nn.functional.smooth_l1_loss(out, torch.tensor([0.4, 0.62]))
loss.backward()
print("output:", loss, ", x.grad:", x.grad)

```
  
[1]: [Athalye, Anish, Nicholas Carlini, and David Wagner. "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples." arXiv preprint arXiv:1802.00420 (2018).](https://github.com/anishathalye/obfuscated-gradients)
