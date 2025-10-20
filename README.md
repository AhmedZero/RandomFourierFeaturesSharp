# Random Fourier Features for TorchSharp

[![.NET](https://img.shields.io/badge/.NET-C%23-blue)](https://dotnet.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Random Fourier Features for TorchSharp provides efficient implementations of Gaussian, Basic, and Positional encodings for neural networks in .NET.  
It is based on the paper *“Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains”* by Tancik et al., and inspired by the original [PyTorch implementation](https://github.com/jmclong/random-fourier-features-pytorch).

This library integrates seamlessly with TorchSharp to enable high-frequency feature learning in .NET environments.

---

## Features

- Gaussian Encoding
- Basic Encoding
- Positional Encoding
- Fully compatible with [TorchSharp](https://github.com/dotnet/TorchSharp)
- Designed for efficient tensor operations and GPU acceleration

---

## Installation

To use this package, clone the repository or add it as a project reference in your .NET solution.

```bash
git clone https://github.com/AhmedZero/RandomFourierFeaturesSharp.git
```

---

## Usage

### Gaussian Encoding

```csharp
using RandomFourierFeaturesSharp;
using TorchSharp;
using static TorchSharp.torch;

// Input tensor
using var input = torch.randn(new long[] { 256, 2 });

// Gaussian encoding
var encoding = new RFFLayers.GaussianEncoding(sigma: 10.0, inputSize: 2, encodedSize: 256);
var encoded = encoding.Forward(input);
```

### Basic Encoding

```csharp
using RandomFourierFeaturesSharp;
using TorchSharp;
using static TorchSharp.torch;

// Input tensor
using var input = torch.randn(new long[] { 256, 2 });

// Basic encoding
var encoding = new RFFLayers.BasicEncoding();
var encoded = encoding.Forward(input);
```

### Positional Encoding

```csharp
using RandomFourierFeaturesSharp;
using TorchSharp;
using static TorchSharp.torch;

// Input tensor
using var input = torch.randn(new long[] { 256, 2 });

// Positional encoding
var encoding = new RFFLayers.PositionalEncoding(sigma: 1.0, m: 10);
var encoded = encoding.Forward(input);
```

---

## Integration Example: TorchSharp MLP

```csharp
using TorchSharp;
using static TorchSharp.torch;
using RandomFourierFeaturesSharp;

class ExampleModel : nn.Module
{
    private readonly nn.Module layers;
    private readonly RFFLayers.GaussianEncoding encoding;

    public ExampleModel(int inputSize, int encodedSize, int hidden, double sigma) : base(nameof(ExampleModel))
    {
        encoding = new RFFLayers.GaussianEncoding(sigma, inputSize, encodedSize);

        layers = nn.Sequential(
            ("fc1", nn.Linear(encodedSize, hidden)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(hidden, 1))
        );
    }

    public override Tensor forward(Tensor input)
    {
        var encoded = encoding.Forward(input);
        return layers.forward(encoded);
    }
}

// Example usage
var model = new ExampleModel(inputSize: 2, encodedSize: 256, hidden: 128, sigma: 10.0);
using var x = torch.randn(new long[] { 100, 2 });
var y = model.forward(x);
```

This example demonstrates how Random Fourier Features can be easily combined with standard TorchSharp layers.

---

## Contributing

Pull requests are welcome!  
If you’d like to contribute:

1. Fork the repository  
2. Create a new branch: `git checkout -b feature-name`  
3. Commit your changes: `git commit -m 'Add feature'`  
4. Push to the branch: `git push origin feature-name`  
5. Open a Pull Request

---

## Citation

If you use this repository, please cite it as:

```bibtex
@article{elsayed2024rffcsharp,
  title={Random Fourier Features for TorchSharp},
  author={Ahmed Elsayed Helal Mohammed},
  journal={GitHub. Note: https://github.com/AhmedZero/RandomFourierFeaturesSharp},
  year={2024}
}
```

Also cite the original work:

```bibtex
@misc{tancik2020fourier,
  title={Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains}, 
  author={Matthew Tancik and Pratul P. Srinivasan and Ben Mildenhall and Sara Fridovich-Keil and Nithin Raghavan and Utkarsh Singhal and Ravi Ramamoorthi and Jonathan T. Barron and Ren Ng},
  year={2020},
  eprint={2006.10739},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
