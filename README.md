# Random Fourier Features for C#

[![.NET](https://img.shields.io/badge/.NET-C%23-blue)](https://dotnet.microsoft.com/)

This repository is a C# implementation of the "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" paper by Tancik et al., inspired by the original [Python implementation](https://github.com/jmclong/random-fourier-features-pytorch). It is designed to seamlessly integrate with any .NET project.

## Features

- Gaussian Encoding
- Basic Encoding
- Positional Encoding
- Designed for efficient tensor operations in .NET environments

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
var encoded = encoding.forward(input);
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
var encoded = encoding.forward(input);
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
var encoded = encoding.forward(input);
```

## Contributing

Pull requests are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## Citation

If you use this repository, please cite it as:

```bibtex
@article{long2025rffcsharp,
  title={Random Fourier Features for C#},
  author={Ahmed Elsayed},
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
