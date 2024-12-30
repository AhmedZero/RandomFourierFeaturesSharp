using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;

namespace RandomFourierFeaturesSharp
{
    public static class RFFLayers
    {
        public class GaussianEncoding : Module<Tensor,Tensor>
        {
            private Tensor b;

            public GaussianEncoding(double? sigma = null, double? inputSize = null, double? encodedSize = null, Tensor? b = null) : base(nameof(GaussianEncoding))
            {
                if (b is null)
                {
                    if (sigma is null || inputSize is null || encodedSize is null)
                    {
                        throw new ArgumentException("Arguments 'sigma,' 'inputSize,' and 'encodedSize' are required.");
                    }
                    this.b = RFFFunctional.SampleB((double)sigma, (long)encodedSize, (long)inputSize);
                }
                else
                {
                    if (sigma is not null || inputSize is not null || encodedSize is not null)
                    {
                        throw new ArgumentException("Only specify the 'b' argument when using it.");
                    }
                    this.b = b;
                }
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                return RFFFunctional.GaussianEncoding(x, b);
            }
        }
        public class BasicEncoding : Module<Tensor, Tensor>
        {
            public BasicEncoding() : base(nameof(BasicEncoding))
            {
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                return RFFFunctional.BasicEncoding(x);
            }

        }
        public class PositionalEncoding : Module<Tensor, Tensor>
        {
            private readonly double sigma;
            private readonly int m;

            public PositionalEncoding(double sigma, int m) : base(nameof(PositionalEncoding))
            {
                this.sigma = sigma;
                this.m = m;
                RegisterComponents();
            }

            public override Tensor forward(Tensor v)
            {
                return RFFFunctional.PositionalEncoding(v, sigma, m);
            }
        }
    }
}
