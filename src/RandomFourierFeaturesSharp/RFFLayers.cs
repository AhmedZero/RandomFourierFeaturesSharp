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
            private readonly Tensor b;

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
            }

            public override Tensor forward(Tensor x)
            {
                return RFFFunctional.GaussianEncoding(x, b);
            }
        }
        public class BasicEncoding() : Module<Tensor, Tensor>(nameof(BasicEncoding))
        {
            public override Tensor forward(Tensor x)
            {
                return RFFFunctional.BasicEncoding(x);
            }

        }
        public class PositionalEncoding(double sigma, int m) : Module<Tensor, Tensor>(nameof(PositionalEncoding))
        {
            private readonly double sigma = sigma;
            private readonly int m = m;

            public override Tensor forward(Tensor v)
            {
                return RFFFunctional.PositionalEncoding(v, sigma, m);
            }
        }
    }
}
