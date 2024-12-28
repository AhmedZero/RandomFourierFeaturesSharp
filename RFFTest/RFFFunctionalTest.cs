using TorchSharp;
using TorchSharp.Modules;
using static RandomFourierFeaturesSharp.RFFFunctional;
using static RandomFourierFeaturesSharp.RFFDataLoader;

namespace RFFTest
{
    public class RFFFunctionalTest
    {
        [Fact]
        public void TestSampleB()
        {
            using var b = SampleB(1.0, 3, 4);
            Assert.Equal([3, 4], b.shape);
        }

        [Fact]
        public void TestGaussianEncoding()
        {
            using var v = RectangularCoordinates([2, 2]);
            using var b = torch.eye(2);
            using var gamma_v = GaussianEncoding(v, b);

            Assert.Equal([2, 2, 4], gamma_v.shape);

            using var xc = torch.cos(2 * Math.PI * v);
            using var yc = torch.sin(2 * Math.PI * v);
            using var gamma_v_expected = torch.cat([xc, yc], dim: -1);

            Assert.True(gamma_v.allclose(gamma_v_expected, atol: 1e-5));
        }

        [Fact]
        public void TestBasicEncoding()
        {
            using var v = RectangularCoordinates([2, 2]);
            using var gamma_v = BasicEncoding(v);

            Assert.Equal([2, 2, 4], gamma_v.shape);

            using var xc = torch.cos(2 * Math.PI * v);
            using var yc = torch.sin(2 * Math.PI * v);
            using var gamma_v_expected = torch.cat([xc, yc], dim: -1);

            Assert.True(gamma_v.allclose(gamma_v_expected, atol: 1e-5));
        }

        [Fact]
        public void TestPositionalEncoding()
        {
            using var v = RectangularCoordinates([2, 2]);
            using var gamma_v = PositionalEncoding(v, sigma: 1.0, m: 1);

            Assert.Equal([2, 2, 4], gamma_v.shape);

            using var xc = torch.cos(2 * Math.PI * v);
            using var yc = torch.sin(2 * Math.PI * v);
            using var unsqueezexc = xc.unsqueeze(-1);
            using var unsqueezeyc = yc.unsqueeze(-1);
            using var gamma_v_expected = torch.cat([unsqueezexc, unsqueezeyc], dim: -1);
            using var reshapegamma_v_expected = gamma_v_expected.reshape([2, 2, 4]);

            Assert.True(gamma_v.allclose(reshapegamma_v_expected, atol: 1e-5));
        }
    }
}
