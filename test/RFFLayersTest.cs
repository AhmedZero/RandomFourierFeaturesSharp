using TorchSharp;
using static RandomFourierFeaturesSharp.RFFFunctional;
using static RandomFourierFeaturesSharp.RFFDataLoader;
using static RandomFourierFeaturesSharp.RFFLayers;


namespace RFFTest
{
    public class RFFLayersTest
    {
        private readonly DeviceType device = torch.cuda_is_available() ? DeviceType.CUDA : DeviceType.CPU;
        [Fact]
        public void TestGaussianEncoding()
        {
            using var b = SampleB(1.0, 256, 2).to(device, disposeAfter: true);
            using var layer = new GaussianEncoding(b: b).to(device);
            using var v = RectangularCoordinates(256, 256).to(device, disposeAfter: true);
            using var gamma_v = layer.forward(v);
            using var gamma_v_expected = GaussianEncoding(v, b);
            Assert.True(gamma_v.cpu().allclose(gamma_v_expected.cpu(), atol: 1e-5));
        }

        [Fact]
        public void TestGaussianEncodingUnFreeze()
        {
            using var b = SampleB(1.0, 256, 2).to(device, disposeAfter: true);
            using var layer = new GaussianEncoding(b: b).to(device);
            foreach (var p in layer.parameters())
            {
                p.requires_grad = true;
            }
            Assert.False(layer.b.requires_grad);

        }

        [Fact]
        public void TestGaussianEncodingRegisterBuffer()
        {
            using var b = SampleB(1.0, 256, 2).to(device, disposeAfter: true);
            using var layer = new GaussianEncoding(b: b).to(device);
            Assert.True(layer.state_dict().ContainsKey("b"));
            var bufferB = layer.state_dict()["b"];
            Assert.True(bufferB.allclose(b, rtol: 1e-5, atol: 1e-5));

        }

        [Fact]
        public void TestBasicEncoding()
        {
            using var layer = new BasicEncoding().to(device);
            using var v = RectangularCoordinates(256, 256).to(device, disposeAfter: true);
            using var gamma_v = layer.forward(v);
            using var gamma_v_expected = BasicEncoding(v);
            Assert.True(gamma_v.cpu().allclose(gamma_v_expected.cpu(), atol: 1e-5));
        }

        [Fact]
        public void TestPositionalEncoding()
        {
            using var layer = new PositionalEncoding(sigma: 1f, m: 10).to(device);
            using var v = RectangularCoordinates(256, 256).to(device, disposeAfter: true);
            using var gamma_v = layer.forward(v);
            using var gamma_v_expected = PositionalEncoding(v, sigma: 1f, m: 10);
            Assert.True(gamma_v.cpu().allclose(gamma_v_expected.cpu(), atol: 1e-5));
        }

    }
}
