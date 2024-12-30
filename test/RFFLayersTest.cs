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

    }
}
