using TorchSharp;
using TorchSharp.Modules;
using static RandomFourierFeaturesSharp.RFFDataLoader;
namespace RFFTest
{
    public class RFFDataLoaderTest
    {
        [Fact]
        public void TestRectangularCoordinatesShape()
        {
            var size = new long[] { 3, 4, 5 };
            using var coords = RectangularCoordinates(size);
            Assert.Equal([3, 4, 5, size.Length], coords.shape);
        }


        [Fact]
        public void TestRectangularCoordinatesValue()
        {
            var size = new long[] { 2, 2 };
            using var coords = RectangularCoordinates(size);
            using var expectedCoords = torch.tensor(new float[,,]
            {
            { { 0, 0 }, { 0, 1 } },
            { { 1, 0 }, { 1, 1 } }
            });

            Assert.Equal(expectedCoords, coords);
        }

        [Fact]
        public void TestToDataset()
        {
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);
            using var dataset = ToDataset("images/cat.jpg");
            Assert.NotNull(dataset);
            Assert.IsType<TensorDataset>(dataset);
        }
    }
}
