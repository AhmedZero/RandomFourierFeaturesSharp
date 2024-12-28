using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace RandomFourierFeaturesSharp
{
    public static class RFFDataLoader
    {
        /// <summary>
        /// Creates a tensor of equally spaced coordinates for use with an image or volume
        /// </summary>
        /// <param name="size">shape of the image or volume</param>
        /// <returns>Tensor: tensor of shape :math:`(*\text{size}, \text{len(size)})`</returns>
        public static Tensor RectangularCoordinates(params long[] size)
        {
            Tensor? ret = null;
            Tensor[]? linspaces = null;
            Tensor[]? coordinates = null;

            try
            {
                static Tensor LinspaceFunc(long nx) => linspace(0.0, 1.0, nx);

                linspaces = Array.ConvertAll(size, LinspaceFunc);
                coordinates = meshgrid(linspaces, indexing: "ij");
                ret = stack(coordinates, dim: -1);
                return ret;

            }
            finally
            {
                if (linspaces != null)
                {
                    foreach (var tensor in linspaces)
                    {
                        tensor.Dispose();
                    }
                }

                if (coordinates != null)
                {
                    foreach (var tensor in coordinates)
                    {
                        tensor.Dispose();
                    }
                }
            }

        }



        public static TensorDataset ToDataset(string path)
        {
            using var image = torchvision.io.read_image(path);
            using var imagefloat32 = image.to_type(ScalarType.Float32);
            var shape = imagefloat32.shape;
            var H = shape[1];
            var W = shape[2];

            using var coords = RectangularCoordinates(H, W);
            using var perimage = imagefloat32.permute(1, 2, 0);
            using var scaleimage = perimage / 255.0;

            using var flattencoords = coords.flatten(0, -2);
            using var flattenimage = scaleimage.flatten(0, -2);

            return utils.data.TensorDataset(flattencoords, flattenimage);
        }


    }
}
