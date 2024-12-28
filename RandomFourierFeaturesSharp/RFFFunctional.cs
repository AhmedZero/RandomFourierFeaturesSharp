using static TorchSharp.torch;

namespace RandomFourierFeaturesSharp
{
    public static class RFFFunctional
    {
        /// <summary>
        /// Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, \sigma^2)`
        /// </summary>
        /// <param name="sigma">standard deviation</param>
        /// <param name="size">size of the matrix sampled</param>
        public static Tensor SampleB(double sigma, params long[] size)
        {
            using var randsize = randn(size);
            return randsize * sigma;
        }

        /// <summary>
        /// Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`
        /// </summary>
        /// <param name="v">input tensor of shape :math:`(N, *, \text{input_size})`</param>
        /// <param name="B">projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`</param>
        /// <returns>Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`</returns>
        public static Tensor GaussianEncoding(Tensor v, Tensor B)
        {
            using var btrans = B.transpose(-1, -2);
            using var vp = 2 * Math.PI * v.matmul(btrans);
            using var cvp = vp.cos();
            using var svp = vp.sin();
            return cat([cvp, svp], dim: -1);
        }

        public static Tensor BasicEncoding(Tensor v)
        {
            var vp = 2 * Math.PI * v;
            using var cvp = vp.cos();
            using var svp = vp.sin();
            return cat([cvp, svp], dim: -1);
        }
        public static Tensor PositionalEncoding(Tensor v, double sigma, int m)
        {
            using var j = arange(m, device: v.device);
            using var tensorpow = pow(sigma, j / (double)m);
            using var coeffs = 2 * Math.PI * tensorpow;
            using var unsqueezev = unsqueeze(v, -1);
            using var vp = coeffs * unsqueezev;
            using var svp = vp.sin();
            using var cvp = vp.cos();
            using var vp_cat = cat([cvp, svp], dim: -1);
            return vp_cat.flatten(-2, -1);
        }

    }
}
