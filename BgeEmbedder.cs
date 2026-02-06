using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Tokenizers;
using Tokenizers.HuggingFace.Tokenizer;

namespace AiSymanticSearchMeal
{
    public sealed class BgeEmbedder : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly Tokenizer _tokenizer;
        private readonly int _maxLen = 512;
        private const int Hidden = 384;

        public BgeEmbedder(string modelDir)
        {
            // Load ONNX model
            _session = new InferenceSession(Path.Combine(modelDir, "model.onnx"));

            // Load tokenizer from tokenizer.json (synchronous)
            _tokenizer = Tokenizer.FromFile(Path.Combine(modelDir, "tokenizer.json"));
        }

        public float[] GenerateEmbedding(string text)
        {
            string prompt = $"Represent this sentence for retrieval: {text}";

            // Synchronous encoding (no Task, no await)
            var encoding = _tokenizer.Encode(prompt);

            // Convert Ids and AttentionMask (uint) to long
            var ids = encoding.Ids.Take(_maxLen).Select(x => (long)x).ToList();
            var mask = encoding.AttentionMask.Take(_maxLen).Select(x => (long)x).ToList();

            // Pad ids and mask to maxLen
            while (ids.Count < _maxLen)
            {
                ids.Add(0);
                mask.Add(0);
            }

            // Create input tensors with shape [1, maxLen]
            var idTensor = new DenseTensor<long>(ids.ToArray(), new[] { 1, _maxLen });
            var maskTensor = new DenseTensor<long>(mask.ToArray(), new[] { 1, _maxLen });

            using var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("input_ids", idTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor)
            });

            var output = results.First().AsTensor<float>();

            var emb = new float[Hidden];
            int validTokens = mask.Count(x => x == 1);

            for (int t = 0; t < _maxLen; t++)
            {
                if (mask[t] == 0) continue;

                for (int d = 0; d < Hidden; d++)
                    emb[d] += output[0, t, d];
            }

            if (validTokens > 0)
            {
                for (int d = 0; d < Hidden; d++)
                    emb[d] /= validTokens;
            }

            // L2 normalization
            float norm = MathF.Sqrt(emb.Sum(x => x * x)) + 1e-9f;
            for (int d = 0; d < Hidden; d++)
                emb[d] /= norm;

            return emb;
        }

        public void Dispose()
        {
            _session?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
