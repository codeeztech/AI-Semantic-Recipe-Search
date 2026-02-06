using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Data.SqlClient;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AllMiniLmL6V2Sharp;
using AllMiniLmL6V2Sharp.Tokenizer;

namespace AISemanticRecipeSearch
{
    internal class Program_Ensemble
    {
        /* ───────── CONFIG ───────── */

        private const string ConnectionString =
            "Data Source=YOUR_SERVER_NAME;Initial Catalog=YOUR_DATABASE_NAME;" +
    "User Id=YOUR_DB_USERNAME;Password=YOUR_DB_PASSWORD;" +
    "TrustServerCertificate=True;";

        private const string OpenAIModel = "text-embedding-3-large";
        private static readonly HttpClient httpClient = new HttpClient();

        private const string MiniLmModelPath = "model/model.onnx";
        private const string BgeModelPath = "model/bge-small-en-v1.5.onnx";
        private const string BgeVocabPath = "model/vocab.txt";

        private const int BgeMaxLen = 512;
        private const int BgeDim = 384;

        /* ───────── MAIN ───────── */

        static async Task Main()
        {
            Console.WriteLine("🍽️ AI Multi-Model Semantic Recipe Search");

            using var miniLm = new AllMiniLmL6V2Embedder(MiniLmModelPath);
            var tokenizer = new BertTokenizer(BgeVocabPath);
            using var bgeSession = new InferenceSession(BgeModelPath);

            while (true)
            {
                Console.Write("\nEnter meal phrase (blank to exit): ");
                var input = Console.ReadLine()?.Trim();
                if (string.IsNullOrWhiteSpace(input)) break;

                await RunOpenAISearch(input);
                await RunMiniLmSearch(input, miniLm);
                await RunBgeSearch(input, tokenizer, bgeSession);
            }
        }

        /* ───────── OPENAI SEARCH ───────── */

        static async Task RunOpenAISearch(string input)
        {
            Console.WriteLine("\n🔵 OpenAI Search");

            var key = Environment.GetEnvironmentVariable("OPENAI_KEY");
            if (string.IsNullOrEmpty(key))
            {
                Console.WriteLine("⚠️ OPENAI_KEY not set.");
                return;
            }

            var embedding = await GenerateOpenAIEmbedding(input, key);
            var recipes = await LoadRecipes("Embedding");

            var results = recipes
                .Select(r => new
                {
                    r.Id,
                    r.Name,
                    Score = CosineSimilarity(embedding, r.Vector)
                })
                .OrderByDescending(r => r.Score)
                .Take(5);

            foreach (var r in results)
                Console.WriteLine($"[{r.Score:F3}] {r.Name}");
        }

        static async Task<List<float>> GenerateOpenAIEmbedding(string text, string key)
        {
            var payload = new
            {
                input = text,
                model = OpenAIModel,
                encoding_format = "float"
            };

            var request = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/embeddings");
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
            request.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

            var response = await httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var parsed = JsonSerializer.Deserialize<OpenAIResponse>(json);

            return parsed.data[0].embedding;
        }

        /* ───────── MINILM SEARCH ───────── */

        static async Task RunMiniLmSearch(string input, AllMiniLmL6V2Embedder embedder)
        {
            Console.WriteLine("\n🟢 MiniLM Search");

            var queryVector = embedder.GenerateEmbedding(input).ToArray();
            var recipes = await LoadRecipes("EmbeddingMiniLm");

            var results = recipes
                .Select(r => new
                {
                    r.Id,
                    r.Name,
                    Score = CosineSimilarity(queryVector, r.Vector)
                })
                 .Where(r => r.Score >= 0.40f) // <-- increased from 0.35
                .OrderByDescending(r => r.Score)
                .Take(5);

            foreach (var r in results)
                Console.WriteLine($"[{r.Score:F3}] {r.Name}");
        }

        /* ───────── BGE SEARCH ───────── */

        static async Task RunBgeSearch(string input, BertTokenizer tokenizer, InferenceSession session)
        {
            Console.WriteLine("\n🟣 BGE Search");

            var queryVector = EmbedBge(input, tokenizer, session);
            var recipes = await LoadRecipes("EmbeddingBGE");

            var results = recipes
                .Select(r => new
                {
                    r.Id,
                    r.Name,
                    Score = CosineSimilarity(queryVector, r.Vector)
                })
                 .Where(r => r.Score >= 0.61f) // <-- increased from 0.35
                .OrderByDescending(r => r.Score)
                .Take(5);

            foreach (var r in results)
                Console.WriteLine($"[{r.Score:F3}] {r.Name}");
        }

        static float[] EmbedBge(string text, BertTokenizer tok, InferenceSession sess)
        {
            var toks = tok.Encode(BgeMaxLen, text).ToArray();
            var idsArr = toks.Select(t => (long)t.InputIds).ToArray();
            var maskArr = idsArr.Select(id => id == 0 ? 0L : 1L).ToArray();

            var idT = new DenseTensor<long>(idsArr, new[] { 1, BgeMaxLen });
            var maskT = new DenseTensor<long>(maskArr, new[] { 1, BgeMaxLen });
            var typeT = new DenseTensor<long>(new long[BgeMaxLen], new[] { 1, BgeMaxLen });

            using var results = sess.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("input_ids", idT),
                NamedOnnxValue.CreateFromTensor("attention_mask", maskT),
                NamedOnnxValue.CreateFromTensor("token_type_ids", typeT)
            });

            var h = results.First().AsTensor<float>();
            var vec = new float[BgeDim];
            int valid = 0;

            for (int t = 0; t < BgeMaxLen; t++)
            {
                if (maskArr[t] == 0) continue;
                valid++;
                for (int d = 0; d < BgeDim; d++)
                    vec[d] += h[0, t, d];
            }

            for (int d = 0; d < BgeDim; d++)
                vec[d] /= valid;

            float norm = (float)Math.Sqrt(vec.Sum(v => v * v)) + 1e-9f;
            for (int d = 0; d < BgeDim; d++)
                vec[d] /= norm;

            return vec;
        }

        /* ───────── SHARED UTILITIES ───────── */

        static async Task<List<(long Id, string Name, float[] Vector)>> LoadRecipes(string column)
        {
            var list = new List<(long, string, float[])>();

            string sql = $@"
                SELECT RecipeId, RecipeName, {column}
                FROM MstrRecipes
                WHERE {column} IS NOT NULL";

            await using var conn = new SqlConnection(ConnectionString);
            await conn.OpenAsync();

            await using var cmd = new SqlCommand(sql, conn);
            await using var reader = await cmd.ExecuteReaderAsync();

            while (await reader.ReadAsync())
            {
                long id = reader.GetInt64(0);
                string name = reader.GetString(1);
                byte[] bin = (byte[])reader[2];

                list.Add((id, name, BytesToFloatArray(bin)));
            }

            return list;
        }

        static float CosineSimilarity(IList<float> v1, IList<float> v2)
        {
            float dot = 0, mag1 = 0, mag2 = 0;

            for (int i = 0; i < v1.Count; i++)
            {
                dot += v1[i] * v2[i];
                mag1 += v1[i] * v1[i];
                mag2 += v2[i] * v2[i];
            }

            return dot / ((float)Math.Sqrt(mag1) * (float)Math.Sqrt(mag2) + 1e-10f);
        }

        static float[] BytesToFloatArray(byte[] bytes)
        {
            var floats = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
            return floats;
        }
    }

    /* ───────── OPENAI RESPONSE MODELS ───────── */

    public class OpenAIResponse
    {
        public List<EmbeddingData> data { get; set; }
    }

    public class EmbeddingData
    {
        public List<float> embedding { get; set; }
    }
}
