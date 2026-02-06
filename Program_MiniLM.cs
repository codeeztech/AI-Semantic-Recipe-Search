/* Search from AllMiniLmL6V2Sharp */

using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using AllMiniLmL6V2Sharp;
using Microsoft.Data.SqlClient;

namespace MiniLmRecipeSearch
{
    internal class Program_MiniLM
    {
        private const string ConnectionString =
 "Data Source=MIS-SHAHROOZ\\SQLEXPRESS;Initial Catalog=db_aa7f0b_bodbdevelopment;" +
 "Integrated Security=True;MultipleActiveResultSets=True;TrustServerCertificate=True;";

        private static async Task Main0()
        {
            Console.WriteLine("🔄 Loading MiniLM model...");

            using var embedder = new AllMiniLmL6V2Embedder("model/model.onnx");

            await RunSearchAsync(embedder);

            Console.WriteLine("✅ Done. Press any key to exit...");
            Console.ReadKey();
        }

        private static async Task RunSearchAsync(AllMiniLmL6V2Embedder embedder)
        {
            Console.Write("🔍 Enter search phrase: ");
            string input = Console.ReadLine()?.Trim() ?? "";

            // Generate vector for input
            float[] queryVector = embedder.GenerateEmbedding(input).ToArray();

            // Load all recipes with vectors
            var recipes = await LoadRecipeEmbeddingsAsync();

            // Search by cosine similarity
            var results = recipes
                .Select(r => new { r.Id, r.Name, Score = CosineSimilarity(queryVector, r.Vector) })
                .Where(r => r.Score > 0.35f)
                .OrderByDescending(r => r.Score)
                .Take(20)
                .ToList();

            Console.WriteLine("\n📌 Top Matches:");
            foreach (var r in results)
                Console.WriteLine($"[{r.Score:F3}] {r.Name} (ID: {r.Id})");
        }

        private static float CosineSimilarity(float[] v1, float[] v2)
        {
            float dot = 0f, normA = 0f, normB = 0f;

            for (int i = 0; i < v1.Length; i++)
            {
                dot += v1[i] * v2[i];
                normA += v1[i] * v1[i];
                normB += v2[i] * v2[i];
            }

            return dot / (float)(Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-10);
        }

        private static async Task<List<(long Id, string Name, float[] Vector)>> LoadRecipeEmbeddingsAsync()
        {
            const string sql = @"
                SELECT RecipeId, RecipeName, EmbeddingMiniLm
                FROM MstrRecipes
                WHERE EmbeddingMiniLm IS NOT NULL";

            var results = new List<(long, string, float[])>();

            await using var conn = new SqlConnection(ConnectionString);
            await conn.OpenAsync();

            await using var cmd = new SqlCommand(sql, conn);
            await using var reader = await cmd.ExecuteReaderAsync();

            while (await reader.ReadAsync())
            {
                long id = reader.GetInt64(0);
                string name = reader.GetString(1);
                byte[] embeddingBytes = (byte[])reader["EmbeddingMiniLm"];
                float[] vector = BytesToFloatArray(embeddingBytes);

                results.Add((id, name, vector));
            }

            return results;
        }

        private static float[] BytesToFloatArray(byte[] bytes)
        {
            var floats = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
            return floats;
        }
    }
}
