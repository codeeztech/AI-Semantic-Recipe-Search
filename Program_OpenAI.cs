/* Semantic Open AI Search */

using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Data.SqlClient;

namespace RecipeSearchApp
{
    class Recipe
    {
        public long RecipeId { get; set; }
        public string RecipeName { get; set; }
        public string Description { get; set; }
        public byte[] Embedding { get; set; }
    }

    class EmbeddingData
    {
        public List<float> embedding { get; set; }
    }

    class EmbeddingResponse
    {
        public List<EmbeddingData> data { get; set; }
    }

    class Program
    {

        private const string connectionString =
  "Data Source=MIS-SHAHROOZ\\SQLEXPRESS;Initial Catalog=db_aa7f0b_bodbdevelopment;" +
  "Integrated Security=True;MultipleActiveResultSets=True;TrustServerCertificate=True;";


        const string openAiKey = "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"; // Replace with your OpenAI key
        const string embeddingModel = "text-embedding-3-large";
        static readonly HttpClient client = new HttpClient();

        static async Task Main0()
        {
            Console.WriteLine("🔍 Recipe Semantic Open AI Search");

            while (true)
            {
                Console.Write("\nEnter a meal phrase to search (or blank to exit): ");
                var input = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(input)) break;

                try
                {
                    var inputEmbedding = await GenerateEmbedding(input);
                    var recipes = await LoadRecipesWithEmbeddings();

                    var results = recipes
                        .Select(r => (Recipe: r, Score: CosineSimilarity(inputEmbedding, BytesToFloatArray(r.Embedding))))
                        .OrderByDescending(r => r.Score)
                        .Take(10)
                        .ToList();

                    Console.WriteLine($"\nTop matches for: \"{input}\"");
                    foreach (var r in results)
                    {
                        Console.WriteLine($"\n➡️ {r.Recipe.RecipeName} [{r.Score:F3}]");
                        //Console.WriteLine($"\n➡️ {r.Recipe.RecipeName} [{r.Score:F3}]\n   {r.Recipe.Description}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Error: {ex.Message}");
                }
            }
        }

        static async Task<List<float>> GenerateEmbedding(string text)
        {
            var payload = new
            {
                input = text,
                model = embeddingModel,
                encoding_format = "float"
            };

            var request = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/embeddings");
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", openAiKey);
            request.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

            var response = await client.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            var parsed = JsonSerializer.Deserialize<EmbeddingResponse>(json);
            return parsed.data[0].embedding;
        }

        static async Task<List<Recipe>> LoadRecipesWithEmbeddings()
        {
            var recipes = new List<Recipe>();

            using var conn = new SqlConnection(connectionString);
            await conn.OpenAsync();

            using var cmd = new SqlCommand("SELECT RecipeId, RecipeName, Description, Embedding FROM MstrRecipes WHERE Embedding IS NOT NULL", conn);
            using var reader = await cmd.ExecuteReaderAsync();

            while (await reader.ReadAsync())
            {
                recipes.Add(new Recipe
                {
                    RecipeId = reader.GetInt64(0),
                    RecipeName = reader.GetString(1),
                    Description = reader.IsDBNull(2) ? "" : reader.GetString(2),
                    Embedding = (byte[])reader["Embedding"]
                });
            }

            return recipes;
        }

        static float[] BytesToFloatArray(byte[] bytes)
        {
            var floats = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
            return floats;
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

            return dot / ((float)Math.Sqrt(mag1) * (float)Math.Sqrt(mag2) + 1e-8f);
        }
    }
}
