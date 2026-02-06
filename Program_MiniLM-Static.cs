/* Search from AllMiniLmL6V2Sharp With foodtypeid additional static search */

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AllMiniLmL6V2Sharp;
using Microsoft.Data.SqlClient;

internal class Program1
{
    private const string ConnectionString =
        "Data Source=YOUR_SERVER_NAME;Initial Catalog=YOUR_DATABASE_NAME;" +
    "User Id=YOUR_DB_USERNAME;Password=YOUR_DB_PASSWORD;" +
    "TrustServerCertificate=True;";

    private static async Task Main0()
    {
        Console.WriteLine("🔄 Loading MiniLM model …");

        string modelPath = "model/model.onnx";

        using var embedder = new AllMiniLmL6V2Embedder(modelPath);

        await RunVectorSearchAsync(embedder);

        Console.WriteLine("🎉 Done. Press any key to exit …");
        Console.ReadKey();
    }

    // 🔍 Accepts user input and finds top similar recipes
    private static async Task RunVectorSearchAsync(AllMiniLmL6V2Embedder embedder)
    {
        Console.Write("🔍 Enter a search phrase: ");
        string input = Console.ReadLine() ?? "";

        Console.WriteLine("Food type filter:");
        Console.WriteLine(" 0 = any (general)");
        Console.WriteLine(" 2 = vegetarian");
        Console.WriteLine(" 3 = non-vegetarian");
        Console.Write("Choose food-type Id (0/2/3): ");
        int foodTypeFilter = int.TryParse(Console.ReadLine(), out var n) ? n : 0;

        // Option A – easiest: call the single-text helper
        var queryVector = embedder.GenerateEmbedding(input).ToArray();   // ✅


        var recipes = await LoadRecipeEmbeddingsAsync(foodTypeFilter);

        var scored = recipes
            .Select(r => new { r.Id, r.Name, Score = CosineSimilarity(queryVector, r.Vector) })
            .Where(r => r.Score > 0.35)
            .OrderByDescending(r => r.Score)
            .Take(20)
            .ToList();

        Console.WriteLine("\n📌 Top Matches:");
        foreach (var item in scored)
            Console.WriteLine($"[{item.Score:F3}] {item.Name} (ID: {item.Id})");
    }

    // 🧠 Cosine similarity calculation
    private static float CosineSimilarity(float[] vec1, float[] vec2)
    {
        float dot = 0f, normA = 0f, normB = 0f;

        for (int i = 0; i < vec1.Length; i++)
        {
            dot += vec1[i] * vec2[i];
            normA += vec1[i] * vec1[i];
            normB += vec2[i] * vec2[i];
        }

        return dot / (float)(Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-10);
    }

    // 📦 Load recipe vectors from SQL Server
    private static async Task<List<(long Id, string Name, float[] Vector)>> LoadRecipeEmbeddingsAsync(int foodTypeId)
    {
        const string sql = "SELECT RecipeId, RecipeName, Embedding FROM MstrRecipes WHERE Embedding IS NOT NULL AND  (@ft = 0 OR FoodTypeId = @ft)";

        var result = new List<(long, string, float[])>();

        await using var conn = new SqlConnection(ConnectionString);
        await conn.OpenAsync();

        await using var cmd = new SqlCommand(sql, conn);

        if (foodTypeId != 0)
            cmd.Parameters.Add("@ft", SqlDbType.Int).Value = foodTypeId;

        await using var rdr = await cmd.ExecuteReaderAsync();
        while (await rdr.ReadAsync())
        {
            long id = rdr.GetInt64(0);
            string name = rdr.GetString(1);
            byte[] vecBytes = (byte[])rdr["Embedding"];
            float[] vec = BytesToFloatArray(vecBytes);

            result.Add((id, name, vec));
        }

        return result;
    }

    // 🔁 Convert byte[] → float[]
    private static float[] BytesToFloatArray(byte[] bytes)
    {
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }
}
