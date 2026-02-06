using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Microsoft.Data.SqlClient;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AllMiniLmL6V2Sharp.Tokenizer;
using Porter2StemmerStandard;
internal class Program
{
    /* ───────── CONFIG ───────── */

    private const string ConnStr =
"Data Source=MIS-SHAHROOZ\\SQLEXPRESS;Initial Catalog=db_aa7f0b_bodbdevelopment;" +
"Integrated Security=True;MultipleActiveResultSets=True;TrustServerCertificate=True;";

    private const string ModelPath = "model/bge-small-en-v1.5.onnx";
    private const string VocabPath = "model/vocab.txt";
    private const string EmbedCol = "EmbeddingBGE";

    private const int MaxLen = 512;  // model context
    private const int Dim = 384;  // embedding size
    /* ────────────────────────── */

    private static async Task Main0()
    {
        Console.WriteLine("🔄 Loading BGE model …");
        var tokenizer = new BertTokenizer(VocabPath);
        using var session = new InferenceSession(ModelPath);

        while (true)
        {
            Console.Write("\n🔍 Enter search phrase (blank to exit): ");
            string query = Console.ReadLine()?.Trim() ?? "";
            if (query.Length == 0) break;

            var expandedWords = await PreprocessQueryAsync(query);


            int? proteinType = DetectProteinType(expandedWords);

            var filteredRecipes = await LoadFilteredRecipesAsync(expandedWords, proteinType);

            var qVec = EmbedQuery(string.Join(" ", expandedWords), tokenizer, session);

            var top = filteredRecipes
                .Select(r => new { r.Id, r.Name, Score = Cosine(qVec, r.Vec) })
                .OrderByDescending(x => x.Score)
                .Where(x => x.Score >= 0.65)
                .Take(100)
                .ToList();

            Console.WriteLine($"\n📌 Top matches for “{query}”");
            foreach (var r in top)
                Console.WriteLine($"[{r.Score:F3}] {r.Name} (ID {r.Id})");
        }

    }

    private static float[] EmbedQuery(string text, BertTokenizer tok, InferenceSession sess)
    {
        var toks = tok.Encode(MaxLen, text).ToArray();
        var idsArr = toks.Select(t => (long)t.InputIds).ToArray();
        var maskArr = idsArr.Select(id => id == 0 ? 0L : 1L).ToArray();

        var idT = new DenseTensor<long>(idsArr, new[] { 1, MaxLen });
        var maskT = new DenseTensor<long>(maskArr, new[] { 1, MaxLen });
        var typeT = new DenseTensor<long>(new long[MaxLen], new[] { 1, MaxLen });

        using var res = sess.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("input_ids", idT),
            NamedOnnxValue.CreateFromTensor("attention_mask", maskT),
            NamedOnnxValue.CreateFromTensor("token_type_ids", typeT)
        });

        var h = res.First().AsTensor<float>();  // [1,512,384]
        var vec = new float[Dim];
        int valid = 0;
        for (int t = 0; t < MaxLen; t++)
        {
            if (maskArr[t] == 0) continue;
            valid++;
            for (int d = 0; d < Dim; d++)
                vec[d] += h[0, t, d];
        }
        for (int d = 0; d < Dim; d++)
            vec[d] /= valid;

        // L2-normalise
        float norm = (float)Math.Sqrt(vec.Sum(v => v * v)) + 1e-9f;
        for (int d = 0; d < Dim; d++) vec[d] /= norm;
        return vec;
    }

    private static async Task<List<(long Id, string Name, float[] Vec)>> LoadFilteredRecipesAsync(List<string> expandedWords, int? proteinType)
    {
        var list = new List<(long, string, float[])>();

        if (expandedWords == null || expandedWords.Count == 0)
            return list;

        var whereClause = string.Join(" OR ", expandedWords.Select((w, i) =>
            $"RecipeName LIKE @kw{i} OR Description LIKE @kw{i}"));

        string sql = $@"
        SELECT RecipeId, RecipeName, EmbeddingBGE
        FROM   MstrRecipes
        WHERE  ({whereClause}) AND EmbeddingBGE IS NOT NULL";

        if (proteinType.HasValue)
            sql += " AND ProteinTypeId = @protein";

        await using var conn = new SqlConnection(ConnStr);
        await conn.OpenAsync();

        await using var cmd = new SqlCommand(sql, conn);

        // Add LIKE parameters
        for (int i = 0; i < expandedWords.Count; i++)
            cmd.Parameters.AddWithValue($"@kw{i}", $"%{expandedWords[i]}%");

        // Add optional protein filter
        if (proteinType.HasValue)
            cmd.Parameters.AddWithValue("@protein", proteinType.Value);

        await using var rdr = await cmd.ExecuteReaderAsync();
        while (await rdr.ReadAsync())
        {
            long id = rdr.GetInt64(0);
            string name = rdr.GetString(1);
            byte[] bin = (byte[])rdr[2];
            list.Add((id, name, BytesToFloats(bin)));
        }

        return list;
    }

    private static float Cosine(float[] a, float[] b)
    {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < Dim; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (float)(Math.Sqrt(na) * Math.Sqrt(nb) + 1e-10);
    }

    private static float[] BytesToFloats(byte[] bytes)
    {
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }

    static List<string> PreprocessQuery(string query)
    {
        var romanMap = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
        {
            ["keema"] = new List<string> { "minced meat" },
            ["keema alu"] = new List<string> { "minced meat with potatoes" },
            ["alu"] = new List<string> { "potato" },
            ["aaloo"] = new List<string> { "potato" },
            ["gosht"] = new List<string> { "meat" },
            ["chicken karahi"] = new List<string> { "chicken curry in wok" },
            ["karahi"] = new List<string> { "curry in wok" },
            ["salan"] = new List<string> { "curry" },
            ["shorba"] = new List<string> { "stew" },
            ["dal"] = new List<string> { "lentils" },
            ["daal"] = new List<string> { "lentils" },
            ["sabzi"] = new List<string> { "vegetables" },
            ["sabji"] = new List<string> { "vegetables" },
            ["biryani"] = new List<string> { "biryani" },
            ["pulao"] = new List<string> { "rice pilaf" },
            ["puloe"] = new List<string> { "rice pilaf" },
            ["yakhni"] = new List<string> { "broth" },
            ["bhuna"] = new List<string> { "sautéed" },
            ["masala"] = new List<string> { "spices" },
            ["tikka"] = new List<string> { "grilled chunks" },
            ["boti"] = new List<string> { "meat chunks" },
            ["nihari"] = new List<string> { "slow-cooked meat stew" },
            ["haleem"] = new List<string> { "wheat and lentil stew" },
            ["halwa"] = new List<string> { "dessert pudding" },
            ["paratha"] = new List<string> { "layered flatbread" },
            ["roti"] = new List<string> { "flatbread" },
            ["chapati"] = new List<string> { "flatbread" },
            ["naan"] = new List<string> { "oven-baked bread" },
            ["korma"] = new List<string> { "creamy curry" },
            ["seekh kabab"] = new List<string> { "skewered minced meat" },
            ["kofta"] = new List<string> { "meatballs" },
            ["kofte"] = new List<string> { "meatballs" },
            ["raita"] = new List<string> { "yogurt dip" },
            ["achar"] = new List<string> { "pickle" },
            ["pakora"] = new List<string> { "fritters" },
            ["samosa"] = new List<string> { "stuffed pastry" },
            ["gulab jamun"] = new List<string> { "syrup balls dessert" },
            ["ras malai"] = new List<string> { "cream dessert" },
            ["barfi"] = new List<string> { "milk fudge" },
            ["khichdi"] = new List<string> { "rice and lentils porridge" },
            ["majlis"] = new List<string> { "meat wrap" },
            ["ouzi"] = new List<string> { "roasted rice and meat" },
            ["kata kat"] = new List<string> { "minced meat on griddle" },
            ["beef"] = new List<string> { "beef" },
            ["mutton"] = new List<string> { "mutton" },
            ["chicken"] = new List<string> { "chicken" },
            ["fish"] = new List<string> { "fish" },
            ["egg"] = new List<string> { "egg" },
            ["boiled egg"] = new List<string> { "boiled egg" },
            ["fried egg"] = new List<string> { "fried egg" },
            ["omelette"] = new List<string> { "omelette" },
            ["bonda"] = new List<string> { "fried snack" },
            ["upma"] = new List<string> { "semolina porridge" },
            ["idli"] = new List<string> { "steamed rice cakes" },
            ["dosa"] = new List<string> { "rice crepe" },
            ["sambar"] = new List<string> { "lentil stew" },
            ["rasam"] = new List<string> { "spicy soup" },
            ["vada"] = new List<string> { "fried savory doughnut" },
            ["pav bhaji"] = new List<string> { "mashed vegetable curry with bread" },
            ["chole bhature"] = new List<string> { "chickpeas with fried bread" },
            ["rajma"] = new List<string> { "kidney beans curry" },
            ["paneer"] = new List<string> { "cottage cheese" },
            ["palak paneer"] = new List<string> { "spinach with cottage cheese" },
            ["matar paneer"] = new List<string> { "peas with cottage cheese" },
            ["malai kofta"] = new List<string> { "cream-based meatballs" },
            ["kheer"] = new List<string> { "rice pudding" },
            ["sheer khurma"] = new List<string> { "vermicelli dessert" },
            ["seviyan"] = new List<string> { "sweet vermicelli" },
            ["ladoo"] = new List<string> { "sweet ball" },
            ["jalebi"] = new List<string> { "syrup-coated sweet spiral" },
            ["peda"] = new List<string> { "milk sweet" },
            ["rabri"] = new List<string> { "thickened milk dessert" },
            ["chawal"] = new List<string> { "rice" },
            ["barbeque"] = new List<string> { "tikka", "malai boti", "beaf boti", "behari boti", "seekh boti", "seekh kabab", "seekh kebab" },
        };

        var inputWords = query
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        var expandedWords = new List<string>();

        foreach (var word in inputWords)
        {
            var normalized = NormalizeSingular(word);
            if (romanMap.TryGetValue(normalized, out var replacements)) 
                expandedWords.AddRange(replacements);
            else
                expandedWords.Add(normalized);

            //if (romanMap.TryGetValue(word, out var mappedList))
            //    expandedWords.AddRange(mappedList);
            //else
            //    expandedWords.Add(word);
        }

        // Remove duplicates and return
        return expandedWords.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }


    


    private static int? DetectProteinType(List<string> expandedWords)
    {
        foreach (var word in expandedWords)
        {
            if (word.Contains("mutton", StringComparison.OrdinalIgnoreCase)) return 2;
            if (word.Contains("chicken", StringComparison.OrdinalIgnoreCase)) return 1;
            if (word.Contains("beef", StringComparison.OrdinalIgnoreCase)) return 3;
            if (word.Contains("fish", StringComparison.OrdinalIgnoreCase)) return 4;
            if (word.Contains("dal", StringComparison.OrdinalIgnoreCase) || word.Contains("lentil", StringComparison.OrdinalIgnoreCase)) return 5;
        }
        return null;
    }

    private static string NormalizeSingular(string word)
    {
        if (word.EndsWith("ies")) return word[..^3] + "y";         // e.g. "curries" → "curry"
        if (word.EndsWith("es")) return word[..^2];                // e.g. "dishes" → "dish"
        if (word.EndsWith("s") && word.Length > 3) return word[..^1]; // e.g. "burgers" → "burger"
        return word;
    }

    private static async Task<Dictionary<string, List<string>>> LoadPreprocessingMapAsync()
    {
        var dict = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);

        string sql = "SELECT SourceTerm, TargetTerm FROM DtlPreprocessingDictionary WHERE IsActive = 1";

        await using var conn = new SqlConnection(ConnStr);
        await conn.OpenAsync();

        await using var cmd = new SqlCommand(sql, conn);
        await using var rdr = await cmd.ExecuteReaderAsync();

        while (await rdr.ReadAsync())
        {
            string source = rdr.GetString(0).Trim().ToLowerInvariant();
            string target = rdr.GetString(1).Trim();

            if (!dict.ContainsKey(source))
                dict[source] = new List<string>();

            dict[source].Add(target);
        }

        return dict;
    }

    private static async Task<List<string>> PreprocessQueryAsync(string query)
    {
        var normalizedTokens = new List<string>();
        var map = await LoadPreprocessingMapAsync();

        foreach (var word in query.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string token = NormalizeSingular(word.ToLowerInvariant());

            if (map.TryGetValue(token, out var replacements))
            {
                normalizedTokens.AddRange(replacements);
            }
            else
            {
                normalizedTokens.Add(token);
            }
        }

        return normalizedTokens.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }

    static int? DetectProteinType(string query)
    {
        var q = query.ToLowerInvariant();
        if (q.Contains("mutton")) return 2;
        if (q.Contains("chicken")) return 1;
        if (q.Contains("beef")) return 3;
        if (q.Contains("fish")) return 4;
        if (q.Contains("dal") || q.Contains("daal")) return 5;
        return null;
    }
    private static async Task<List<(long Id, string Name, float[] Vec)>> LoadFilteredRecipesAsyncWithputProtienType(List<string> expandedWords)
    {
        if (expandedWords == null || expandedWords.Count == 0)
            return new();

        // Build WHERE clause with LIKE for each expanded keyword on RecipeName and Description
        var whereConditions = new List<string>();
        for (int i = 0; i < expandedWords.Count; i++)
        {
            whereConditions.Add($"RecipeName LIKE @kw{i}");
            whereConditions.Add($"Description LIKE @kw{i}");
        }
        string whereClause = string.Join(" OR ", whereConditions);

        string sql = $@"
            SELECT RecipeId, RecipeName, EmbeddingBGE
            FROM   MstrRecipes
            WHERE  ({whereClause}) AND EmbeddingBGE IS NOT NULL";

        var list = new List<(long, string, float[])>();
        await using var conn = new SqlConnection(ConnStr);
        await conn.OpenAsync();

        await using var cmd = new SqlCommand(sql, conn);
        for (int i = 0; i < expandedWords.Count; i++)
        {
            // Add same param twice for RecipeName and Description LIKE search
            cmd.Parameters.AddWithValue($"@kw{i}", $"%{expandedWords[i]}%");
        }

        await using var rdr = await cmd.ExecuteReaderAsync();
        while (await rdr.ReadAsync())
        {
            long id = rdr.GetInt64(0);
            string name = rdr.GetString(1);
            byte[] bin = (byte[])rdr[2];
            list.Add((id, name, BytesToFloats(bin)));
        }

        return list;
    }

    private static string NormalizeSingularByEnglishPorter2Stemmer(string word)
    {
        var stemmer = new EnglishPorter2Stemmer();
        var stemmedToken = stemmer.Stem(word);
        return stemmedToken.Value; // Convert token to string
    }


    private static bool IsEnglish(string word)
    {
        return word.All(c => char.IsLetter(c) && c < 128);  // ASCII letters only
    }



}
