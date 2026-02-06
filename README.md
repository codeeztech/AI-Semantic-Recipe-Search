# AI Semantic Recipe Search

A **multi-model semantic search application** for recipes built in **.NET Console**.  
Leverages **OpenAI embeddings**, **MiniLM ONNX model**, and **BGE ONNX model** to retrieve relevant recipes based on user search queries.

This project demonstrates **semantic search**, **vector similarity**, and **multi-model ensemble techniques** for practical AI applications in the food domain.

---

## **Features**

- **OpenAI Embeddings:** Semantic search using OpenAI’s `text-embedding-3-large` model.
- **MiniLM (ONNX):** Local lightweight embedding search using `AllMiniLmL6V2Sharp`.
- **BGE (ONNX):** Local high-quality embedding search with BGE model for better context understanding.
- **Ensemble Search:** Combines results from all three models and removes duplicates.
- **Configurable Thresholds:** Fine-tune minimum similarity scores for each model.
- **Console-based:** Simple, easy-to-run .NET console application.
- **Safe OpenAI Integration:** Handles null embeddings and API errors gracefully.

---

## **Project Structure**

| File Name | Purpose |
|-----------|---------|
| `Program_OpenAI.cs` | Semantic search using **OpenAI embeddings** only. |
| `Program_MiniLM.cs` | Semantic search using **MiniLM ONNX model**. |
| `Program_BGE.cs` | Semantic search using **BGE ONNX model**. |
| `Program_Ensemble.cs` | Combined search from all three models, with **deduplicated results**. |
| `Program_Ensemble_Final.cs` | Final tuned version with **safe OpenAI calls** and **optimized thresholds**. |

---

## **Setup Instructions**

1. Clone the repository:
```bash
git clone https://github.com/your-username/AI-Semantic-Recipe-Search.git
cd AI-Semantic-Recipe-Search
