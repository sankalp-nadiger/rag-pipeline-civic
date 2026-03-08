const path = require("path");
const dotenv = require("dotenv");

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });

module.exports = {
  port: Number(process.env.PORT || 8080),
  mongoUri: process.env.MONGO_URI || "",
  mongoDbName: process.env.MONGO_DB_NAME || "rag_policy_db",
  mongoCollectionName: process.env.MONGO_COLLECTION_NAME || "policy_chunks",
  mongoVectorIndexName:
    process.env.MONGO_VECTOR_INDEX_NAME || "policy_chunks_vector_index",
  embeddingModel: process.env.EMBED_MODEL_NAME || "BAAI/bge-m3",
  pythonCommand: process.env.PYTHON_COMMAND || "python",
  embeddingScriptPath:
    process.env.EMBED_QUERY_SCRIPT || "../processing/embed_worker.py",
  ollamaBaseUrl: process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434",
  ollamaChatModel: process.env.OLLAMA_CHAT_MODEL || "llama3:8b",
  ollamaNumPredict: Number(process.env.OLLAMA_NUM_PREDICT || 256),
  embedTimeoutMs: Number(process.env.EMBED_TIMEOUT_MS || 120000),
  ollamaTimeoutMs: Number(process.env.OLLAMA_TIMEOUT_MS || 120000),
};
