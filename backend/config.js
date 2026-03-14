const path = require("path");
const dotenv = require("dotenv");

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });

const parseBooleanEnv = (value) => {
  const normalized = String(value || "").trim().toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes";
};

const isProdEnv = parseBooleanEnv(process.env.prod_env);

module.exports = {
  port: Number(process.env.PORT || 8080),
  prodEnv: isProdEnv,
  mongoUri: process.env.MONGO_URI || "",
  mongoDbName: process.env.MONGO_DB_NAME || "rag_policy_db",
  mongoCollectionName: process.env.MONGO_COLLECTION_NAME || "policy_chunks",
  complaintCollectionName:
    process.env.MONGO_COMPLAINT_COLLECTION_NAME || "complaints",
  departmentCollectionName:
    process.env.MONGO_DEPARTMENT_COLLECTION_NAME || "departments",
  areaCollectionName: process.env.MONGO_AREA_COLLECTION_NAME || "areas",
  mongoVectorIndexName:
    process.env.MONGO_VECTOR_INDEX_NAME || "policy_chunks_vector_index",
  embeddingModel: process.env.EMBED_MODEL_NAME || "BAAI/bge-m3",
  pythonCommand: process.env.PYTHON_COMMAND || "python",
  embeddingScriptPath:
    process.env.EMBED_QUERY_SCRIPT || "../processing/embed_worker.py",
  llmProvider:
    process.env.LLM_PROVIDER ||
    (isProdEnv ? "groq" : "ollama"),
  groqApiKey: process.env.GROQ_API_KEY || "",
  groqApiBaseUrl:
    process.env.GROQ_API_BASE_URL || "https://api.groq.com/openai/v1",
  groqChatModel: process.env.GROQ_CHAT_MODEL || "llama-3.1-8b-instant",
  groqMaxTokens: Number(process.env.GROQ_MAX_TOKENS || 256),
  groqTimeoutMs: Number(process.env.GROQ_TIMEOUT_MS || 60000),
  ollamaBaseUrl: process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434",
  ollamaChatModel: process.env.OLLAMA_CHAT_MODEL || "llama3:8b",
  ollamaNumPredict: Number(process.env.OLLAMA_NUM_PREDICT || 256),
  embedTimeoutMs: Number(process.env.EMBED_TIMEOUT_MS || 120000),
  ollamaTimeoutMs: Number(process.env.OLLAMA_TIMEOUT_MS || 120000),
};
