const path = require("path");
const { spawn } = require("child_process");
const readline = require("readline");
const http = require("http");
const https = require("https");
const config = require("./config");
const { getDb } = require("./db");
let embedWorker;
let embedRl;
let workerReady = false;
let reqSeq = 0;
const pending = new Map();

function startEmbedWorker() {
  if (embedWorker && !embedWorker.killed) return;

  const scriptPath = path.resolve(__dirname, config.embeddingScriptPath);
  embedWorker = spawn(config.pythonCommand, [scriptPath], {
    cwd: path.resolve(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
  });

  embedRl = readline.createInterface({ input: embedWorker.stdout });
  workerReady = false;

  embedRl.on("line", (line) => {
    try {
      const msg = JSON.parse(line);
      if (msg.type === "ready") {
        workerReady = true;
        console.log(
          `[RAG] Embedding worker ready | model=${msg.model} | device=${msg.device}`
        );
        return;
      }
      const req = pending.get(msg.id);
      if (!req) return;
      clearTimeout(req.timer);
      pending.delete(msg.id);
      if (msg.error) req.reject(new Error(`Embedding worker error: ${msg.error}`));
      else req.resolve(msg.embedding);
    } catch (err) {
      console.error("[RAG] Failed parsing embed worker output:", err.message);
    }
  });

  embedWorker.stderr.on("data", (chunk) => {
    console.error("[RAG][embed-worker][stderr]", chunk.toString().trim());
  });

  embedWorker.on("close", (code) => {
    console.error(`[RAG] Embedding worker exited (code=${code})`);
    workerReady = false;
    for (const [, req] of pending) {
      clearTimeout(req.timer);
      req.reject(new Error("Embedding worker exited before responding"));
    }
    pending.clear();
  });
}

function buildPrompt(question, contexts) {
  const contextBlock = contexts
    .map((c, i) => {
      return `[${i + 1}] ${c.title || "Untitled"}\nURL: ${c.url || "N/A"}\n${c.text}`;
    })
    .join("\n\n");

  return [
    "You are a civic governance assistant for Karnataka policy documents.",
    "Use only the provided context. If context is insufficient, say so clearly.",
    "Provide a concise answer and include citation markers like [1], [2].",
    "",
    `Question: ${question}`,
    "",
    "Context:",
    contextBlock,
  ].join("\n");
}

function embedText(text) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    startEmbedWorker();
    const id = ++reqSeq;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(
        new Error(
          `Embedding timed out after ${config.embedTimeoutMs}ms. Check Python env/model load.`
        )
      );
    }, config.embedTimeoutMs);

    pending.set(id, {
      timer,
      resolve: (vec) => {
        console.log(
          `[RAG] Query embedding ready in ${Date.now() - start}ms (dim=${vec.length})`
        );
        resolve(vec);
      },
      reject,
    });

    const send = () => {
      const payload = JSON.stringify({ id, text });
      embedWorker.stdin.write(payload + "\n");
    };

    if (workerReady) send();
    else {
      const waitStart = Date.now();
      const interval = setInterval(() => {
        if (workerReady) {
          clearInterval(interval);
          send();
          return;
        }
        if (Date.now() - waitStart > config.embedTimeoutMs) {
          clearInterval(interval);
          clearTimeout(timer);
          pending.delete(id);
          reject(
            new Error(
              `Embedding worker startup timed out after ${config.embedTimeoutMs}ms`
            )
          );
        }
      }, 200);
    }
  });
}

async function retrieveRelevantChunks(question, topK = 5) {
  const start = Date.now();
  console.log("[RAG] Generating query embedding...");
  const embedding = await embedText(question);
  const db = await getDb();
  const collection = db.collection(config.mongoCollectionName);

  const pipeline = [
    {
      $vectorSearch: {
        index: config.mongoVectorIndexName,
        path: "embedding",
        queryVector: embedding,
        numCandidates: Math.max(topK * 20, 100),
        limit: topK,
      },
    },
    {
      $project: {
        _id: 0,
        chunk_id: 1,
        doc_id: 1,
        title: 1,
        url: 1,
        text: 1,
        score: { $meta: "vectorSearchScore" },
      },
    },
  ];

  const chunks = await collection.aggregate(pipeline).toArray();
  console.log(
    `[RAG] Vector search returned ${chunks.length} chunks in ${Date.now() - start}ms`
  );
  return chunks;
}

async function answerWithRag(question, topK = 5) {
  console.log("[RAG] Starting answerWithRag...");
  const chunks = await retrieveRelevantChunks(question, topK);
  const prompt = buildPrompt(question, chunks);
  console.log(
    `[RAG] Sending prompt to Ollama model=${config.ollamaChatModel} (chunks=${chunks.length})`
  );

  const completion = await callOllamaGenerate({
    baseUrl: config.ollamaBaseUrl,
    timeoutMs: config.ollamaTimeoutMs,
    payload: {
      model: config.ollamaChatModel,
      prompt,
      stream: false,
      options: {
        temperature: 0.2,
        num_predict: config.ollamaNumPredict,
      },
    },
  });

  const answer = String(completion.response || "").trim();
  console.log(`[RAG] Ollama response received (answerLen=${answer.length})`);
  const citations = chunks.map((chunk, idx) => ({
    id: idx + 1,
    chunk_id: chunk.chunk_id,
    doc_id: chunk.doc_id,
    title: chunk.title || "Untitled",
    url: chunk.url || "",
    score: Number(chunk.score || 0),
  }));

  return { answer, citations };
}

function callOllamaGenerate({ baseUrl, payload, timeoutMs }) {
  return new Promise((resolve, reject) => {
    const endpoint = new URL("/api/generate", baseUrl);
    const client = endpoint.protocol === "https:" ? https : http;
    const body = JSON.stringify(payload);

    const req = client.request(
      endpoint,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
      },
      (res) => {
        let raw = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          raw += chunk;
        });
        res.on("end", () => {
          if ((res.statusCode || 500) >= 400) {
            reject(
              new Error(`Ollama generation failed: ${res.statusCode} ${raw}`)
            );
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch (err) {
            reject(new Error(`Invalid Ollama JSON response: ${err.message}`));
          }
        });
      }
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Ollama request timed out after ${timeoutMs}ms`));
    });
    req.on("error", (err) => reject(err));
    req.write(body);
    req.end();
  });
}

module.exports = { answerWithRag };
