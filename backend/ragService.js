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
const NO_DATA_TEXT = "data not available, pls try asking again";
const EMBED_WORKER_STDERR_MAX = 8;
let embedWorkerLastStderr = [];

function captureEmbedWorkerStderr(line) {
  if (!line) return;
  embedWorkerLastStderr.push(line);
  if (embedWorkerLastStderr.length > EMBED_WORKER_STDERR_MAX) {
    embedWorkerLastStderr = embedWorkerLastStderr.slice(-EMBED_WORKER_STDERR_MAX);
  }
}

function formatEmbedWorkerDebug() {
  const stderrSummary = embedWorkerLastStderr.length
    ? embedWorkerLastStderr.join(" | ")
    : "no-stderr-captured";
  return `pythonCommand=${config.pythonCommand} script=${config.embeddingScriptPath} timeoutMs=${config.embedTimeoutMs} lastStderr=${stderrSummary}`;
}

function startEmbedWorker() {
  if (embedWorker && !embedWorker.killed) return;

  const scriptPath = path.resolve(__dirname, config.embeddingScriptPath);
  console.log(
    `[RAG] Starting embed worker | python=${config.pythonCommand} | script=${scriptPath} | timeoutMs=${config.embedTimeoutMs}`
  );
  embedWorkerLastStderr = [];
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
    const line = chunk.toString().trim();
    captureEmbedWorkerStderr(line);
    console.error("[RAG][embed-worker][stderr]", line);
  });

  embedWorker.on("error", (err) => {
    console.error("[RAG] Embedding worker failed to start:", err.message);
    workerReady = false;
    for (const [, req] of pending) {
      clearTimeout(req.timer);
      req.reject(
        new Error(
          `Embedding worker process error: ${err.message}. ${formatEmbedWorkerDebug()}`
        )
      );
    }
    pending.clear();
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
              `Embedding worker startup timed out after ${config.embedTimeoutMs}ms. ${formatEmbedWorkerDebug()}`
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
    `[RAG] Sending prompt to ${getEffectiveProvider().toUpperCase()} (chunks=${chunks.length})`
  );

  const answer = await generateTextFromPrompt(prompt, {
    temperature: 0.2,
    maxTokens: config.ollamaNumPredict,
  });
  console.log(`[RAG] Model response received (answerLen=${answer.length})`);
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

function parseGovEmpIntent(question) {
  const q = question.toLowerCase();
  const pendingWords = ["pending", "not solved", "open", "in progress","unresolved", "assigned", "investigating", "reopened"];
  const resolvedWords = ["resolved", "closed", "completed", "solved"];

  const departmentMatch = question.match(
    /department\s+(?:named|name|called)?\s*[:\-]?\s*([a-z0-9&().,\- ]+)/i
  );
  const areaMatch = question.match(
    /(area|ward|zone|panchayat|city|village|locality)\s+(?:named|name|called)?\s*[:\-]?\s*([a-z0-9&().,\- ]+)/i
  );
  const contractorMatch = question.match(
    /(contractor|vendor)\s+(?:named|name|called)?\s*[:\-]?\s*([a-z0-9&().,\- ]+)/i
  );

  const wantsDeptWise =
    q.includes("department wise") || q.includes("by department");
  const wantsAreaWise = q.includes("area wise") || q.includes("by area");

  let status = "any";
  if (pendingWords.some((w) => q.includes(w))) status = "pending";
  if (resolvedWords.some((w) => q.includes(w))) status = "resolved";

  return {
    status,
    departmentName: (departmentMatch?.[1] || "").trim(),
    areaName: (areaMatch?.[2] || "").trim(),
    contractorName: (contractorMatch?.[2] || "").trim(),
    wantsDeptWise,
    wantsAreaWise,
  };
}

function statusRegexForIntent(status) {
  if (status === "pending") {
    return /registered|open|in[-_\s]?progress|pending|assigned|investigating|reopened/i;
  }
  if (status === "resolved") {
    return /resolved|closed|completed|done|solved|fixed/i;
  }
  return null;
}

function textContains(value, search) {
  if (!value || !search) return false;
  return String(value).toLowerCase().includes(String(search).toLowerCase());
}

async function answerGovEmployeeFromDb(question) {
  const db = await getDb();
  const complaintsCol = db.collection(config.complaintCollectionName);
  const departmentsCol = db.collection(config.departmentCollectionName);
  const areasCol = db.collection(config.areaCollectionName);

  const complaintCount = await complaintsCol.countDocuments({});
  if (complaintCount === 0) {
    return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
  }

  const intent = parseGovEmpIntent(question);
  const statusRegex = statusRegexForIntent(intent.status);
  const complaintMatch = {};
  if (statusRegex) complaintMatch.status = { $regex: statusRegex };

  if (intent.wantsDeptWise) {
    const rows = await complaintsCol
      .aggregate([
        { $match: complaintMatch },
        {
          $lookup: {
            from: config.departmentCollectionName,
            localField: "departmentId",
            foreignField: "_id",
            as: "department",
          },
        },
        {
          $addFields: {
            departmentName: {
              $ifNull: [{ $arrayElemAt: ["$department.name", 0] }, "Unassigned"],
            },
          },
        },
        { $group: { _id: "$departmentName", count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 10 },
      ])
      .toArray();

    if (!rows.length) return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
    const lines = rows.map((r) => `${r._id}: ${r.count}`).join(", ");
    const factual = `Complaint counts by department: ${lines}.`;
    const answer = await generateGovEmpAnswer(question, factual);
    return { answer, citations: [], mode: "govemp_db" };
  }

  if (intent.wantsAreaWise) {
    const rows = await complaintsCol
      .aggregate([
        { $match: complaintMatch },
        {
          $lookup: {
            from: config.areaCollectionName,
            localField: "areaId",
            foreignField: "_id",
            as: "area",
          },
        },
        {
          $addFields: {
            areaName: { $ifNull: [{ $arrayElemAt: ["$area.name", 0] }, "$location"] },
          },
        },
        { $group: { _id: "$areaName", count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 10 },
      ])
      .toArray();

    if (!rows.length) return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
    const lines = rows.map((r) => `${r._id || "Unknown"}: ${r.count}`).join(", ");
    const factual = `Complaint counts by area: ${lines}.`;
    const answer = await generateGovEmpAnswer(question, factual);
    return { answer, citations: [], mode: "govemp_db" };
  }

  let departmentDoc = null;
  if (intent.departmentName) {
    const deptCandidates = await departmentsCol.find({}, { projection: { name: 1 } }).toArray();
    departmentDoc =
      deptCandidates.find((d) => textContains(d.name, intent.departmentName)) || null;
    if (!departmentDoc) {
      return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
    }
    complaintMatch.departmentId = departmentDoc._id;
  }

  let areaDoc = null;
  if (intent.areaName) {
    const areaCandidates = await areasCol.find({}, { projection: { name: 1 } }).toArray();
    areaDoc = areaCandidates.find((a) => textContains(a.name, intent.areaName)) || null;
    if (areaDoc) {
      complaintMatch.areaId = areaDoc._id;
    } else {
      complaintMatch.location = { $regex: intent.areaName, $options: "i" };
    }
  }

  if (intent.contractorName) {
    complaintMatch.$or = [
      { contractorName: { $regex: intent.contractorName, $options: "i" } },
      { contractor: { $regex: intent.contractorName, $options: "i" } },
      { assignedContractor: { $regex: intent.contractorName, $options: "i" } },
    ];
  }

  const count = await complaintsCol.countDocuments(complaintMatch);
  if (!count) return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };

  const filters = [];
  if (intent.status !== "any") filters.push(intent.status);
  if (departmentDoc?.name) filters.push(`department ${departmentDoc.name}`);
  if (areaDoc?.name) filters.push(`area ${areaDoc.name}`);
  else if (intent.areaName) filters.push(`area ${intent.areaName}`);
  if (intent.contractorName) filters.push(`contractor ${intent.contractorName}`);

  const suffix = filters.length ? ` for ${filters.join(", ")}` : "";
  const factual = `Total complaints${suffix}: ${count}.`;
  const answer = await generateGovEmpAnswer(question, factual);
  return { answer, citations: [], mode: "govemp_db" };
}

async function generateGovEmpAnswer(question, factualLine) {
  try {
    const prompt = [
      "You are a government complaint assistant.",
      "Respond in 1-2 clear sentences for an employee dashboard.",
      "Use only the factual data provided below.",
      "Do not invent numbers or entities.",
      "",
      `Question: ${question}`,
      `Factual data: ${factualLine}`,
    ].join("\n");

    const text = await generateTextFromPrompt(prompt, {
      temperature: 0.1,
      maxTokens: Math.min(config.ollamaNumPredict, 96),
    });
    return text || factualLine;
  } catch (err) {
    console.error(
      "[RAG] GovEmp generation failed, using factual fallback:",
      err.message
    );
    return factualLine;
  }
}

function getEffectiveProvider() {
  if (config.prodEnv) return "groq";
  const configured = String(config.llmProvider || "").toLowerCase();
  return configured === "groq" ? "groq" : "ollama";
}

async function generateTextFromPrompt(prompt, { temperature, maxTokens }) {
  const provider = getEffectiveProvider();

  if (provider === "groq") {
    if (!config.groqApiKey) {
      throw new Error("GROQ_API_KEY is required when prod_env=true");
    }

    const completion = await callGroqChatCompletions({
      baseUrl: config.groqApiBaseUrl,
      timeoutMs: config.groqTimeoutMs,
      apiKey: config.groqApiKey,
      payload: {
        model: config.groqChatModel,
        messages: [{ role: "user", content: prompt }],
        temperature,
        max_tokens: Math.min(maxTokens, config.groqMaxTokens),
      },
    });

    const text = String(completion?.choices?.[0]?.message?.content || "").trim();
    if (!text) throw new Error("Groq returned an empty response");
    return text;
  }

  const completion = await callOllamaGenerate({
    baseUrl: config.ollamaBaseUrl,
    timeoutMs: config.ollamaTimeoutMs,
    payload: {
      model: config.ollamaChatModel,
      prompt,
      stream: false,
      options: {
        temperature,
        num_predict: maxTokens,
      },
    },
  });

  return String(completion.response || "").trim();
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

function callGroqChatCompletions({ baseUrl, payload, timeoutMs, apiKey }) {
  return new Promise((resolve, reject) => {
    const endpoint = new URL(baseUrl.replace(/\/+$/, "") + "/chat/completions");
    const client = endpoint.protocol === "https:" ? https : http;
    const body = JSON.stringify(payload);

    const req = client.request(
      endpoint,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
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
            reject(new Error(`Groq generation failed: ${res.statusCode} ${raw}`));
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch (err) {
            reject(new Error(`Invalid Groq JSON response: ${err.message}`));
          }
        });
      }
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Groq request timed out after ${timeoutMs}ms`));
    });
    req.on("error", (err) => reject(err));
    req.write(body);
    req.end();
  });
}

module.exports = { answerWithRag, answerGovEmployeeFromDb };
