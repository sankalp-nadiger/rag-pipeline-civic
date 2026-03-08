const express = require("express");
const { answerWithRag } = require("./ragService");

const router = express.Router();

router.post("/ragChat", async (req, res) => {
  try {
    const question = String(req.body?.question || "").trim();
    const topK = Number(req.body?.topK || 5);
    console.log(
      `[RAG] Incoming /ragChat request | questionLen=${question.length} | topK=${topK}`
    );

    if (!question) {
      console.warn("[RAG] Rejected /ragChat request: empty question");
      return res.status(400).json({ error: "question is required" });
    }

    const result = await answerWithRag(question, topK);
    console.log(
      `[RAG] Completed /ragChat | citations=${result.citations?.length || 0}`
    );
    return res.json(result);
  } catch (error) {
    console.error("ragChat error:", error);
    return res.status(500).json({
      error: "RAG request failed",
      detail: error.message || "Unknown error",
    });
  }
});

module.exports = router;
