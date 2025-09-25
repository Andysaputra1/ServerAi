import express from "express";
import cors from "cors";
import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

type Chunk = { source: string; text: string; embedding: number[] };
let STORE: Chunk[] = [];

// ---- util
function chunkText(text: string, size = 800, overlap = 120) {
  const words = text.split(/\s+/);
  const out: string[] = [];
  for (let i = 0; i < words.length; i += (size - overlap)) {
    out.push(words.slice(i, i + size).join(" "));
    if (i + size >= words.length) break;
  }
  return out.filter(Boolean);
}
function cosine(a: number[], b: number[]) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
async function embedOne(text: string) {
  const e = await openai.embeddings.create({ model: "text-embedding-3-large", input: text });
  return e.data[0].embedding as number[];
}

// ---- load & build vector store on startup
async function buildStore() {
  const p = path.join(process.cwd(), "data", "profile.json");
  const raw = fs.readFileSync(p, "utf8");
  const data = JSON.parse(raw);

  const entries: { source: string; text: string }[] = [];
  if (data.profile?.summary) entries.push({ source: "profile:summary", text: data.profile.summary });
  for (const prj of (data.projects || [])) {
    entries.push({
      source: `project:${prj.name}`,
      text: `${prj.name}\n${prj.description}\nImpact: ${prj.impact}\nStack: ${(prj.tech_stack||[]).join(", ")}\n${prj.repo_url||""}`
    });
  }
  for (const exp of (data.experiences || [])) {
    entries.push({ source: `exp:${exp.company}`, text: `${exp.role} @ ${exp.company}\n${(exp.achievements||[]).join("; ")}` });
  }
  for (const f of (data.faqs || [])) {
    entries.push({ source: `faq:${f.q}`, text: `${f.q}\n${f.a}` });
  }

  const chunks: { source: string; text: string }[] = [];
  for (const e of entries) {
    for (const c of chunkText(e.text)) chunks.push({ source: e.source, text: c });
  }

  const batched: Chunk[] = [];
  // embed in small batches to be gentle
  for (const c of chunks) {
    const emb = await embedOne(c.text);
    batched.push({ source: c.source, text: c.text, embedding: emb });
  }
  STORE = batched;
  console.log(`Vector store built with ${STORE.length} chunks.`);
}

const SYSTEM_PROMPT = `
You are the candidate’s public career chatbot. Audience: HR and general public.
Tone: professional, concise, friendly. Mirror ID/EN automatically.
Answer ONLY from the provided context. If missing, say you don't have that info.
Refuse sensitive PII (NIK/NPWP/SSN, full home address, family, religion, marital status).
Do NOT state salary unless present in context.
Always include short [source: ...] tags based on the context you used.
`;

// ---- API
app.post("/api/chat", async (req, res) => {
  try {
    const question: string = req.body?.question ?? "";
    if (!question) return res.status(400).json({ error: "question required" });

    // embed query
    const qEmb = await embedOne(question);

    // top-K from in-memory store
    const ranked = STORE
      .map((c) => ({ ...c, score: cosine(qEmb, c.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);

    const context = ranked
      .map(r => `SOURCE: ${r.source}\n${r.text}`)
      .join("\n\n---\n\n");

    const cmp = await openai.chat.completions.create({
      model: "gpt-4.1-mini",
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: `Context:\n${context}\n\nQuestion:\n${question}\n\nAnswer with [source: ...] tags.` }
      ]
    });

    res.json({ answer: cmp.choices?.[0]?.message?.content ?? "No answer." });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "server error" });
  }
});

// build store then start
buildStore().then(() => {
  app.get("/", (_, res) => res.send("OK")); 
  app.listen(process.env.PORT || 8787, () => {
    console.log("✅ API running on", process.env.PORT || 8787);
  });
}).catch(err => {
  console.error("Failed to build vector store:", err);
  process.exit(1);
});
