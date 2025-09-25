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

type Chunk = { source: string; text: string; embedding: number[]; score?: number };
let STORE: Chunk[] = [];

const MODEL_EMB = "text-embedding-3-large";
const MODEL_CHAT = "gpt-4.1-mini";
const TOP_K = 6;

// ---------- utils
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
  const e = await openai.embeddings.create({ model: MODEL_EMB, input: text });
  return e.data[0].embedding as number[];
}
function safeJoin(...parts: string[]) {
  return path.join(...parts).replace(/\\/g, "/");
}

// ---------- load & build store
async function buildStore() {
  // pakai __dirname supaya aman untuk dist/
  const dataPath = safeJoin(__dirname, "..", "data", "profile.json");
  const raw = fs.readFileSync(dataPath, "utf8");
  const data = JSON.parse(raw);

  // normalisasi key profile "aneh"
  if (data.profile?.["Soft_Skills:"]) {
    data.profile.soft_skills = data.profile["Soft_Skills:"];
    delete data.profile["Soft_Skills:"];
  }
  if (data.profile?.Hard_Skills) {
    data.profile.hard_skills = data.profile.Hard_Skills;
    delete data.profile.Hard_Skills;
  }
  if (data.profile?.Language) {
    data.profile.languages = data.profile.Language;
    delete data.profile.Language;
  }

  const entries: { source: string; text: string }[] = [];

  // PROFILE
  if (data.profile?.summary) {
    entries.push({ source: "profile:summary", text: String(data.profile.summary) });
  }
  if (data.profile?.full_name || data.profile?.headline) {
    entries.push({
      source: "profile:headline",
      text: `${data.profile.full_name ?? ""}\n${data.profile.headline ?? ""}`.trim()
    });
  }
  if (data.profile?.hard_skills || data.profile?.soft_skills || data.profile?.languages) {
    entries.push({
      source: "profile:skills",
      text: [
        data.profile.hard_skills ? `Hard skills: ${data.profile.hard_skills}` : "",
        data.profile.soft_skills ? `Soft skills: ${data.profile.soft_skills}` : "",
        data.profile.languages  ? `Languages: ${data.profile.languages}`   : "",
      ].filter(Boolean).join("\n")
    });
  }

  // PROJECTS
  for (const prj of (data.projects || [])) {
    entries.push({
      source: `project:${prj.name}`,
      text: [
        prj.name,
        prj.description,
        prj.impact ? `Impact: ${prj.impact}` : "",
        (prj.tech_stack && prj.tech_stack.length) ? `Stack: ${prj.tech_stack.join(", ")}` : "",
        prj.repo_url ?? ""
      ].filter(Boolean).join("\n")
    });
  }

  // EXPERIENCES
  for (const exp of (data.experiences || [])) {
    const org = exp.organization || exp.company || exp.institution || "Unknown";
    const bullets = Array.isArray(exp.highlights)
      ? exp.highlights
      : Array.isArray(exp.responsibilities)
      ? exp.responsibilities
      : [];
    const period = `${exp.start_date ?? ""}–${(exp.end_date ?? "Present")}`;
    const extra = Array.isArray(exp.projects) && exp.projects.length ? `Projects: ${exp.projects.join(", ")}` : "";

    entries.push({
      source: `exp:${org}`,
      text: [
        `${exp.role} @ ${org} (${period})`,
        bullets.join("; "),
        extra
      ].filter(Boolean).join("\n")
    });
  }

  // EDUCATION (akomodasi "Education" / "education")
  const education = data.education ?? data.Education ?? [];
  for (const ed of (education || [])) {
    const period = `${ed.start_date ?? ""}–${(ed.end_date_expected ?? ed.end_date ?? "")}`;
    const coursework = Array.isArray(ed.coursework) && ed.coursework.length ? `Coursework: ${ed.coursework.join(", ")}` : "";
    const activities = Array.isArray(ed.activities) && ed.activities.length ? `Activities: ${ed.activities.join(", ")}` : "";
    const campusOrLoc = ed.campus ? `Campus: ${ed.campus}` : (ed.location ? `Location: ${ed.location}` : "");
    const gpa = ed.gpa ? `GPA: ${ed.gpa}` : "";

    entries.push({
      source: `edu:${ed.institution}`,
      text: [
        `${ed.institution}${campusOrLoc ? `, ${campusOrLoc}` : ""}`,
        [ed.degree ?? ed.program ?? "", ed.stream ? `(${ed.stream})` : ""].join(" ").trim(),
        period,
        gpa,
        coursework,
        activities
      ].filter(Boolean).join("\n")
    });
  }

  // FAQs
  for (const f of (data.faqs || [])) {
    entries.push({ source: `faq:${f.q}`, text: `${f.q}\n${f.a}` });
  }

  // chunk + embed
  const chunks: { source: string; text: string }[] = [];
  for (const e of entries) for (const c of chunkText(e.text)) chunks.push({ source: e.source, text: c });

  const batched: Chunk[] = [];
  for (const c of chunks) {
    try {
      const emb = await embedOne(c.text);
      batched.push({ source: c.source, text: c.text, embedding: emb });
    } catch (err) {
      console.error("[embed error]", c.source, (err as any)?.message || err);
    }
  }
  STORE = batched;
  console.log(`[build] file: ${dataPath}`);
  console.log(`[build] projects: ${(data.projects||[]).length} | experiences: ${(data.experiences||[]).length} | education: ${education.length} | faqs: ${(data.faqs||[]).length}`);
  console.log(`Vector store built with ${STORE.length} chunks.`);
}

const SYSTEM_PROMPT = `
You are the candidate’s public career chatbot. Audience: HR and general public.
Tone: professional, concise, friendly. Mirror Indonesian/English automatically.
Answer ONLY from the provided context. If missing, say you don't have that info.
Refuse sensitive PII (NIK/NPWP/SSN, full home address, family, religion, marital status).
Do NOT state salary unless present in context.
Always include short [source: ...] tags derived from the context you used (e.g., [source: edu:Binus University]).
`;

// ---------- API
app.get("/api/health", (_req, res) => res.json({ ok: true, chunks: STORE.length }));

app.get("/api/debug/store", (_req, res) => {
  const bySrc = STORE.reduce((m, c) => {
    m[c.source] = (m[c.source] || 0) + 1;
    return m;
  }, {} as Record<string, number>);
  res.json({ chunks: STORE.length, bySrc });
});

app.post("/api/chat", async (req, res) => {
  try {
    const question: string = req.body?.question ?? "";
    if (!question) return res.status(400).json({ error: "question required" });

    const qEmb = await embedOne(question);
    const ranked = STORE
      .map((c) => ({ ...c, score: cosine(qEmb, c.embedding) }))
      .sort((a, b) => (b.score! - a.score!))
      .slice(0, TOP_K);

    const context = ranked.map(r => `SOURCE: ${r.source}\n${r.text}`).join("\n\n---\n\n");

    const cmp = await openai.chat.completions.create({
      model: MODEL_CHAT,
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: `Context:\n${context}\n\nQuestion:\n${question}\n\nAnswer with [source: ...] tags.` }
      ]
    });

    res.json({ answer: cmp.choices?.[0]?.message?.content ?? "No answer." });
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: e?.message || "server error" });
  }
});

// ---------- start
buildStore().then(() => {
  app.get("/", (_req, res) => res.send("OK"));
  const PORT = Number(process.env.PORT) || 8787;
  app.listen(PORT, () => console.log("✅ API running on", PORT));
}).catch(err => {
  console.error("Failed to build vector store:", err);
  process.exit(1);
});
