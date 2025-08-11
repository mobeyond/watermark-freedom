import express from "express";
import cors from "cors";
import { config } from "./config";
import { ChatRequestPayload, ProviderId } from "./types";
import { OpenAIProvider } from "./providers/openai";
import { XAIProvider } from "./providers/xai";
import { BrowserProvider } from "./providers/browser";

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(
  cors({
    origin: (origin, callback) => callback(null, config.corsOrigin === "*" ? true : config.corsOrigin),
    credentials: true,
  })
);

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

function getProvider(providerId: ProviderId) {
  switch (providerId) {
    case "openai":
      return new OpenAIProvider();
    case "xai":
      return new XAIProvider();
    case "browser":
      return new BrowserProvider();
    default:
      throw new Error(`Unsupported provider: ${providerId}`);
  }
}

app.post("/v1/chat", async (req, res) => {
  const body = req.body as ChatRequestPayload;
  if (!body || !body.provider || !Array.isArray(body.messages)) {
    return res.status(400).json({ error: "Invalid request body" });
  }

  try {
    const provider = getProvider(body.provider);

    if (body.stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");

      const flush = (data: string) => {
        res.write(`data: ${data}\n\n`);
      };

      const end = () => res.end();

      const providerResp: any = await (provider as any).chat({
        model: body.model,
        site: body.site,
        messages: body.messages,
        stream: true,
        temperature: body.temperature,
        max_tokens: body.max_tokens,
        top_p: body.top_p,
      });

      if (providerResp && typeof providerResp[Symbol.asyncIterator] === "function") {
        for await (const delta of providerResp) {
          const content = delta?.choices?.[0]?.delta?.content;
          if (typeof content === "string" && content.length > 0) {
            flush(JSON.stringify({ type: "text", content }));
          }
        }
        flush(JSON.stringify({ type: "done" }));
        end();
        return;
      }

      if (providerResp && typeof providerResp.on === "function") {
        providerResp.on("data", (chunk: Buffer) => {
          const s = chunk.toString("utf8");
          s.split(/\r?\n/).forEach((line) => {
            if (!line.startsWith("data:")) return;
            const json = line.slice(5).trim();
            if (json === "[DONE]") {
              flush(JSON.stringify({ type: "done" }));
            } else if (json) {
              try {
                const obj = JSON.parse(json);
                const content = obj?.choices?.[0]?.delta?.content;
                if (typeof content === "string" && content.length > 0) {
                  flush(JSON.stringify({ type: "text", content }));
                }
              } catch {
              }
            }
          });
        });
        providerResp.on("end", () => {
          flush(JSON.stringify({ type: "done" }));
          end();
        });
        providerResp.on("error", (err: Error) => {
          flush(JSON.stringify({ type: "error", error: err.message }));
          end();
        });
        return;
      }

      flush(JSON.stringify({ type: "error", error: "Unknown stream response" }));
      end();
      return;
    }

    const result: any = await (provider as any).chat({
      model: body.model,
      site: body.site,
      messages: body.messages,
      temperature: body.temperature,
      max_tokens: body.max_tokens,
      top_p: body.top_p,
      stream: false,
    });

    return res.json({ content: result.text || "" });
  } catch (err: any) {
    const status = err?.response?.status || 500;
    const message = err?.response?.data || err?.message || "Internal error";
    return res.status(status).json({ error: message });
  }
});

app.listen(config.port, () => {
  // eslint-disable-next-line no-console
  console.log(`Chat proxy listening on http://localhost:${config.port}`);
});