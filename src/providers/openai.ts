import OpenAI from "openai";
import type { ChatMessage } from "../types";
import { config } from "../config";

export class OpenAIProvider {
  private client: OpenAI;

  constructor() {
    if (!config.openaiApiKey) {
      throw new Error("OPENAI_API_KEY is required to use the OpenAI provider");
    }
    this.client = new OpenAI({ apiKey: config.openaiApiKey });
  }

  async chat(params: {
    model?: string;
    messages: ChatMessage[];
    stream?: boolean;
    temperature?: number;
    max_tokens?: number | null;
    top_p?: number;
  }) {
    const {
      model = "gpt-4o-mini",
      messages,
      stream = false,
      temperature,
      max_tokens = null,
      top_p,
    } = params;

    const openaiMessages = messages.map((m) => ({ role: m.role, content: m.content }));

    const base: any = {
      model,
      messages: openaiMessages,
    };
    if (typeof temperature === "number") base.temperature = temperature;
    if (max_tokens !== null) base.max_tokens = max_tokens;
    if (typeof top_p === "number") base.top_p = top_p;

    if (stream) {
      const streamResp = await this.client.chat.completions.create({
        ...base,
        stream: true,
      } as any);
      return streamResp;
    }

    const resp = await this.client.chat.completions.create(base);
    const text = resp.choices?.[0]?.message?.content ?? "";
    return { text };
  }
}