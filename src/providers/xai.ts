import axios from "axios";
import { ChatMessage } from "../types";
import { config } from "../config";
import type { AxiosRequestConfig } from "axios";

export class XAIProvider {
  private baseURL: string;
  private apiKey: string;

  constructor(options?: { baseURL?: string }) {
    if (!config.xaiApiKey) {
      throw new Error("XAI_API_KEY is required to use the xAI provider");
    }
    this.apiKey = config.xaiApiKey;
    // xAI aims to be OpenAI-compatible; adjust if your account docs show a different base URL
    this.baseURL = options?.baseURL || "https://api.x.ai/v1";
  }

  async chat(params: {
    model?: string;
    messages: ChatMessage[];
    stream?: boolean;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
  }) {
    const {
      model = "grok-2-latest",
      messages,
      stream = false,
      temperature,
      max_tokens,
      top_p,
    } = params;

    const payload = {
      model,
      messages,
      stream,
      temperature,
      max_tokens,
      top_p,
    };

    const headers = {
      Authorization: `Bearer ${this.apiKey}`,
      "Content-Type": "application/json",
      Accept: stream ? "text/event-stream" : "application/json",
    } as Record<string, string>;

    if (stream) {
      const response = await axios.post(`${this.baseURL}/chat/completions`, payload, {
        headers,
        responseType: "stream",
      } as AxiosRequestConfig);
      return response.data as NodeJS.ReadableStream;
    }

    const response = await axios.post(`${this.baseURL}/chat/completions`, payload, { headers });
    const text = response.data?.choices?.[0]?.message?.content ?? "";
    return { text };
  }
}