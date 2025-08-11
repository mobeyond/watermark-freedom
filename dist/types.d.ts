export type Role = "system" | "user" | "assistant";
export interface ChatMessage {
    role: Role;
    content: string;
}
export type ProviderId = "openai" | "xai" | "browser";
export type BrowserSite = "chatgpt" | "grok";
export interface ChatRequestPayload {
    provider: ProviderId;
    model?: string;
    site?: BrowserSite;
    messages: ChatMessage[];
    stream?: boolean;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
}
export interface ProviderResponseChunk {
    type: "text" | "tool" | "error" | "done";
    content?: string;
    error?: string;
}
//# sourceMappingURL=types.d.ts.map