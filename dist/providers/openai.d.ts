import OpenAI from "openai";
import type { ChatMessage } from "../types";
export declare class OpenAIProvider {
    private client;
    constructor();
    chat(params: {
        model?: string;
        messages: ChatMessage[];
        stream?: boolean;
        temperature?: number;
        max_tokens?: number | null;
        top_p?: number;
    }): Promise<(OpenAI.Chat.Completions.ChatCompletion & {
        _request_id?: string | null;
    }) | {
        text: string;
    }>;
}
//# sourceMappingURL=openai.d.ts.map