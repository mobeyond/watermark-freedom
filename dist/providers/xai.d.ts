import { ChatMessage } from "../types";
export declare class XAIProvider {
    private baseURL;
    private apiKey;
    constructor(options?: {
        baseURL?: string;
    });
    chat(params: {
        model?: string;
        messages: ChatMessage[];
        stream?: boolean;
        temperature?: number;
        max_tokens?: number;
        top_p?: number;
    }): Promise<NodeJS.ReadableStream | {
        text: any;
    }>;
}
//# sourceMappingURL=xai.d.ts.map