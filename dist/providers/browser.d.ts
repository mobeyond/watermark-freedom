import { ChatMessage, BrowserSite } from "../types";
interface BrowserChatParams {
    site: BrowserSite;
    messages: ChatMessage[];
    stream?: boolean;
}
export declare class BrowserProvider {
    chat(params: BrowserChatParams): Promise<AsyncGenerator<any, void, unknown> | {
        text: string;
    }>;
    private chatWithChatGPT;
    private buildPrompt;
    private streamAssistantText;
    private waitForAssistantFinalText;
    private static readLastAssistantText;
}
export {};
//# sourceMappingURL=browser.d.ts.map