import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      // Ensure JSON responses have the correct Content-Type
      const headers = new Headers(fixCors().headers);
      headers.set('Content-Type', 'application/json');
      return new Response(JSON.stringify({ error: err.message }), {
        headers,
        status: err.status ?? 500
      });
    };
    try {
      const auth = request.headers.get("Authorization");
      // Allow API Key from Authorization header or query param
      let apiKey = auth?.split(" ")[1];
      const url = new URL(request.url);
      if (!apiKey) {
          apiKey = url.searchParams.get('key');
      }


      const assert = (success, message = "Invalid request") => {
        if (!success) {
          throw new HttpError(message, 400);
        }
      };

      const { pathname } = url;
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST", "Method Not Allowed");
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST", "Method Not Allowed");
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET", "Method Not Allowed");
          return handleModels(apiKey)
            .catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText } = {}) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Allow-Methods", "*"); // Added for completeness
  headers.set("Access-Control-Allow-Headers", "*"); // Added for completeness
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

// https://github.com/google-gemini/generative-ai-js/blob/cf223ff4a1ee5a2d944c53cddb8976136382bee6/src/requests/request.ts#L71
const API_CLIENT = "genai-js/0.21.0"; // npm view @google/generative-ai version
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

async function handleModels (apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let body;
  let fixedResponse = fixCors(response);

  if (response.ok) {
    try {
      const { models } = await response.json();
      body = JSON.stringify({
        object: "list",
        data: models
          .filter(model => model.name.startsWith("models/")) // Filter out non-model entries if any
          .map(({ name, displayName, version, description, inputTokenLimit, outputTokenLimit, supportedGenerationMethods, temperature, topP, topK }) => ({
            id: name.replace("models/", ""), // Use model name as ID
            object: "model",
            created: 0, // Placeholder
            owned_by: "google", // Or infer from name
            permission: [], // Placeholder
            root: name.replace("models/", ""),
            parent: null,
          })),
      }, null, "  ");
      fixedResponse.headers.set('Content-Type', 'application/json'); // Ensure correct content type
    } catch (e) {
        console.error("Failed to parse models response:", e);
        // Fallback to original body or error message
        body = await response.text();
        if (!fixedResponse.headers.has('Content-Type')) {
             fixedResponse.headers.set('Content-Type', 'text/plain;charset=UTF-8');
        }
    }
  } else {
      body = await response.text();
      if (!fixedResponse.headers.has('Content-Type')) {
           fixedResponse.headers.set('Content-Type', 'text/plain;charset=UTF-8');
      }
  }
  return new Response(body, fixedResponse);
}


const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings (req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  // OpenAI embedding input can be a string or array of strings
  if (!Array.isArray(req.input)) {
    req.input = [ req.input ];
  }

  let model = req.model;
  // Ensure model name has "models/" prefix for Gemini API
  if (!model.startsWith("models/")) {
      // Check if it's a known default or throw error
      if (model === DEFAULT_EMBEDDINGS_MODEL) {
          model = "models/" + model;
      } else {
           // You might want to check available models via handleModels first,
           // or just assume it's a valid Gemini embedding model name
           // For now, let's assume valid if starts with 'embedding'
           if (model.startsWith('embedding')) {
              model = "models/" + model;
           } else {
             throw new HttpError(`Unsupported embedding model: ${req.model}`, 400);
           }
      }
  }

  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => {
          if (typeof text !== 'string') {
             console.warn("Embedding input is not a string:", text);
             // Handle non-string input if necessary, or throw error
             // For now, just log and use empty string or skip
             text = ''; // Or throw new HttpError("Embedding input must be string", 400);
          }
          return {
            model, // Model needs to be in each request object for batch
            content: { parts: [{ text }] }, // Text must be in a parts array
            outputDimensionality: req.dimensions, // Optional, maps to req.dimensions
        };
      })
    })
  });

  let body;
  let fixedResponse = fixCors(response);

  if (response.ok) {
    try {
      const { embeddings } = await response.json(); // Gemini returns { embeddings: [...] }
      body = JSON.stringify({
        object: "list",
        data: embeddings.map(({ values }, index) => ({
          object: "embedding",
          index,
          embedding: values, // Gemini's 'values' are the embedding array
        })),
        model: req.model, // Use original requested model name
        usage: { // Gemini API might not provide usage per embedding request, add placeholders
            prompt_tokens: 0, // Can't easily get from Gemini batch embed response
            total_tokens: 0, // Can't easily get from Gemini batch embed response
        }
      }, null, "  ");
       fixedResponse.headers.set('Content-Type', 'application/json');
    } catch (e) {
       console.error("Failed to parse embeddings response:", e);
        body = await response.text();
         if (!fixedResponse.headers.has('Content-Type')) {
             fixedResponse.headers.set('Content-Type', 'text/plain;charset=UTF-8');
        }
    }
  } else {
       body = await response.text();
        if (!fixedResponse.headers.has('Content-Type')) {
           fixedResponse.headers.set('Content-Type', 'text/plain;charset=UTF-8');
        }
  }

  return new Response(body, fixedResponse);
}


const DEFAULT_MODEL = "gemini-1.5-pro-latest"; // Or gemini-1.0-pro, gemini-2.0-flash etc.
// Use a model name that is likely to support tool calling
const DEFAULT_TOOL_CALLING_MODEL = "gemini-1.5-pro-latest"; // Or gemini-2.0-flash

async function handleCompletions (req, apiKey) {
  let model = DEFAULT_MODEL;
  // Determine the Gemini model name from the OpenAI model name
  switch(true) {
    case typeof req.model !== "string":
      // Use default
      break;
    case req.model.startsWith("models/"):
      model = req.model.substring(7); // Remove "models/" prefix if present
      break;
    case req.model.startsWith("gemini-"):
    case req.model.startsWith("learnlm-"):
      model = req.model; // Use the provided Gemini-like name
      break;
    default:
       // Handle other potential OpenAI model names if necessary, e.g., "gpt-3.5-turbo" -> "gemini-1.0-pro"
       // For simplicity, if it's not a known Gemini name or starts with "models/",
       // we can use a default or throw an error. Let's just use the default for now.
       console.warn(`Unknown model name "${req.model}", using default "${DEFAULT_MODEL}"`);
       model = DEFAULT_MODEL;
       break;
  }

  // If tools are provided, ensure a model capable of tool calling is used
  if (req.tools && req.tools.length > 0) {
      // This is a simple check; a real proxy might need to verify model capability
      // against the /models list.
      if (!model.includes("gemini-1.5") && !model.includes("gemini-2.0-flash")) {
         console.warn(`Model "${model}" might not fully support tool calling. Using "${DEFAULT_TOOL_CALLING_MODEL}" instead.`);
         model = DEFAULT_TOOL_CALLING_MODEL;
      }
  }


  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) { url += "?alt=sse"; } // Add alt=sse for Server-Sent Events stream format

  // Transform the request body to Gemini format, including tools and tool_choice
  const geminiRequestBody = await transformRequest(req);

  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(geminiRequestBody),
  });

  let body = response.body;
  let fixedResponse = fixCors(response);

  if (response.ok) {
    let id = generateChatcmplId(); // Generate a unique ID for the completion

    if (req.stream) {
        // Streaming response processing
        // Gemini SSE sends chunks like data: {...}\n\n or event: {...}\n\n (sometimes)
        // We need to parse each chunk, transform it, and re-encode as OpenAI SSE
       body = response.body
           .pipeThrough(new TextDecoderStream())
           .pipeThrough(new TransformStream({
               transform: parseStream, // Parses SSE 'data: ' lines
               flush: parseStreamFlush,
               buffer: "", // State for parseStream
           }))
           .pipeThrough(new TransformStream({
               transform: toOpenAiStream, // Transforms Gemini chunk JSON to OpenAI chunk JSON
               flush: toOpenAiStreamFlush,
               streamIncludeUsage: req.stream_options?.include_usage,
               model, id,
               lastCandidates: [], // State for toOpenAiStream: store last candidate state
           }))
           .pipeThrough(new TextEncoderStream()); // Encode back to bytes
         fixedResponse.headers.set('Content-Type', 'text/event-stream'); // Correct Content-Type for SSE

    } else {
        // Non-streaming response processing
        try {
          const geminiResponseBody = await response.json();
          body = processCompletionsResponse(geminiResponseBody, model, id); // Transform Gemini JSON to OpenAI JSON
          fixedResponse.headers.set('Content-Type', 'application/json');
        } catch (e) {
           console.error("Failed to parse non-streaming response:", e);
           body = await response.text(); // Fallback to raw text
            if (!fixedResponse.headers.has('Content-Type')) {
                 fixedResponse.headers.set('Content-Type', 'text/plain;charset=UTF-8');
            }
        }
    }
  } else {
      // If response is not ok, just return the raw response body
       body = await response.text();
       if (!fixedResponse.headers.has('Content-Type')) {
           fixedResponse.headers.set('Content-Type', 'text/plain;charset=UTF-8');
       }
  }
  // Ensure response body is a ReadableStream for streaming, or a string/buffer for non-streaming
  // fetch(..., {body: ReadableStream}) is supported in Workers
  return new Response(body, fixedResponse);
}

// --- Request Transformation ---

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  // "HARM_CATEGORY_CIVIC_INTEGRITY", // Not always supported, may cause errors
];
// Default safety settings - can be overridden by request if needed
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE", // Or BLOCK_LOW_AND_ABOVE etc.
}));

const fieldsMap = {
  // OpenAI Request Field -> Gemini Request Field
  stop: "stopSequences",
  n: "candidateCount", // Note: Gemini streaming only returns 1 candidate (n=1 effectively)
  max_tokens: "maxOutputTokens",
  // Gemini doesn't have direct equivalents for frequency/presence penalty like OpenAI
  // frequency_penalty: "frequencyPenalty", // Not a standard Gemini parameter
  // presence_penalty: "presencePenalty", // Not a standard Gemini parameter
  temperature: "temperature",
  top_p: "topP",
  // top_k: "topK", // Non-standard but sometimes supported
};

const transformConfig = (req) => {
  let cfg = {};
  // Apply standard generation config fields
  for (const key in fieldsMap) {
    if (req[key] !== undefined) { // Use !== undefined to include null/0/false if they become valid
      cfg[fieldsMap[key]] = req[key];
    }
  }

  // Handle response_format (JSON mode) - Maps to Gemini responseMimeType and responseSchema
  if (req.response_format) {
    switch(req.response_format.type) {
      case "json_schema":
        // Gemini supports responseSchema (Google's OpenAPI schema format)
        // OpenAI's json_schema might need conversion if not already in the right format
        // Assuming req.response_format.json_schema?.schema is the correct Google format schema
        if (req.response_format.json_schema?.schema) {
           cfg.responseSchema = req.response_format.json_schema.schema;
           // Determine mime type based on schema hint if available, or default to application/json
           // Example: if schema is for an enum-like string, use text/x.enum
           if (cfg.responseSchema && "enum" in cfg.responseSchema) {
             cfg.responseMimeType = "text/x.enum";
           } else {
             cfg.responseMimeType = "application/json"; // Default for schema
           }
        } else {
             // Schema type requires a schema definition
             throw new HttpError("response_format.type 'json_schema' requires 'json_schema.schema'", 400);
        }
        break;
      case "json_object":
        // Simple JSON object mode - Map to application/json mime type
        cfg.responseMimeType = "application/json";
        // Optional: Add a system instruction to ensure JSON output
        // This is a common technique for JSON mode proxies if the API doesn't strictly enforce it
        // by mime type alone. However, rely on mime type first.
        break;
      case "text":
        // Default text mode - Map to text/plain
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError(`Unsupported response_format.type: "${req.response_format.type}"`, 400);
    }
  }
  // Note: There is no direct mapping for seed, logit_bias, user, function_call (deprecated), tools (handled separately), tool_choice (handled separately) in generationConfig

  return cfg;
};


// Helper to handle image/audio parsing for multimodal input
const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        // It's better to return a structured error response here
        // or include a placeholder/error text part, rather than throwing
        console.error(`Error fetching image from URL: ${response.status} ${response.statusText} (${url})`);
        // Option 1: Throw HttpError (breaks batch processing)
        // throw new HttpError(`Failed to fetch image from URL: ${url}`, response.status);
        // Option 2: Return a text part indicating error
        return { text: `[ERROR: Failed to load image from ${url}. Status: ${response.status}]` };
        // Option 3: Return null or undefined and handle in caller (less robust)
      }
      mimeType = response.headers.get("content-type");
       if (!mimeType || !mimeType.startsWith('image/') && !mimeType.startsWith('audio/')) {
          console.error(`Fetched URL does not have a valid image/audio content type: ${mimeType} (${url})`);
           return { text: `[ERROR: URL ${url} did not return image/audio content. Content-Type: ${mimeType || 'none'}]` };
       }
      const arrayBuffer = await response.arrayBuffer();
      data = Buffer.from(arrayBuffer).toString("base64");
      // Check for audio type specifically
      if (mimeType.startsWith('audio/')) {
          return {
               inlineData: {
                   mimeType: mimeType, // e.g., audio/wav, audio/mpeg
                   data: data,
               }
          };
      } else { // Assume image
          return {
            inlineData: {
              mimeType: mimeType, // e.g., image/png, image/jpeg
              data: data,
            },
          };
      }

    } catch (err) {
      console.error('Error processing image URL:', err);
       return { text: `[ERROR: Processing image URL ${url} failed: ${err.message}]` };
      // throw new Error("Error fetching image: " + err.toString()); // Option 1
    }
  } else {
    // Handle base64 data URL
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      console.error("Invalid data URL format:", url);
       return { text: "[ERROR: Invalid data URL format provided for image/audio]" };
      // throw new Error("Invalid image data: " + url); // Option 1
    }
    const { mimeType, data: base64Data } = match.groups;
     if (!mimeType || (!mimeType.startsWith('image/') && !mimeType.startsWith('audio/'))) {
         console.error(`Data URL does not have a valid image/audio content type: ${mimeType}`);
         return { text: `[ERROR: Data URL did not specify valid image/audio content type: ${mimeType || 'none'}]` };
     }
    // No need to decode base64 if the regex captured it correctly
    if (mimeType.startsWith('audio/')) {
         return {
            inlineData: {
                mimeType: mimeType,
                data: base64Data,
            }
         };
    } else { // Assume image
        return {
          inlineData: {
            mimeType: mimeType,
            data: base64Data,
          },
        };
    }
  }
};

// Transform OpenAI message object to Gemini content object
const transformMsg = async (msg) => {
  const parts = [];
  const role = msg.role === "assistant" ? "model" : msg.role === "user" ? "user" : msg.role; // Map roles

  if (typeof msg.content === "string") {
    parts.push({ text: msg.content });
  } else if (Array.isArray(msg.content)) {
    // Multimodal content (text, image_url, input_audio)
    for (const item of msg.content) {
      switch (item.type) {
        case "text":
          if (typeof item.text === 'string') {
            parts.push({ text: item.text });
          } else {
            console.warn("Skipping non-string text part:", item.text);
          }
          break;
        case "image_url":
          if (item.image_url?.url) {
            const imgPart = await parseImg(item.image_url.url);
             if (imgPart) parts.push(imgPart);
          } else {
             console.warn("Skipping image_url part with no URL:", item);
          }
          break;
         case "input_audio": // OpenAI spec name
           if (item.input_audio?.url || item.input_audio?.data) {
             // Gemini expects inlineData for audio
             if (item.input_audio.url) {
                // Need to fetch and convert URL to base64
                 const audioPart = await parseImg(item.input_audio.url); // Reuse parseImg for URL fetching
                 if (audioPart) parts.push(audioPart);
             } else if (item.input_audio.data && item.input_audio.format) {
                // Base64 data provided directly
                 parts.push({
                   inlineData: {
                     mimeType: `audio/${item.input_audio.format}`, // Assuming format like 'wav', 'mp3' etc.
                     data: item.input_audio.data, // Assuming base64 data
                   }
                 });
             } else {
                 console.warn("Skipping input_audio part with no URL/data or format:", item);
             }
           } else {
              console.warn("Skipping input_audio part with no data:", item);
           }
          break;
        default:
          console.warn(`Skipping unknown "content" item type: "${item.type}"`);
          // throw new TypeError(`Unknown "content" item type: "${item.type}"`); // Option to throw
      }
    }
     // If only images are provided, add an empty text part to avoid API error
     // Check if there's at least one part that is NOT inlineData (i.e., text or function response)
     const hasNonImageData = parts.some(p => !p.inlineData);
     if (parts.length > 0 && !hasNonImageData) {
        console.log("Adding empty text part for image-only message.");
        parts.push({ text: "" });
     }

  } else if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
     // Handling tool_calls in assistant message history
     // OpenAI: { role: "assistant", tool_calls: [...] }
     // Gemini: { role: "model", parts: [{ functionCall: {...} }] }
     // This proxy only supports function type tool calls currently
     for (const toolCall of msg.tool_calls) {
         if (toolCall.type === "function" && toolCall.function?.name && toolCall.function?.arguments !== undefined) {
             try {
                 // OpenAI arguments is a stringified JSON object
                 const args = JSON.parse(toolCall.function.arguments);
                 parts.push({
                     functionCall: {
                         name: toolCall.function.name,
                         args: args, // Gemini expects args as a JSON object
                     }
                 });
             } catch (e) {
                 console.error("Failed to parse tool_call arguments JSON:", toolCall.function.arguments, e);
                 // Add an error text part or skip? Skipping for now.
             }
         } else {
             console.warn("Skipping unsupported tool_call type or missing info:", toolCall);
         }
     }
      // If an assistant message has tool_calls AND content, combine them
      // Gemini allows text + functionCall in parts
      if (typeof msg.content === "string" && msg.content) {
          parts.push({ text: msg.content });
      }

  } else if (msg.role === "tool" && msg.tool_call_id && msg.content !== undefined) {
      // Handling tool message (result of tool call)
      // OpenAI: { role: "tool", tool_call_id: "...", content: "..." }
      // Gemini: { role: "function", parts: [{ functionResponse: { name: "...", response: {...} } }] }
      // Need to know the function name corresponding to tool_call_id.
      // This requires storing state or inferring from context, which is complex for a stateless proxy.
      // A simpler approach for the proxy is to assume the *immediately preceding* assistant message's
      // tool_call_id matches this tool message's tool_call_id, and that there's only one tool call.
      // Or, require the original function name to be passed in the tool message content somehow (not standard OpenAI).
      // Or, transform the tool message content into a text part and rely on the LLM to understand it's a result.
      // The most compatible way assuming a simple flow is to convert it to a 'function' role message
      // with a 'functionResponse' part, but the 'name' is missing.
      // Let's add a placeholder/guess name if possible, or rely on the LLM understanding a generic result.

      // *** Simplification / Potential Issue ***
      // Mapping tool results accurately requires knowing the *name* of the function that was called,
      // which isn't directly available in the OpenAI `role: "tool"` message itself.
      // The Gemini `functionResponse` *requires* the `name`.
      // For a simple proxy, we might have to skip these messages, or add a dummy name,
      // or convert them to user text, e.g., "Tool result for call ID ...: [content]".
      // A better proxy would need to track tool_call_ids and function names from assistant messages.

      // For now, let's skip `role: "tool"` messages in this basic transformation
      // as accurately mapping them requires state. A more robust proxy is needed
      // to fully support multi-turn conversations with tool results.
      console.warn(`Skipping role: "tool" message as full transformation requires state.`);
      // If you MUST include something, maybe convert to a user message:
      // parts.push({ text: `[Tool result for call ID ${msg.tool_call_id}]:\n${msg.content}` });
      // role = "user"; // Change role to user for the converted message
      // --- END Simplification ---


  } else if (msg.function_call) {
       // Handling deprecated function_call in assistant message history
       // OpenAI: { role: "assistant", function_call: {...} }
       // Gemini: { role: "model", parts: [{ functionCall: {...} }] }
       // Convert deprecated function_call to new tool_calls format internally if needed, then map
       console.warn("Deprecated function_call used. Converting to tool_calls format.");
       const toolCall = {
           id: '', // Deprecated format didn't have ID
           type: 'function',
           function: {
               name: msg.function_call.name,
               arguments: msg.function_call.arguments, // Already stringified JSON
           }
       };
       // Now process like a tool_calls message
        try {
            const args = JSON.parse(toolCall.function.arguments);
             parts.push({
                 functionCall: {
                     name: toolCall.function.name,
                     args: args, // Gemini expects args as a JSON object
                 }
             });
        } catch (e) {
            console.error("Failed to parse deprecated function_call arguments JSON:", toolCall.function.arguments, e);
            // Add an error text part or skip? Skipping for now.
        }
        // If an assistant message has deprecated function_call AND content, combine them
        if (typeof msg.content === "string" && msg.content) {
            parts.push({ text: msg.content });
        }
  } else if (msg.content === null && !msg.tool_calls && !msg.function_call) {
      // Sometimes messages might have null content and no tool/function calls.
      // This could be valid (e.g., an image-only message before adding the empty text part).
      // If parts are still empty, add an empty text part to prevent API errors.
       if (parts.length === 0) {
          console.log(`Adding empty text part for message with null content and no calls (role: ${msg.role}).`);
           parts.push({ text: "" });
       }
  }


  // Filter out any null/undefined parts resulting from errors or skipped items
  const filteredParts = parts.filter(p => p != null && (p.text !== undefined || p.inlineData || p.functionCall || p.functionResponse));

   if (filteredParts.length === 0 && role !== "system") {
       // If after processing, there are still no parts (e.g., unsupported message type),
       // add a placeholder or skip the message entirely? Skipping seems safer.
       console.warn(`Message for role "${role}" resulted in no valid parts after transformation. Skipping message.`);
       return null; // Indicate message should be skipped
   }


  return { role, parts: filteredParts };
};


const transformMessages = async (messages) => {
  if (!messages) { return { contents: [] }; } // Return empty contents if no messages

  const contents = [];
  let system_instruction = null;
  let foundInitialUserMessage = false; // Gemini requires user message first (or system instruction)

  for (const item of messages) {
      if (item.role === "system") {
          // Handle system instruction - only the first one is typically used by Gemini
          if (!system_instruction) {
              // Transform system message content
              // Note: Gemini system instruction does NOT have a role field
              const transformedSystemContent = await transformMsg({ content: item.content });
              if (transformedSystemContent && transformedSystemContent.parts && transformedSystemContent.parts.length > 0) {
                  system_instruction = { parts: transformedSystemContent.parts };
              } else {
                 console.warn("System message resulted in no valid parts. Skipping system instruction.");
              }
          } else {
              console.warn("Multiple system messages provided. Using only the first one.");
          }
      } else {
          // Handle user/assistant/tool messages - map to Gemini roles and parts
          const transformedMessage = await transformMsg(item);
          if (transformedMessage) { // Only add if transformation resulted in valid parts
              contents.push(transformedMessage);
              if (transformedMessage.role === "user") {
                  foundInitialUserMessage = true;
              }
          }
      }
  }

   // Gemini requires alternating user/model roles and typically starts with user (or system instruction)
   // If the first message is 'model' (assistant), add a dummy user message to satisfy API
    if (contents.length > 0 && contents[0].role === "model" && !system_instruction) {
        console.warn("First message is 'model'. Prepending dummy 'user' message.");
        contents.unshift({ role: "user", parts: [{ text: " " }] }); // Add a dummy user message
    }

   // Ensure alternating roles: user, model, user, model...
   // This basic proxy doesn't enforce strict alternation, but it's good practice.
   // Gemini API might return errors if roles are not alternating correctly.
   // E.g., consecutive 'user' or 'model' roles without a 'functionResponse' part in between.
   // Handling this robustly requires more complex history management.

  return { system_instruction, contents };
};


const transformRequest = async (req) => {
    const { system_instruction, contents } = await transformMessages(req.messages);

    const requestBody = {
        contents: contents,
        generationConfig: transformConfig(req),
        safetySettings: req.safety_settings || safetySettings, // Use request safety settings if provided, else default
        // Gemini tools definition - directly map from OpenAI tools
        tools: req.tools && Array.isArray(req.tools) && req.tools.length > 0
               ? req.tools.map(tool => {
                   // Assuming only 'function' type tools are supported
                   if (tool.type === 'function' && tool.function?.name && tool.function?.parameters) {
                       // OpenAI function parameters use JSON Schema format, which Gemini also uses
                       return { functionDeclaration: {
                           name: tool.function.name,
                           description: tool.function.description, // Optional
                           parameters: tool.function.parameters, // JSON Schema object
                       }};
                   } else {
                       console.warn("Skipping unsupported tool type or missing function details:", tool);
                       return null; // Skip invalid tool definitions
                   }
               }).filter(t => t != null) // Remove skipped tools
               : undefined, // Pass undefined if no valid tools
    };

    // Map OpenAI tool_choice to Gemini tool_config function_calling_config
    if (req.tool_choice !== undefined && requestBody.tools && requestBody.tools.length > 0) {
        let mode;
        let allowedFunctionNames = [];

        if (typeof req.tool_choice === 'string') {
            switch (req.tool_choice) {
                case 'none':
                    mode = 'NONE';
                    break;
                case 'auto': // Equivalent to 'any' for Gemini tool calling
                case 'any': // Sometimes 'any' is also used
                    mode = 'ANY'; // Gemini 'ANY' tries to call *any* provided function if appropriate
                    break;
                case 'required': // Also maps to 'any' in a simple proxy context, means it *must* call *a* tool
                     mode = 'ANY';
                     // Gemini doesn't have a strict "required" mode like OpenAI that *guarantees* a call.
                     // 'ANY' with tools provided makes it very likely if the query fits.
                     // A more sophisticated proxy might retry with a system instruction if 'ANY' doesn't trigger a call.
                    break;
                default:
                    console.warn(`Unsupported tool_choice string "${req.tool_choice}". Using 'AUTO' (ANY).`);
                    mode = 'ANY';
            }
        } else if (typeof req.tool_choice === 'object' && req.tool_choice.type === 'function' && req.tool_choice.function?.name) {
            // Specific function call requested (OpenAI object format)
            mode = 'ANY'; // Must use ANY mode to specify allowed names
            allowedFunctionNames.push(req.tool_choice.function.name);
            // Note: Gemini API `allowedFunctionNames` requires `mode: ANY`.
        } else {
            console.warn("Unsupported tool_choice object format:", req.tool_choice);
             // Default to 'AUTO' (ANY) if tool_choice is invalid but tools are present
            mode = 'ANY';
        }

        requestBody.tool_config = {
            function_calling_config: {
                mode: mode,
                ...(allowedFunctionNames.length > 0 && { allowedFunctionNames: allowedFunctionNames }),
            }
        };
        // Note: If mode is NONE, allowedFunctionNames is ignored by Gemini.
    }
    // If req.tool_choice is undefined, the tool_config field is omitted, which defaults to AUTO in Gemini.

    if (system_instruction) {
        requestBody.system_instruction = system_instruction;
    }

  //console.log("Transformed Gemini Request:", JSON.stringify(requestBody, null, 2));
  return requestBody;
};

// --- Response Transformation ---

const generateChatcmplId = () => {
  // Simple random ID generation
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return "chatcmpl-" + Array.from({ length: 29 }, randomChar).join("");
};

// Mapping Gemini finishReason to OpenAI finish_reason
const reasonsMap = { //https://ai.google.dev/api/rest/v1/GenerateContentResponse#finishreason
  "STOP": "stop", // Natural stop point
  "MAX_TOKENS": "length", // Exceeded max_tokens
  "SAFETY": "content_filter", // Blocked by safety settings
  "RECITATION": "content_filter", // Blocked by recitation
  "OTHER": "other", // Catch-all for other reasons
  // Gemini's function calls also result in finishReason: "STOP"
  // We need to detect tool calls separately
};

// Helper to transform a single Gemini Candidate into an OpenAI choice message/delta
const transformCandidates = (key, cand, isStreaming = false) => {
  // key will be "message" for non-streaming, "delta" for streaming
  const openaiOutput = {
      index: cand.index || 0, // Gemini usually provides index
      logprobs: null, // Gemini API does not provide logprobs in this format
  };

  const content = { role: "assistant" }; // Base structure for message/delta content

  // --- Handle Function Calls ---
  if (cand.functionCalls && cand.functionCalls.length > 0) {
      content.tool_calls = cand.functionCalls.map(geminiFuncCall => {
          // Transform Gemini functionCall to OpenAI tool_call
          return {
              id: '', // Gemini doesn't provide an ID, use empty string or generate one if needed
              type: 'function',
              function: {
                  name: geminiFuncCall.name,
                  // Gemini args is a JSON object, OpenAI arguments is a stringified JSON object
                  arguments: JSON.stringify(geminiFuncCall.args),
              },
          };
      });
      content.content = null; // OpenAI requires content=null when tool_calls is present

      // For streaming, the finish_reason indicating tool_calls comes *after* the delta with tool_calls
      // For non-streaming, the finish_reason on the choice is "tool_calls"
      if (!isStreaming) {
           openaiOutput.finish_reason = "tool_calls";
      } else {
           // Streaming delta might contain tool_calls, but finish_reason comes later
           // Leave finish_reason null for intermediate streaming chunks
           openaiOutput.finish_reason = null; // Will be updated in the flush
      }

  } else {
       // --- Handle Text/Other Content ---
       // Concatenate text parts. If no text parts, content should be null.
       const textParts = cand.content?.parts
           ?.filter(p => p.text !== undefined) // Filter for parts that have text
           .map(p => p.text)
           .join(''); // Join all text parts
       content.content = textParts || null; // Use null if no text

       // If it's the first streaming chunk and has content, OpenAI delta content is ""
       // This is handled by `toOpenAiStream`'s `first` flag logic, not here.

       content.tool_calls = null; // No tool calls if no functionCalls from Gemini

       // Map Gemini finishReason to OpenAI finish_reason for non-tool-call stops
       // For streaming, finish_reason is added in the flush function
       if (!isStreaming) {
           openaiOutput.finish_reason = reasonsMap[cand.finishReason] || cand.finishReason;
       } else {
            openaiOutput.finish_reason = null; // Will be updated in the flush
       }
  }

  // Add the transformed content (message or delta) to the output object
  openaiOutput[key] = content;

  return openaiOutput;
};

// Non-streaming response processing
const processCompletionsResponse = (data, model, id) => {
  // Check for errors in Gemini response structure
   if (data.candidates === undefined) {
       console.error("Gemini response missing 'candidates' field:", data);
       // Check for Gemini specific error structure
       if (data.error?.message) {
            throw new HttpError(`Gemini API error: ${data.error.message}`, data.error.code || 500);
       }
        // Fallback to a generic error if structure is unexpected
       throw new HttpError("Unexpected Gemini API response structure", 500);
   }
    if (data.candidates.length === 0 && !data.promptFeedback?.blockReason) {
       // No candidates, but not blocked? This might mean a safety block without a specific reason field filled.
       // Or an input issue. Gemini sometimes returns an empty candidates array on certain safety blocks.
        console.warn("Gemini response has 0 candidates:", data);
         // If prompt was blocked, map the reason
         if (data.promptFeedback?.blockReason) {
             const blockReason = data.promptFeedback.blockReason; // e.g., "SAFETY"
             const blockedCategories = data.promptFeedback.safetyRatings
                ?.filter(r => r.blocked)
                .map(r => r.category)
                .join(', ') || 'unknown categories';

             throw new HttpError(`Prompt blocked by Gemini safety settings. Reason: ${blockReason}. Categories: ${blockedCategories}.`, 400); // Use 400 for input block
         }
        // If no candidates and no block reason, return a generic error choice
        return JSON.stringify({
             id,
             choices: [{
                 index: 0,
                 message: { role: "assistant", content: "Error: Gemini API returned no candidates." },
                 finish_reason: "error", // Custom finish reason for internal error
                 logprobs: null,
             }],
             created: Math.floor(Date.now()/1000),
             model,
             object: "chat.completion",
             usage: transformUsage(data.usageMetadata), // Usage might still be present
        }, null, "  ");
    }


  // Transform each candidate into an OpenAI choice
  const choices = data.candidates.map(cand => transformCandidates("message", cand, false));

  // If there was a safety block *after* generating some content (unlikely for Gemini single turn, but possible)
  // or if the prompt was blocked (handled above), add a relevant finish reason if not already set.
  // The `transformCandidates` should handle the 'SAFETY' finish reason.

  return JSON.stringify({
    id,
    choices: choices,
    created: Math.floor(Date.now()/1000),
    model,
    //system_fingerprint: "fp_69829325d0", // Optional
    object: "chat.completion",
    usage: transformUsage(data.usageMetadata), // usageMetadata contains token counts
  }, null, "  ");
};

// Transform Gemini usageMetadata to OpenAI usage object
const transformUsage = (data) => {
    if (!data) return undefined; // Return undefined if no usage data

    // Gemini usageMetadata structure: { promptTokenCount, candidatesTokenCount, totalTokenCount }
    return {
        completion_tokens: data.candidatesTokenCount || 0,
        prompt_tokens: data.promptTokenCount || 0,
        total_tokens: data.totalTokenCount || 0,
        // Add any other relevant fields if available and mappable
    };
};

// --- Streaming Response Processing ---

// Parser for Server-Sent Events (SSE) format
const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
async function parseStream (chunk, controller) {
  // chunk is expected to be a string
  if (typeof chunk !== 'string') {
      console.error("Received non-string chunk in stream parser:", chunk);
      // Attempt to decode if it's bytes
      try {
          chunk = new TextDecoder().decode(chunk);
      } catch (e) {
          console.error("Failed to decode non-string chunk:", e);
          this.buffer = ""; // Clear buffer to avoid infinite loops
          return; // Skip this chunk
      }
  }

  this.buffer += chunk;
  do {
    const match = this.buffer.match(responseLineRE);
    if (!match) { break; }
    // Enqueue the JSON string part after "data: "
    controller.enqueue(match[1]);
    // Remove the matched part from the buffer
    this.buffer = this.buffer.substring(match[0].length);
  } while (this.buffer.length > 0 && this.buffer.match(responseLineRE)); // Keep processing if buffer has more complete messages
}

async function parseStreamFlush (controller) {
  if (this.buffer) {
    console.error("Stream parser buffer not empty on flush:", this.buffer);
    // Optional: Enqueue remaining buffer as an error chunk?
    // controller.enqueue(JSON.stringify({ error: "Incomplete stream data" }));
  }
}

const delimiter = "\n\n"; // OpenAI SSE delimiter

// Transformer from Gemini stream chunk JSON to OpenAI stream chunk JSON
async function toOpenAiStream (chunkJsonString, controller) {
  const transform = transformResponseStream.bind(this); // Bind 'this' context (state)

  // chunkJsonString is expected to be the JSON string from parseStream
  if (!chunkJsonString) {
      console.warn("Received empty JSON string from parseStream.");
      return; // Skip empty chunks
  }

  let data;
  try {
    data = JSON.parse(chunkJsonString);
    //console.log("Processing Gemini stream chunk:", JSON.stringify(data));
  } catch (err) {
    console.error("Failed to parse JSON from stream chunk:", chunkJsonString, err);
    // Enqueue an error chunk
    controller.enqueue(`data: ${JSON.stringify({
        id: this.id,
        choices: [{
            index: 0, // Assume index 0 for error chunk
            delta: { role: "assistant", content: `[Error parsing stream chunk: ${err.message}]` },
            finish_reason: "error", // Custom error reason
        }],
        created: Math.floor(Date.now()/1000),
        model: this.model,
        object: "chat.completion.chunk",
    })}${delimiter}`);
    return; // Skip this problematic chunk
  }

  // A Gemini stream chunk might have multiple candidates, but usually it's one per chunk
  if (!data.candidates || data.candidates.length === 0) {
       // This might be a chunk with only usageMetadata or promptFeedback
       // Check for promptFeedback (safety block)
       if (data.promptFeedback?.blockReason) {
           const blockReason = data.promptFeedback.blockReason;
            const blockedCategories = data.promptFeedback.safetyRatings
                ?.filter(r => r.blocked)
                .map(r => r.category)
                .join(', ') || 'unknown categories';
           console.warn(`Stream blocked by Gemini safety settings. Reason: ${blockReason}. Categories: ${blockedCategories}.`, data);
            // Enqueue a stop chunk with content_filter reason for index 0
            controller.enqueue(`data: ${JSON.stringify({
                id: this.id,
                choices: [{
                    index: 0, // Assume index 0 is the one being blocked
                    delta: {}, // Empty delta
                    finish_reason: "content_filter",
                }],
                created: Math.floor(Date.now()/1000),
                model: this.model,
                object: "chat.completion.chunk",
                 // Include usage on the final block chunk if requested
                ...(this.streamIncludeUsage && data.usageMetadata && { usage: transformUsage(data.usageMetadata) }),
            })}${delimiter}`);
            // It might be appropriate to end the stream here as well, although Gemini might send more block info.
            // Adding [DONE] here would require closing the controller.
           return; // Skip this chunk after handling block
       }

       // If no candidates and no block reason, it might be an empty chunk, skip.
       // Or a chunk with only usage (should be handled later in flush if needed).
       // console.warn("Received stream chunk with no candidates:", data);
       return; // Skip chunks with no candidates or block info
   }


  // We expect candidates, usually just one in a streaming chunk
  data.candidates.forEach(cand => {
      // Find or initialize state for this candidate index
      const index = cand.index || 0;
      if (!this.lastCandidates[index]) {
          this.lastCandidates[index] = { delta: {}, finish_reason: null };
      }
      const last = this.lastCandidates[index];

      // --- Process Delta Content or Tool Calls ---
      const transformedCand = transformCandidates("delta", cand, true); // isStreaming = true
      const delta = transformedCand.delta; // This delta might have 'content' or 'tool_calls'


      // Append content delta if present
      if (delta.content !== undefined && delta.content !== null) { // Check for undefined and null explicitly
          if (!last.delta.content) {
              last.delta.content = delta.content; // First content chunk for this index
          } else {
              last.delta.content += delta.content; // Append content chunks
          }
           // On the very first content chunk across *all* candidates, OpenAI spec sometimes suggests delta.content: ""
           // This logic was in the original code (if first) but might interfere with tool_calls.
           // Let's rely on the delta transformation providing the content.
           // If the first actual text chunk has content, the delta will have it.
      } else if (delta.tool_calls !== undefined && delta.tool_calls !== null) {
           // Tool calls delta - These usually appear fully formed in one chunk per tool call
           // OpenAI groups multiple tool calls under one 'tool_calls' array in the delta/message
           if (!last.delta.tool_calls) {
              last.delta.tool_calls = [];
           }
            // Append new tool_calls from this chunk to the candidate's state
            last.delta.tool_calls.push(...delta.tool_calls);
      }

      // Update finish reason if present in the chunk
      // Gemini streaming chunks *usually* only have the finishReason on the *last* chunk.
      // If a chunk *does* contain a finishReason, store it.
      if (cand.finishReason) {
           last.finish_reason = reasonsMap[cand.finishReason] || cand.finishReason;
      }

       // --- Enqueue OpenAI compatible chunk ---
       // The chunk to enqueue should contain the *current* delta for this candidate
       const openaiChunk = {
            id: this.id,
            choices: [{
                index: index,
                delta: delta, // Send the delta processed from THIS chunk
                finish_reason: null, // finish_reason is typically sent on the final DONE chunk
            }],
            created: Math.floor(Date.now()/1000),
            model: this.model,
            object: "chat.completion.chunk",
            // Usage is only included in the final chunk if requested
            // ...(this.streamIncludeUsage && ??? ), // Usage comes in usageMetadata
       };

       // OpenAI spec sometimes requires 'role: assistant' in the first delta for a choice
       // The transformCandidates function already adds role: "assistant" to the delta object structure.
       // Ensure it's there for the very first delta chunk of a candidate.
        if (!this.sentFirstDelta[index]) {
            if (!openaiChunk.choices[0].delta.role) {
                 openaiChunk.choices[0].delta.role = "assistant";
            }
             this.sentFirstDelta[index] = true;
        }


       // Enqueue the chunk
       controller.enqueue(`data: ${JSON.stringify(openaiChunk)}${delimiter}`);

        // Store the latest state of the candidate for the flush function
        // The lastCandidates state accumulates content and final finish_reason
        if (cand.content?.parts) {
            // Accumulate content for flush, only if content is present in this chunk
            // This is needed to get the total usage calculation right potentially,
            // or if you needed to reconstruct the full message in the flush (less common for simple proxy)
             // Note: Simple content accumulation might be complex with tool calls interleaved.
             // Let's primarily use the finish_reason for flush logic.
        }
        if (cand.finishReason) {
             // If a finish reason is in this chunk, mark this candidate as finished
             last.finish_reason = reasonsMap[cand.finishReason] || cand.finishReason;
             last.finished = true; // Custom flag for flushing
             // Accumulate usage metadata if present on this final chunk
             if (data.usageMetadata && this.streamIncludeUsage) {
                 last.usageMetadata = data.usageMetadata;
             }
        }
   });

   // Check if all active candidates are finished. If so, prepare for DONE.
   // This requires more complex state tracking than currently implemented.
   // Let's rely on the flush function being called when the upstream stream ends.

}

// State for toOpenAiStream:
// buffer: string accumulated from parseStream
// id: chat completion ID
// model: model name
// streamIncludeUsage: boolean from req.stream_options?.include_usage
// lastCandidates: Array<{ delta: { content?: string, tool_calls?: Array }, finish_reason?: string, finished?: boolean, usageMetadata?: object }>
// sentFirstDelta: Map<number, boolean> to track if role: assistant was sent for each index

async function toOpenAiStreamFlush (controller) {
   // When the upstream stream ends, send finish_reason for any candidates that finished
   // and then send the [DONE] signal.

   // If there were any candidates at all
   if (this.lastCandidates.length > 0) {
       this.lastCandidates.forEach((last, index) => {
           // If the candidate finished (indicated by finish_reason being set)
           if (last.finish_reason) {
                // Send a final chunk for this candidate with the finish_reason
                const openaiChunk = {
                   id: this.id,
                   choices: [{
                       index: index,
                       delta: {}, // Empty delta for the final stop chunk
                       finish_reason: last.finish_reason,
                   }],
                   created: Math.floor(Date.now()/1000),
                   model: this.model,
                   object: "chat.completion.chunk",
                    // Include usage metadata if requested and available for this candidate
                    ...(this.streamIncludeUsage && last.usageMetadata && { usage: transformUsage(last.usageMetadata) }),
               };
                controller.enqueue(`data: ${JSON.stringify(openaiChunk)}${delimiter}`);
           } else {
               // Candidate ended without a finish reason? Could be an error or unexpected end.
                console.warn(`Stream ended for candidate index ${index} without a finish_reason.`);
                // Optionally send an error finish reason
                 const openaiChunk = {
                   id: this.id,
                   choices: [{
                       index: index,
                       delta: {}, // Empty delta
                       finish_reason: "error", // Custom error reason
                   }],
                   created: Math.floor(Date.now()/1000),
                   model: this.model,
                   object: "chat.completion.chunk",
                   // Usage might not be available for interrupted streams
               };
                controller.enqueue(`data: ${JSON.stringify(openaiChunk)}${delimiter}`);
           }
       });
   } else {
       // If no candidates were processed at all, perhaps the stream was empty or errored early.
       // If streamIncludeUsage was requested, and there was usageMetadata on a non-candidate chunk,
       // you might need to store and send it here. This requires more complex state.
       // For simplicity, usage might be missed in edge cases if not attached to a candidate's final chunk.
       console.warn("Stream ended with no candidates processed.");
       // Optionally send an empty choice with an error finish reason for index 0
         const openaiChunk = {
            id: this.id,
            choices: [{
                index: 0,
                delta: {},
                finish_reason: "error",
            }],
            created: Math.floor(Date.now()/1000),
            model: this.model,
            object: "chat.completion.chunk",
         };
         controller.enqueue(`data: ${JSON.stringify(openaiChunk)}${delimiter}`);

   }


  // Send the final [DONE] signal
  controller.enqueue(`data: [DONE]${delimiter}`);
}

// Initial state setup for toOpenAiStream
// Note: This requires the TransformStream constructor to support initial state or for `this` to be an object
// passed into the transform/flush methods. Cloudflare Workers' TransformStream supports this.
// Initialize sentFirstDelta state
Object.defineProperty(toOpenAiStream.prototype, 'sentFirstDelta', {
    value: new Map(),
    writable: true,
    configurable: true
});
// Initialize lastCandidates state
Object.defineProperty(toOpenAiStream.prototype, 'lastCandidates', {
    value: [],
    writable: true,
    configurable: true
});
// Initialize buffer state for parseStream
Object.defineProperty(parseStream.prototype, 'buffer', {
    value: "",
    writable: true,
    configurable: true
});
// parseStreamFlush also needs the buffer state
Object.defineProperty(parseStreamFlush.prototype, 'buffer', {
    value: "", // Should reference the same buffer as parseStream
    writable: true,
    configurable: true
});
