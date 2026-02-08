# Integrating LlamaCPP Studio with OpenCode

This guide will help you integrate LlamaCPP Studio's llama.cpp server with OpenCode, an open-source AI coding agent.

## Overview

OpenCode can use local models through OpenAI-compatible endpoints. LlamaCPP Studio provides a fully configured llama.cpp server optimized for the GLM-4.7 Flash model, making it an excellent choice for running OpenCode locally.

**Benefits of this integration:**
- **Privacy First**: All code runs locally on your machine
- **Cost Effective**: No API costs, just electricity
- **Performance**: GLM-4.7 Flash is optimized for coding tasks
- **Reliability**: Local server, no network dependencies

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.9+**: Already installed in your environment
2. **GPU** (optional but recommended): NVIDIA with CUDA support or AMD with ROCm
3. **Models directory**: Configured with GLM-4.7 Flash model
4. **LlamaCPP Studio**: Already installed and server is working

The server should be running and accessible at `http://127.0.0.1:11433`

## Starting the LlamaCPP Server

Make sure your LlamaCPP Studio server is running:

```bash
# From the project root
./scripts/start_server.sh

# Verify it's working
curl http://127.0.0.1:11433/v1/models
```

You should see a JSON response with the model information, confirming the server is ready.

## OpenCode Configuration

OpenCode needs a configuration file to connect to your local server. Create or modify the `opencode.json` file in your project directory:

### Basic Configuration

Create or edit `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp-studio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LlamaCPP Studio (Local)",
      "options": {
        "baseURL": "http://127.0.0.1:11433/v1"
      },
      "models": {
        "glm-4.7-flash-ud-q4-k-xl": {
          "name": "GLM-4.7 Flash (Local)",
          "limit": {
            "context": 200000,
            "output": 65536
          }
        }
      }
    }
  }
}
```

### Configuration Explained

- **provider.llamacpp-studio**: The provider ID used in the configuration
- **npm**: The AI SDK package for OpenAI-compatible APIs
- **name**: Display name shown in OpenCode UI
- **options.baseURL**: The endpoint URL of your local llama.cpp server
- **models.glm-4.7-flash-ud-q4-k-xl**: Model configuration
  - `name`: Display name for the model
  - `limit.context`: Maximum input tokens (200K for GLM-4.7 Flash)
  - `limit.output`: Maximum output tokens (65K)

## Using OpenCode with LlamaCPP Studio

Once configured, you can use OpenCode in your coding projects.

### Initialize OpenCode

Navigate to your project directory:

```bash
cd /path/to/your/project
```

Run OpenCode:

```bash
opencode
```

### Select Your Model

In the OpenCode interface:

1. Press `/models` to see available models
2. Select **LlamaCPP Studio** as your provider
3. Choose **GLM-4.7 Flash (Local)** as your model
4. Press Enter to confirm

### Start Using OpenCode

Now you can ask OpenCode to help with your code:

```
How do I implement authentication in this project?
```

Or for more complex tasks:

```
I need to add a feature that allows users to export their data as JSON.
```

### Common OpenCode Commands

- **Tab**: Toggle between Plan and Build modes
- **Ctrl+C**: Stop current action
- **/undo**: Undo previous changes
- **/redo**: Redo changes
- **@**: Search for files in your project

### Best Practices

1. **Provide Context**: Include file references with `@filename.ext`
2. **Use Plan Mode**: Toggle to Plan mode (Tab) before complex changes
3. **Review Changes**: Always review what OpenCode suggests
4. **Iterate**: Don't be afraid to ask for refinements

## Example Workflow

Here's a typical workflow using OpenCode with LlamaCPP Studio:

### 1. Start Your Project

```bash
cd ~/projects/my-new-app
opencode
```

### 2. Initialize OpenCode

```bash
/init
```

This analyzes your project structure and creates an `AGENTS.md` file.

### 3. Configure Model

```bash
/models
```

Select **LlamaCPP Studio** → **GLM-4.7 Flash (Local)**

### 4. Ask for Help

```
I need to create a REST API endpoint for user registration. Look at how authentication is implemented in @src/api/auth.ts and create a similar endpoint in @src/api/users.ts
```

### 5. Review and Apply

```bash
# Toggle to Plan mode
<TAB>

# Review the suggested implementation

# Toggle back to Build mode
<TAB>

# Ask to implement
Sounds good! Make the changes.
```

## Monitoring and Optimization

While OpenCode is working, you can monitor your server's performance:

```bash
# In another terminal, run llama-run monitoring
./llama-run run --port 11433

# Or check basic stats
./llama-run --version
```

## Advanced Configuration

### Using a Different Port

If your llama.cpp server runs on a different port:

```json
{
  "provider": {
    "llamacpp-studio": {
      "options": {
        "baseURL": "http://127.0.0.1:PORT/v1"
      }
    }
  }
}
```

### Different Context Sizes

For projects requiring less context:

```json
{
  "models": {
    "glm-4.7-flash-ud-q4-k-xl": {
      "limit": {
        "context": 80000,
        "output": 32768
      }
    }
  }
}
```

### Multiple Models

If you have multiple models configured:

```json
{
  "provider": {
    "llamacpp-studio": {
      "models": {
        "glm-4.7-flash-ud-q4-k-xl": {
          "name": "GLM-4.7 Flash (Local)",
          "limit": { "context": 200000, "output": 65536 }
        },
        "small-model": {
          "name": "Small Model (Fast)",
          "limit": { "context": 8000, "output": 2048 }
        }
      }
    }
  }
}
```

## Getting Started Fast

If you want to get started quickly:

1. **Ensure server is running**:
   ```bash
   ./scripts/start_server.sh
   ```

2. **Create opencode.json** in your project root with the configuration above

3. **Run OpenCode**:
   ```bash
   cd your-project
   opencode
   ```

4. **Select model**: `/models` → LlamaCPP Studio → GLM-4.7 Flash

5. **Start coding**: Ask OpenCode to help with your tasks!

## Additional Resources

- **[OpenCode Documentation](https://opencode.ai/docs)** - Official OpenCode guides
- **[LlamaCPP Studio README](../README.md)** - Project overview and features
- **[GLM-4.7 Setup Guide](GLM-4.7_SETUP.md)** - Detailed server configuration

---

**Enjoy coding with OpenCode and your local GLM-4.7 Flash model!** 🚀
