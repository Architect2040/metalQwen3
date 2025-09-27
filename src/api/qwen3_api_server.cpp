#include <iostream>
#include <string>
#include <memory>
#include <signal.h>
#include <httplib.h>
#include <fmt/format.h>
#include <fmt/color.h>

#include "MetalContext.h"
#include "Qwen3ApiHandler.h"

// Global server pointer for signal handling
std::unique_ptr<httplib::Server> g_server;

void signalHandler(int signum) {
    fmt::print(fmt::fg(fmt::color::yellow), "\n🛑 Shutting down Qwen3 API server...\n");
    if (g_server) {
        g_server->stop();
    }
    exit(signum);
}

void printUsage(const char* program_name) {
    fmt::print(fmt::fg(fmt::color::cyan) | fmt::emphasis::bold, "Qwen3 Metal API Server - OpenAI Compatible\n");
    fmt::print("Usage: {} <model_file> [options]\n\n", program_name);

    fmt::print(fmt::fg(fmt::color::green) | fmt::emphasis::bold, "Options:\n");
    fmt::print("  --host <address>     Server host (default: localhost)\n");
    fmt::print("  --port <port>        Server port (default: 8080)\n");
    fmt::print("  --threads <num>      Worker threads (default: 8)\n");
    fmt::print("  --help               Show this help message\n\n");

    fmt::print(fmt::fg(fmt::color::yellow) | fmt::emphasis::bold, "API Endpoints:\n");
    fmt::print("  POST /v1/chat/completions    OpenAI compatible chat completions\n");
    fmt::print("  POST /v1/completions         OpenAI compatible text completions\n");
    fmt::print("  POST /v1/embeddings          OpenAI compatible embeddings\n");
    fmt::print("  GET  /v1/models             List available models\n");
    fmt::print("  GET  /health                Health check endpoint\n\n");

    fmt::print(fmt::fg(fmt::color::magenta) | fmt::emphasis::bold, "Example Usage:\n");
    fmt::print("  {} qwen3-4B.bin --port 8080\n\n", program_name);

    fmt::print(fmt::fg(fmt::color::blue) | fmt::emphasis::bold, "Compatible with:\n");
    fmt::print("  - OpenAI Python client\n");
    fmt::print("  - prompt-test benchmark suite\n");
    fmt::print("  - Any OpenAI API compatible tool\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string host = "localhost";
    int port = 8080;
    int threads = 8;

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Set up signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

        fmt::print(fmt::fg(fmt::color::cyan) | fmt::emphasis::bold, "\n=== Qwen3 Metal API Server ===\n");
    fmt::print("Model: {}\n", model_path);
    fmt::print("Host: {}\n", host);
    fmt::print("Port: {}\n", port);
    fmt::print("Threads: {}\n\n", threads);

    try {
        // Initialize Metal context
        fmt::print("🔧 Initializing Metal context...\n");
        MetalContext metalContext;
        if (!metalContext.initialize()) {
            fmt::print(fmt::fg(fmt::color::red), "❌ Failed to initialize Metal context\n");
            return 1;
        }

        // Initialize API handler
        fmt::print("📦 Loading Qwen3 model...\n");
        Qwen3ApiHandler apiHandler(metalContext);
        if (!apiHandler.initialize(model_path)) {
            fmt::print(fmt::fg(fmt::color::red), "❌ Failed to initialize API handler with model: {}\n", model_path);
            return 1;
        }

        // Create HTTP server
        g_server = std::make_unique<httplib::Server>();
        auto& server = *g_server;

        // Configure server (httplib defaults should be fine for testing)

        // API endpoints
        server.Post("/v1/chat/completions", [&apiHandler](const httplib::Request& req, httplib::Response& res) {
            apiHandler.handleChatCompletions(req, res);
        });

        server.Post("/v1/completions", [&apiHandler](const httplib::Request& req, httplib::Response& res) {
            apiHandler.handleCompletions(req, res);
        });

        server.Post("/v1/embeddings", [&apiHandler](const httplib::Request& req, httplib::Response& res) {
            apiHandler.handleEmbeddings(req, res);
        });

        server.Get("/v1/models", [&apiHandler](const httplib::Request& req, httplib::Response& res) {
            apiHandler.handleModels(req, res);
        });

        server.Get("/health", [&apiHandler](const httplib::Request& req, httplib::Response& res) {
            apiHandler.handleHealth(req, res);
        });

        // CORS preflight handler
        server.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.status = 200;
        });

        // Root endpoint with API information
        server.Get("/", [&](const httplib::Request&, httplib::Response& res) {
            std::ostringstream html;
            html << "<html><head><title>Qwen3 Metal API Server</title></head><body>";
            html << "<h1>Qwen3 Metal API Server</h1>";
            html << "<h2>Status: Running</h2>";
            html << "<p><strong>Model:</strong> " << model_path << "</p>";
            html << "<h2>API Endpoints:</h2>";
            html << "<ul>";
            html << "<li><code>POST /v1/chat/completions</code> - OpenAI compatible chat completions</li>";
            html << "<li><code>POST /v1/completions</code> - OpenAI compatible text completions</li>";
            html << "<li><code>POST /v1/embeddings</code> - OpenAI compatible embeddings</li>";
            html << "<li><code>GET /v1/models</code> - List available models</li>";
            html << "<li><code>GET /health</code> - Health check</li>";
            html << "</ul>";
            html << "<h2>Example cURL:</h2>";
            html << "<pre>";
            html << "curl -X POST http://" << host << ":" << port << "/v1/chat/completions \\\n";
            html << "  -H \"Content-Type: application/json\" \\\n";
            html << "  -d '{\n";
            html << "    \"model\": \"qwen3-metal\",\n";
            html << "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],\n";
            html << "    \"max_tokens\": 100\n";
            html << "  }'";
            html << "</pre>";
            html << "<p><a href=\"/health\">Health Check</a> | <a href=\"/v1/models\">Models</a></p>";
            html << "</body></html>";

            res.set_content(html.str(), "text/html");
        });

        // Error handler
        server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
            std::ostringstream error_html;
            error_html << "<html><body>";
            error_html << "<h1>Error " << res.status << "</h1>";
            error_html << "<p>The requested resource was not found.</p>";
            error_html << "<p><a href=\"/\">Back to main page</a></p>";
            error_html << "</body></html>";
            res.set_content(error_html.str(), "text/html");
        });

        // Start server
        fmt::print(fmt::fg(fmt::color::green) | fmt::emphasis::bold, "🚀 Starting server...\n");
        fmt::print(fmt::fg(fmt::color::blue), "🌐 API available at: http://{}:{}\n", host, port);
        fmt::print(fmt::fg(fmt::color::magenta), "📚 Documentation: http://{}:{}/\n", host, port);
        fmt::print(fmt::fg(fmt::color::yellow), "❤️  Health check: http://{}:{}/health\n", host, port);
        fmt::print("\n");
        fmt::print(fmt::fg(fmt::color::green) | fmt::emphasis::bold, "✅ Ready for OpenAI-compatible requests!\n");
        fmt::print("Press Ctrl+C to stop the server.\n\n");

        // This will block until server is stopped
        if (!server.listen(host, port)) {
            fmt::print(fmt::fg(fmt::color::red), "❌ Failed to start server on {}:{}\n", host, port);
            return 1;
        }

    } catch (const std::exception& e) {
        fmt::print(fmt::fg(fmt::color::red), "❌ Error: {}\n", e.what());
        return 1;
    } catch (...) {
        fmt::print(fmt::fg(fmt::color::red), "❌ Unknown error occurred\n");
        return 1;
    }

    return 0;
}
