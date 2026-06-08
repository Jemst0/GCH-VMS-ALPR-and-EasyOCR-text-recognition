const http = require("http");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const port = Number(process.env.PORT || 4173);

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
};

function sendFile(res, filePath) {
  const extension = path.extname(filePath).toLowerCase();
  const contentType = mimeTypes[extension] || "application/octet-stream";

  fs.readFile(filePath, (error, data) => {
    if (error) {
      res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Internal server error");
      return;
    }

    res.writeHead(200, { "Content-Type": contentType });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const requestPath = new URL(req.url, `http://${req.headers.host}`).pathname;
  const safePath = path.normalize(decodeURIComponent(requestPath)).replace(/^([.]{2}[\/\\])+/, "");
  let filePath = path.join(root, safePath);

  if (requestPath === "/") {
    filePath = path.join(root, "index.html");
  }

  fs.stat(filePath, (error, stats) => {
    if (!error && stats.isDirectory()) {
      sendFile(res, path.join(filePath, "index.html"));
      return;
    }

    if (error) {
      const fallbackPath = path.join(root, "index.html");
      if (fs.existsSync(fallbackPath)) {
        sendFile(res, fallbackPath);
        return;
      }

      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not found");
      return;
    }

    sendFile(res, filePath);
  });
});

server.listen(port, () => {
  console.log(`ALPR Vision Studio running at http://localhost:${port}`);
});