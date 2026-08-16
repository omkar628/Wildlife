/**
 * Start the local FastAPI backend, or attach if it is already running.
 */
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const HEALTH = "http://127.0.0.1:8000/api/health";

function pythonExecutable() {
  const venv = path.join(ROOT, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(venv)) {
    return venv;
  }
  return "python";
}

async function backendUp() {
  try {
    const response = await fetch(HEALTH);
    return response.ok;
  } catch {
    return false;
  }
}

async function main() {
  if (await backendUp()) {
    console.log("Backend already running on http://127.0.0.1:8000");
    setInterval(() => {}, 1 << 30);
    return;
  }

  const python = pythonExecutable();
  console.log(`Starting FastAPI with ${python}`);
  const child = spawn(
    python,
    ["-m", "uvicorn", "backend.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
    {
      cwd: ROOT,
      stdio: "inherit",
      windowsHide: true,
    },
  );
  child.on("exit", (code) => {
    process.exit(code ?? 0);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
