import type { ImageMetadata } from "astro";
import { promises as fs } from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { pdf } from "pdf-to-img";
import sizeOf from "image-size";

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function sanitizeInputPath(inputFilePath: string) {
  let cleaned = inputFilePath.trim();

  if (cleaned.startsWith("file://")) {
    return fileURLToPath(new URL(cleaned));
  }

  cleaned = decodeURIComponent(cleaned.split("?")[0].split("#")[0]).replace(/\\/g, "/");

  if (cleaned.startsWith("/@fs/")) {
    cleaned = cleaned.slice(4);
  }

  return cleaned;
}

async function resolvePdfPath(inputFilePath: string) {
  const cleaned = sanitizeInputPath(inputFilePath);
  const candidates = new Set<string>();

  const addCandidate = (...segments: string[]) => {
    candidates.add(path.resolve(process.cwd(), ...segments));
  };

  if (cleaned.includes("/_astro/")) {
    throw new Error(
      [
        `renderPDF() received a built asset URL: ${inputFilePath}`,
        `Please pass the source PDF path string instead, e.g. "./assets/paper_material/xxx.pdf".`,
        `If you want the "Open figure" button, pass the imported PDF URL to href separately.`,
      ].join("\n"),
    );
  }

  if (path.isAbsolute(cleaned)) {
    candidates.add(cleaned);
  }

  const srcAssetsIndex = cleaned.indexOf("/src/assets/");
  if (srcAssetsIndex !== -1) {
    const relativeFromRoot = cleaned.slice(srcAssetsIndex + 1);
    addCandidate(relativeFromRoot);
  }

  if (cleaned.startsWith("/src/")) {
    addCandidate(cleaned.slice(1));
  }

  if (cleaned.startsWith("src/")) {
    addCandidate(cleaned);
  }

  addCandidate("src", cleaned);
  addCandidate("src", cleaned.replace(/^\/+/, ""));

  addCandidate("src/pages", cleaned);
  addCandidate("src/pages", cleaned.replace(/^\/+/, ""));

  addCandidate(cleaned);
  addCandidate(cleaned.replace(/^\/+/, ""));

  for (const candidate of candidates) {
    if ((await exists(candidate)) && path.extname(candidate).toLowerCase() === ".pdf") {
      return candidate;
    }
  }

  throw new Error(
    [
      `Could not resolve PDF path: ${inputFilePath}`,
      "Tried the following locations:",
      ...Array.from(candidates).map((candidate) => `- ${candidate}`),
    ].join("\n"),
  );
}

function toViteFsUrl(filePath: string) {
  const normalized = filePath.split(path.sep).join("/");
  return normalized.startsWith("/") ? `/@fs${normalized}` : `/@fs/${normalized}`;
}

export async function renderPDF(inputFilePath: string): Promise<ImageMetadata> {
  const pdfPath = await resolvePdfPath(inputFilePath);
  const stat = await fs.stat(pdfPath);

  const cacheKey = crypto
    .createHash("sha1")
    .update(pdfPath)
    .update(String(stat.mtimeMs))
    .digest("hex")
    .slice(0, 12);

  const baseName = path.basename(pdfPath, path.extname(pdfPath));
  const fileName = `${baseName}.${cacheKey}.png`;

  const cacheDir = path.resolve(process.cwd(), ".astro", "pdf-cache");
  const cacheFilePath = path.join(cacheDir, fileName);

  await fs.mkdir(cacheDir, { recursive: true });

  if (!(await exists(cacheFilePath))) {
    const document = await pdf(pdfPath, { scale: 2 });
    const firstPageImage = await document.getPage(1);

    if (!firstPageImage) {
      throw new Error(`Failed to render first page of PDF: ${pdfPath}`);
    }

    await fs.writeFile(cacheFilePath, firstPageImage);
  }

  const outputBuffer = await fs.readFile(cacheFilePath);
  const dimensions = sizeOf(outputBuffer);

  if (!dimensions.width || !dimensions.height) {
    throw new Error(`Could not determine size for rendered PDF preview: ${pdfPath}`);
  }

  if (import.meta.env.PROD) {
    const outputDir = path.resolve(process.cwd(), "dist", "_astro");
    const outputFilePath = path.join(outputDir, fileName);

    await fs.mkdir(outputDir, { recursive: true });

    if (!(await exists(outputFilePath))) {
      await fs.copyFile(cacheFilePath, outputFilePath);
    }

    return {
      src: `_astro/${fileName}`,
      width: dimensions.width,
      height: dimensions.height,
      format: "png",
    };
  }

  return {
    src: toViteFsUrl(cacheFilePath),
    width: dimensions.width,
    height: dimensions.height,
    format: "png",
  };
}