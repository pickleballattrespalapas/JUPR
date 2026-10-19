/** Browser-only visual export. Loaded on demand so ordinary recap reads stay light. */
type Region = { top: number; bottom: number };

export function recapPageBreaks(
  height: number,
  pageHeight: number,
  regions: Region[],
): Region[] {
  if (
    !Number.isFinite(height) ||
    !Number.isFinite(pageHeight) ||
    height <= 0 ||
    pageHeight <= 0
  )
    return [];
  const pages: Region[] = [];
  let top = 0;
  while (top < height) {
    let bottom = Math.min(height, top + pageHeight);
    if (bottom < height) {
      const crossing = regions.filter(
        (r) =>
          r.top > top + 24 &&
          r.bottom - r.top <= pageHeight &&
          r.top < bottom &&
          r.bottom > bottom,
      );
      if (crossing.length)
        bottom = Math.floor(Math.min(...crossing.map((r) => r.top)));
    }
    pages.push({ top, bottom });
    top = bottom;
  }
  return pages;
}

export function downloadRecapBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function asDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("A photo could not be read."));
    reader.readAsDataURL(blob);
  });
}

export async function exportWeeklyRecap(
  html: string,
  format: "png" | "pdf",
  filename: string,
) {
  const frame = document.createElement("iframe");
  frame.setAttribute("sandbox", "allow-same-origin");
  frame.style.cssText =
    "position:fixed;left:-10000px;top:0;width:636px;height:1px;border:0;";
  frame.title = "Preparing weekly recap download";
  let timeout: number | undefined;
  try {
    const loaded = new Promise<void>((resolve, reject) => {
      timeout = window.setTimeout(
        () =>
          reject(
            new Error(
              "The export timed out. Try again or use Print-friendly view.",
            ),
          ),
        15000,
      );
      frame.onload = () => {
        window.clearTimeout(timeout);
        resolve();
      };
    });
    frame.srcdoc = html;
    document.body.appendChild(frame);
    await loaded;
    const target = frame.contentDocument?.body;
    if (!target) throw new Error("The export preview is unavailable.");
    // Fetch every photo explicitly: never silently download a PDF with missing images.
    await Promise.all(
      Array.from(target.querySelectorAll("img")).map(async (img) => {
        try {
          if (!img.src.startsWith("data:")) {
            const controller = new AbortController();
            const timer = window.setTimeout(() => controller.abort(), 15000);
            try {
              const response = await fetch(img.src, {
                credentials: "omit",
                mode: "cors",
                signal: controller.signal,
              });
              if (!response.ok) throw new Error("Photo unavailable");
              const blob = await response.blob();
              if (
                !blob.type.startsWith("image/") ||
                blob.size > 12 * 1024 * 1024
              )
                throw new Error("Photo unavailable");
              img.src = await asDataUrl(blob);
            } finally {
              window.clearTimeout(timer);
            }
          }
          await img.decode();
          if (!img.naturalWidth) throw new Error("Photo unavailable");
        } catch {
          throw new Error(
            "A photo could not be included. Check its URL and sharing permissions, or use Print-friendly view to save a PDF.",
          );
        }
      }),
    );
    const height = Math.ceil(target.scrollHeight);
    if (height > 16000)
      throw new Error(
        "This recap is too long for an image export. Use Print-friendly view to save it as a PDF.",
      );
    const origin = target.getBoundingClientRect();
    const regions = Array.from(
      target.querySelectorAll("[data-recap-item], [data-recap-block]"),
    ).map((el) => {
      const rect = el.getBoundingClientRect();
      // Keep a section heading with its first card, without forcing the
      // entire section onto a fresh page and leaving large blank areas.
      const firstCard = el.hasAttribute("data-recap-block")
        ? el.querySelector("[data-recap-item]")?.getBoundingClientRect()
        : undefined;
      return {
        top: rect.top - origin.top,
        bottom: (firstCard?.bottom ?? rect.bottom) - origin.top,
      };
    });
    const links = Array.from(target.querySelectorAll("a[href]")).flatMap(
      (a) => {
        const href = a.getAttribute("href") || "";
        return /^https?:\/\//i.test(href)
          ? Array.from(a.getClientRects()).map((rect) => ({
              href,
              top: rect.top - origin.top,
              bottom: rect.bottom - origin.top,
              left: rect.left - origin.left,
              width: rect.width,
            }))
          : [];
      },
    );
    const { toCanvas } = await import("html-to-image");
    const canvas = await toCanvas(target, {
      pixelRatio: 2,
      backgroundColor: "#f4f0e8",
      width: 636,
      height,
      skipFonts: true,
    });
    if (format === "png") {
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (value) =>
            value
              ? resolve(value)
              : reject(new Error("The image could not be created.")),
          "image/png",
        ),
      );
      downloadRecapBlob(blob, `${filename}.png`);
      return;
    }
    const { jsPDF } = await import("jspdf");
    const pdf = new jsPDF({ unit: "mm", format: "a4", compress: true });
    const width = 190,
      scale = width / 636,
      pageHeight = 277 / scale;
    const pages = recapPageBreaks(height, pageHeight, regions);
    for (const [index, page] of pages.entries()) {
      if (index) pdf.addPage();
      const slice = document.createElement("canvas");
      const pixelTop = Math.round(page.top * 2),
        pixelBottom = Math.round(page.bottom * 2);
      slice.width = canvas.width;
      slice.height = pixelBottom - pixelTop;
      const ctx = slice.getContext("2d");
      if (!ctx) throw new Error("The PDF could not be created.");
      ctx.drawImage(
        canvas,
        0,
        pixelTop,
        canvas.width,
        slice.height,
        0,
        0,
        slice.width,
        slice.height,
      );
      pdf.addImage(
        slice,
        "PNG",
        10,
        10,
        width,
        (page.bottom - page.top) * scale,
      );
      for (const link of links) {
        const top = Math.max(link.top, page.top),
          bottom = Math.min(link.bottom, page.bottom);
        if (bottom > top)
          pdf.link(
            10 + link.left * scale,
            10 + (top - page.top) * scale,
            link.width * scale,
            (bottom - top) * scale,
            { url: link.href },
          );
      }
      slice.width = slice.height = 0;
    }
    downloadRecapBlob(pdf.output("blob"), `${filename}.pdf`);
    canvas.width = canvas.height = 0;
  } finally {
    window.clearTimeout(timeout);
    frame.remove();
  }
}
