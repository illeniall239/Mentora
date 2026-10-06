"use client";
import { useEffect, useRef, useState } from "react";
import { Excalidraw, getSceneVersion, serializeAsJSON } from "@excalidraw/excalidraw";
import type { ExcalidrawInitialDataState } from "@excalidraw/excalidraw/types";
import "@excalidraw/excalidraw/index.css";
import { saveSketch } from "@/app/actions";

// Loaded only in the browser (see Workspace): Excalidraw needs `window`.
export default function SketchPad({ exerciseId, initialScene }: { exerciseId: string; initialScene: string | null }) {
  const [saved, setSaved] = useState<"saved" | "saving" | "unsaved">("saved");
  const lastVersion = useRef<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    return () => clearTimeout(timer.current);
  }, []);

  const initialData: ExcalidrawInitialDataState | null = (() => {
    if (!initialScene) return null;
    try {
      const scene = JSON.parse(initialScene);
      return { elements: scene.elements ?? [], files: scene.files ?? {}, appState: { viewBackgroundColor: scene.appState?.viewBackgroundColor } };
    } catch {
      return null;
    }
  })();

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex items-center justify-between text-xs text-muted">
        <span>Boxes for variables, arrows for pointers and links, one frame per step. Saved with this exercise.</span>
        <span aria-live="polite">{saved === "saving" ? "Saving…" : saved === "unsaved" ? "Unsaved changes" : "Saved"}</span>
      </div>
      <div className="panel min-h-[480px] flex-1 overflow-hidden">
        <Excalidraw
          initialData={initialData}
          theme="light"
          UIOptions={{ canvasActions: { loadScene: false, saveToActiveFile: false, export: { saveFileToDisk: true } } }}
          onChange={(elements, appState, files) => {
            const version = getSceneVersion(elements);
            if (lastVersion.current === null) {
              lastVersion.current = version; // first render of the stored scene: nothing new to save
              return;
            }
            if (version === lastVersion.current) return; // only the view moved (scroll, zoom, selection)
            lastVersion.current = version;
            setSaved("unsaved");
            clearTimeout(timer.current);
            timer.current = setTimeout(async () => {
              setSaved("saving");
              await saveSketch(exerciseId, serializeAsJSON(elements, appState, files, "local"));
              setSaved("saved");
            }, 1200);
          }}
        />
      </div>
    </div>
  );
}
